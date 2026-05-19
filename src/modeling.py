"""VQA model wrappers: smoke (toy CNN+MLP), blip2, and multilingual-blip2 backends."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


SUPPORTED_LANGUAGES = ["en", "zh", "fr", "es", "de", "ja", "ko", "pt", "ar"]
LANG_TO_ID = {lang: i for i, lang in enumerate(SUPPORTED_LANGUAGES)}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class VQAModelWrapper(ABC):
    @abstractmethod
    def train_step(self, batch: Dict) -> torch.Tensor:
        """Return scalar loss for one training batch."""

    @abstractmethod
    def generate(self, batch: Dict) -> List[str]:
        """Return predicted answer strings."""

    @abstractmethod
    def state_dict(self) -> Dict:
        """Return saveable state."""

    @abstractmethod
    def load_state_dict(self, sd: Dict):
        """Restore from saved state."""

    @abstractmethod
    def parameters(self):
        """Trainable parameters."""

    @abstractmethod
    def train_mode(self):
        ...

    @abstractmethod
    def eval_mode(self):
        ...


# ---------------------------------------------------------------------------
# Smoke backend – tiny CNN classifier that maps images to answer tokens
# ---------------------------------------------------------------------------

class _SmokeCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


class SmokeModel(VQAModelWrapper):
    """Rule-aware toy model.

    Maintains a fixed answer vocabulary built from training data.
    Uses a tiny CNN to predict an answer index from the image.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.vocab: List[str] = []
        self._model = None
        self._loss_fn = nn.CrossEntropyLoss()

    def warmup_vocab(self, answers: List[str]):
        """Pre-build the full vocabulary so the model is initialized once."""
        self._ensure_model(answers)

    def _ensure_model(self, answers: List[str]):
        new_tokens = [a for a in answers if a not in self.vocab]
        if new_tokens:
            self.vocab.extend(new_tokens)
        if self._model is None or self._model.head.out_features != len(self.vocab):
            self._model = _SmokeCNN(max(len(self.vocab), 1)).to(self.device)

    def train_step(self, batch: Dict) -> torch.Tensor:
        self._ensure_model(batch["answers"])
        images = batch["images"].to(self.device)
        labels = torch.tensor(
            [self.vocab.index(a) if a in self.vocab else 0 for a in batch["answers"]],
            device=self.device,
        )
        logits = self._model(images)
        return self._loss_fn(logits, labels)

    def generate(self, batch: Dict) -> List[str]:
        if self._model is None or not self.vocab:
            return ["unknown"] * len(batch["prompts"])
        images = batch["images"].to(self.device)
        with torch.no_grad():
            logits = self._model(images)
            preds = logits.argmax(dim=-1).tolist()
        return [self.vocab[p] if p < len(self.vocab) else "unknown" for p in preds]

    def state_dict(self) -> Dict:
        sd = {}
        if self._model is not None:
            sd["model"] = self._model.state_dict()
        sd["vocab"] = self.vocab
        return sd

    def load_state_dict(self, sd: Dict):
        self.vocab = sd.get("vocab", [])
        if self.vocab:
            self._model = _SmokeCNN(len(self.vocab)).to(self.device)
            if "model" in sd:
                self._model.load_state_dict(sd["model"])

    def parameters(self):
        self._ensure_model(self.vocab or ["_pad"])
        return self._model.parameters()

    def train_mode(self):
        if self._model:
            self._model.train()

    def eval_mode(self):
        if self._model:
            self._model.eval()


# ---------------------------------------------------------------------------
# BLIP-2 backend (original, no multilingual)
# ---------------------------------------------------------------------------

class Blip2Model(VQAModelWrapper):
    """Wrapper around HuggingFace Blip2ForConditionalGeneration.

    By default freezes the vision encoder and language model, exposing only
    the Q-Former parameters for fine-tuning.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b",
                 device: str = "cpu", freeze_vision: bool = True,
                 freeze_lm: bool = True, max_answer_length: int = 10):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.device = device
        self.max_answer_length = max_answer_length
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(device)

        if freeze_vision:
            for p in self.model.vision_model.parameters():
                p.requires_grad = False
        if freeze_lm:
            for p in self.model.language_model.parameters():
                p.requires_grad = False

    def _prepare(self, batch: Dict):
        from torchvision.transforms.functional import to_pil_image
        pil_images = [to_pil_image(img) for img in batch["images"]]
        inputs = self.processor(
            images=pil_images, text=batch["prompts"],
            return_tensors="pt", padding=True,
        ).to(self.device)
        return inputs

    def train_step(self, batch: Dict) -> torch.Tensor:
        inputs = self._prepare(batch)
        labels = self.processor.tokenizer(
            batch["answers"], return_tensors="pt", padding=True,
        ).input_ids.to(self.device)
        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    def generate(self, batch: Dict) -> List[str]:
        inputs = self._prepare(batch)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_answer_length)
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    def state_dict(self) -> Dict:
        return {k: v for k, v in self.model.state_dict().items() if v.requires_grad
                } if False else self.model.state_dict()

    def load_state_dict(self, sd: Dict):
        self.model.load_state_dict(sd, strict=False)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


# ---------------------------------------------------------------------------
# Multilingual adaptation modules (Yellow/Gold blocks in pipeline)
# ---------------------------------------------------------------------------

class LanguageEmbedding(nn.Module):
    """Learnable embedding L_i for each language i in {en, zh, fr, es, ...}."""

    def __init__(self, num_languages: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_languages, embed_dim)

    def forward(self, lang_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(lang_ids)


class LanguagePrefixTokens(nn.Module):
    """Language-aware prefix tokens prepended to Q-Former query tokens.

    Each language gets num_prefix_tokens learnable vectors of dimension
    hidden_dim.  These are concatenated IN FRONT of the original query
    tokens before being fed into Q-Former, so cross-attention inside the
    Q-Former operates over [prefix | query] jointly.
    """

    def __init__(self, num_languages: int, num_prefix_tokens: int, hidden_dim: int):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_embeddings = nn.Embedding(
            num_languages * num_prefix_tokens, hidden_dim
        )
        self.num_languages = num_languages

    def forward(self, lang_ids: torch.Tensor) -> torch.Tensor:
        offsets = lang_ids * self.num_prefix_tokens
        indices = offsets.unsqueeze(1) + torch.arange(
            self.num_prefix_tokens, device=lang_ids.device
        ).unsqueeze(0)
        return self.prefix_embeddings(indices)  # (B, num_prefix, hidden)


class LanguageConditionedAdapter(nn.Module):
    """Language-conditioned LoRA-style adapter injected into Q-Former
    cross-attention layers.

    Applied to the cross-attention VALUE projection output:
        v' = v + scale * up(down(v) * gate(lang_emb))
    This conditions how visual information flows through cross-attention
    based on the target language.
    """

    def __init__(self, hidden_dim: int, lang_embed_dim: int, rank: int = 8,
                 alpha: float = 16.0):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.lang_gate = nn.Sequential(
            nn.Linear(lang_embed_dim, rank),
            nn.Sigmoid(),
        )
        self.scale = alpha / rank
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor,
                lang_emb: torch.Tensor) -> torch.Tensor:
        down = self.down(hidden_states)
        gate = self.lang_gate(lang_emb)
        if gate.dim() == 2:
            gate = gate.unsqueeze(1)
        adapted = self.up(down * gate) * self.scale
        return hidden_states + adapted


class _LoRALayer(nn.Module):
    """Standalone LoRA adapter for a single linear layer."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 16.0):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x)) * self.scale


class LanguageConditionedQFormerBridge(nn.Module):
    """The main trainable cross-modal bridge (green block in pipeline).

    Injects language conditioning INTO the Q-Former:
    1. Language prefix tokens: concatenated with query_tokens before Q-Former
       forward, so they participate in self-attention AND cross-attention
    2. Cross-attention adapters: language-gated LoRA on the value projection
       inside each Q-Former cross-attention layer (not post-hoc)
    3. Language embedding: L_i conditions the adapters via gating
    """

    def __init__(self, qformer_hidden_dim: int, num_languages: int,
                 num_prefix_tokens: int = 4, adapter_rank: int = 8,
                 adapter_alpha: float = 16.0,
                 num_cross_attn_layers: int = 6,
                 lang_embed_dim: int = 64):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.lang_embedding = LanguageEmbedding(num_languages, lang_embed_dim)
        self.prefix_tokens = LanguagePrefixTokens(
            num_languages, num_prefix_tokens, qformer_hidden_dim
        )
        # One adapter per cross-attention layer (not every Q-Former layer)
        self.cross_attn_adapters = nn.ModuleList([
            LanguageConditionedAdapter(
                qformer_hidden_dim, lang_embed_dim, adapter_rank, adapter_alpha
            )
            for _ in range(num_cross_attn_layers)
        ])

    def get_language_embeddings(self, lang_ids: torch.Tensor):
        return self.lang_embedding(lang_ids)

    def get_prefix_tokens(self, lang_ids: torch.Tensor):
        return self.prefix_tokens(lang_ids)


# ---------------------------------------------------------------------------
# Multilingual BLIP-2 backend (pipeline diagram implementation)
# ---------------------------------------------------------------------------

class MultilingualBlip2Model(VQAModelWrapper):
    """Language-conditioned BLIP-2 for multilingual medical VQA.

    Architecture (matching pipeline.png):
    - Vision Encoder (Gray): BLIP-2 ViT, frozen; optional LoRA on QKV
    - Language-Conditioned Q-Former (Green): trainable bridge with
      language prefix tokens concatenated to query tokens, and
      language-gated adapters INSIDE cross-attention value path
    - Language Model Decoder (Gray): frozen
    - Language Embeddings (Yellow): L_i for each language, gate adapters
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b",
                 device: str = "cpu",
                 freeze_vision: bool = True,
                 freeze_lm: bool = True,
                 max_answer_length: int = 10,
                 languages: Optional[List[str]] = None,
                 num_prefix_tokens: int = 4,
                 adapter_rank: int = 8,
                 adapter_alpha: float = 16.0,
                 lang_embed_dim: int = 64,
                 lora_vision: bool = False,
                 lora_vision_rank: int = 8):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.device = device
        self.max_answer_length = max_answer_length
        self.languages = languages or SUPPORTED_LANGUAGES
        self.lang_to_id = {lang: i for i, lang in enumerate(self.languages)}
        self.num_prefix_tokens = num_prefix_tokens

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(device)

        if freeze_vision:
            for p in self.model.vision_model.parameters():
                p.requires_grad = False
        if freeze_lm:
            for p in self.model.language_model.parameters():
                p.requires_grad = False

        qformer_cfg = self.model.config.qformer_config
        qformer_hidden_dim = qformer_cfg.hidden_size
        num_qformer_layers = qformer_cfg.num_hidden_layers
        cross_attn_freq = qformer_cfg.cross_attention_frequency
        num_cross_attn_layers = sum(
            1 for i in range(num_qformer_layers) if i % cross_attn_freq == 0
        )

        self.lang_bridge = LanguageConditionedQFormerBridge(
            qformer_hidden_dim=qformer_hidden_dim,
            num_languages=len(self.languages),
            num_prefix_tokens=num_prefix_tokens,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha,
            num_cross_attn_layers=num_cross_attn_layers,
            lang_embed_dim=lang_embed_dim,
        ).to(device)

        # Vision LoRA
        self.vision_loras = None
        if lora_vision:
            self._add_vision_lora(lora_vision_rank)

        self._hooks = []
        self._patch_query_token_expansion()
        self._patch_cross_attention_layers(cross_attn_freq, num_qformer_layers)
        if self.vision_loras is not None:
            self._patch_vision_self_attention()

        # Runtime state set per forward pass
        self._current_lang_emb = None
        self._current_prefix = None

    # ---- FIX 1: prefix tokens actually concatenated to query_tokens ----

    def _patch_query_token_expansion(self):
        """Monkey-patch the BLIP-2 forward to concatenate language prefix
        tokens with query_tokens before they enter the Q-Former.

        HuggingFace does: query_tokens = self.query_tokens.expand(B, -1, -1)
        We intercept the Q-Former call to prepend [prefix | query_tokens].
        """
        original_qformer_forward = self.model.qformer.forward
        wrapper = self

        def patched_qformer_forward(query_embeds, *args, **kwargs):
            if wrapper._current_prefix is not None:
                # query_embeds: (B, num_query, hidden)
                # prefix:       (B, num_prefix, hidden)
                query_embeds = torch.cat(
                    [wrapper._current_prefix, query_embeds], dim=1
                )
                # Update query_length so cross-attention applies to all query+prefix tokens
                if "query_length" in kwargs and kwargs["query_length"] is not None:
                    kwargs["query_length"] = (
                        kwargs["query_length"] + wrapper.num_prefix_tokens
                    )
            out = original_qformer_forward(query_embeds, *args, **kwargs)
            if wrapper._current_prefix is not None and out.last_hidden_state is not None:
                # Strip prefix tokens from output so downstream shape matches
                out.last_hidden_state = out.last_hidden_state[
                    :, wrapper.num_prefix_tokens:, :
                ]
            return out

        self.model.qformer.forward = patched_qformer_forward

    # ---- FIX 2: language conditioning INSIDE cross-attention ----

    def _patch_cross_attention_layers(self, cross_attn_freq: int,
                                      num_layers: int):
        """Patch the value projection inside each Q-Former cross-attention
        layer to apply a language-conditioned adapter.

        Instead of hooking the layer output, we wrap the cross-attention
        module's value Linear so the adaptation happens before attention
        scores are computed:
            v = original_value(encoder_hidden_states)
            v = v + adapter(v, lang_emb)
        """
        adapter_idx = 0
        for layer_i in range(num_layers):
            if layer_i % cross_attn_freq != 0:
                continue

            qformer_layer = self.model.qformer.encoder.layer[layer_i]
            cross_attn_mod = qformer_layer.crossattention.self
            original_value = cross_attn_mod.value
            adapter = self.lang_bridge.cross_attn_adapters[adapter_idx]
            wrapper = self

            def make_patched_value(orig_v, adpt):
                def patched_value(x):
                    out = orig_v(x)
                    if wrapper._current_lang_emb is not None:
                        out = adpt(out, wrapper._current_lang_emb)
                    return out
                return patched_value

            cross_attn_mod.value = make_patched_value(original_value, adapter)
            adapter_idx += 1

    # ---- FIX 3: Vision LoRA actually wired into ViT forward ----

    def _add_vision_lora(self, rank: int):
        """Create LoRA layers for each ViT encoder layer's fused QKV."""
        self.vision_loras = nn.ModuleList()
        for layer in self.model.vision_model.encoder.layers:
            qkv = layer.self_attn.qkv
            lora = _LoRALayer(qkv.in_features, qkv.out_features, rank)
            self.vision_loras.append(lora)
        self.vision_loras = self.vision_loras.to(self.device)

    def _patch_vision_self_attention(self):
        """Monkey-patch each ViT layer's QKV linear to add LoRA delta.

        Original: mixed_qkv = self.qkv(hidden_states)
        Patched:  mixed_qkv = self.qkv(hidden_states) + lora(hidden_states)
        """
        for layer_i, layer in enumerate(self.model.vision_model.encoder.layers):
            original_qkv = layer.self_attn.qkv
            lora = self.vision_loras[layer_i]

            def make_patched_qkv(orig_qkv, lora_layer):
                def patched_qkv(x):
                    return orig_qkv(x) + lora_layer(x)
                return patched_qkv

            layer.self_attn.qkv = make_patched_qkv(original_qkv, lora)

    # ---- Helpers ----

    def _resolve_lang_ids(self, batch: Dict) -> torch.Tensor:
        langs = batch.get("langs", ["en"] * len(batch["prompts"]))
        ids = [self.lang_to_id.get(l, 0) for l in langs]
        return torch.tensor(ids, device=self.device)

    def _prepare(self, batch: Dict):
        from torchvision.transforms.functional import to_pil_image
        pil_images = [to_pil_image(img) for img in batch["images"]]
        inputs = self.processor(
            images=pil_images, text=batch["prompts"],
            return_tensors="pt", padding=True,
        ).to(self.device)
        return inputs

    def _set_lang_context(self, lang_ids: torch.Tensor):
        self._current_lang_emb = self.lang_bridge.get_language_embeddings(lang_ids)
        self._current_prefix = self.lang_bridge.get_prefix_tokens(lang_ids)

    def _clear_lang_context(self):
        self._current_lang_emb = None
        self._current_prefix = None

    # ---- VQAModelWrapper interface ----

    def train_step(self, batch: Dict) -> torch.Tensor:
        lang_ids = self._resolve_lang_ids(batch)
        self._set_lang_context(lang_ids)

        inputs = self._prepare(batch)
        labels = self.processor.tokenizer(
            batch["answers"], return_tensors="pt", padding=True,
        ).input_ids.to(self.device)

        outputs = self.model(**inputs, labels=labels)
        self._clear_lang_context()
        return outputs.loss

    def generate(self, batch: Dict) -> List[str]:
        lang_ids = self._resolve_lang_ids(batch)
        self._set_lang_context(lang_ids)

        inputs = self._prepare(batch)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_answer_length)
        self._clear_lang_context()
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    def state_dict(self) -> Dict:
        sd = {}
        sd["blip2"] = self.model.state_dict()
        sd["lang_bridge"] = self.lang_bridge.state_dict()
        if self.vision_loras is not None:
            sd["vision_loras"] = self.vision_loras.state_dict()
        sd["languages"] = self.languages
        return sd

    def load_state_dict(self, sd: Dict):
        if "blip2" in sd:
            self.model.load_state_dict(sd["blip2"], strict=False)
        if "lang_bridge" in sd:
            self.lang_bridge.load_state_dict(sd["lang_bridge"])
        if "vision_loras" in sd and self.vision_loras is not None:
            self.vision_loras.load_state_dict(sd["vision_loras"])

    def parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        params += list(self.lang_bridge.parameters())
        if self.vision_loras is not None:
            params += list(self.vision_loras.parameters())
        return params

    def train_mode(self):
        self.model.train()
        self.lang_bridge.train()
        if self.vision_loras is not None:
            self.vision_loras.train()

    def eval_mode(self):
        self.model.eval()
        self.lang_bridge.eval()
        if self.vision_loras is not None:
            self.vision_loras.eval()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict) -> VQAModelWrapper:
    backend = cfg.get("backend", "smoke")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "smoke":
        return SmokeModel(device=device)
    elif backend == "blip2":
        return Blip2Model(
            model_name=cfg.get("model_name", "Salesforce/blip2-opt-2.7b"),
            device=device,
            freeze_vision=cfg.get("freeze_vision", True),
            freeze_lm=cfg.get("freeze_lm", True),
            max_answer_length=cfg.get("max_answer_length", 10),
        )
    elif backend == "multilingual_blip2":
        return MultilingualBlip2Model(
            model_name=cfg.get("model_name", "Salesforce/blip2-opt-2.7b"),
            device=device,
            freeze_vision=cfg.get("freeze_vision", True),
            freeze_lm=cfg.get("freeze_lm", True),
            max_answer_length=cfg.get("max_answer_length", 10),
            languages=cfg.get("languages", SUPPORTED_LANGUAGES),
            num_prefix_tokens=cfg.get("num_prefix_tokens", 4),
            adapter_rank=cfg.get("adapter_rank", 8),
            adapter_alpha=cfg.get("adapter_alpha", 16.0),
            lang_embed_dim=cfg.get("lang_embed_dim", 64),
            lora_vision=cfg.get("lora_vision", False),
            lora_vision_rank=cfg.get("lora_vision_rank", 8),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
