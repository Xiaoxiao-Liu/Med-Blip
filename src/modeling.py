"""VQA model wrappers: smoke (toy CNN+MLP) and blip2 backends."""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn


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
# BLIP-2 backend
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
    else:
        raise ValueError(f"Unknown backend: {backend}")
