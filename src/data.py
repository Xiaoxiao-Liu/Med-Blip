"""Dataset and prompt formatting for image-QA and MedCon NLG tasks."""

import json
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import normalise_answer

PROMPT_TEMPLATE = "Question: {question} Answer:"

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


class VQADataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str = ".", transform=None):
        self.samples = load_jsonl(jsonl_path)
        self.image_root = image_root
        self.transform = transform or DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.image_root, item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        prompt = format_prompt(item["question"])
        answer = normalise_answer(item["answer"])
        return {
            "image": image,
            "prompt": prompt,
            "answer": answer,
            "image_path": item["image"],
            "question": item["question"],
            "raw_answer": item["answer"],
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    prompts = [b["prompt"] for b in batch]
    answers = [b["answer"] for b in batch]
    meta = [{k: b[k] for k in ("image_path", "question", "raw_answer")} for b in batch]
    return {"images": images, "prompts": prompts, "answers": answers, "meta": meta}


# ---------------------------------------------------------------------------
# MedCon dataset — encounter-based medical NLG
# ---------------------------------------------------------------------------

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class MedConDataset(Dataset):
    """Loads encounter-based medical conversation data.

    Each sample is one encounter with ``encounter_id`` and a list of
    ``responses``.  For references the list may contain multiple annotator
    answers; for predictions it typically contains one.
    """

    def __init__(self, json_path: str, lang: str = "en",
                 image_root: Optional[str] = None, transform=None):
        self.data = load_json(json_path)
        self.lang = lang
        self.content_key = f"content_{lang}"
        self.image_root = image_root
        self.transform = transform or DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encounter_id = item["encounter_id"]
        responses = item["responses"]

        texts = [r[self.content_key] for r in responses if self.content_key in r]
        prompt = f"Medical encounter {encounter_id}: provide a medical response."

        result = {
            "encounter_id": encounter_id,
            "prompt": prompt,
            "texts": texts,
            "responses_raw": responses,
        }

        if self.image_root and "image" in item:
            img_path = os.path.join(self.image_root, item["image"])
            image = Image.open(img_path).convert("RGB")
            result["image"] = self.transform(image)

        return result


def medcon_collate_fn(batch):
    encounter_ids = [b["encounter_id"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    all_texts = [b["texts"] for b in batch]
    all_responses = [b["responses_raw"] for b in batch]

    result = {
        "encounter_ids": encounter_ids,
        "prompts": prompts,
        "texts": all_texts,
        "responses_raw": all_responses,
    }

    has_images = all("image" in b for b in batch)
    if has_images:
        result["images"] = torch.stack([b["image"] for b in batch])

    return result
