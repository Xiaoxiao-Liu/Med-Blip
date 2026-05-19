"""Single-sample inference: given an image and question, produce an answer."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
from torchvision import transforms

from src.data import format_prompt
from src.modeling import build_model
from src.utils import load_config, normalise_answer

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def infer(cfg, image_path: str, question: str, lang: str = "en"):
    model = build_model(cfg)

    ckpt_path = os.path.join(cfg["output_dir"], "checkpoints", "last.pt")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No checkpoint found, running with untrained model.")

    model.eval_mode()

    image = Image.open(image_path).convert("RGB")
    image_tensor = TRANSFORM(image).unsqueeze(0)
    prompt = format_prompt(question, lang=lang)

    batch = {
        "images": image_tensor,
        "prompts": [prompt],
        "answers": [""],
        "langs": [lang],
    }

    preds = model.generate(batch)
    answer = normalise_answer(preds[0])

    print(f"Image:    {image_path}")
    print(f"Question: {question}")
    print(f"Language: {lang}")
    print(f"Answer:   {answer}")
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--lang", default="en",
                        help="Language code (en, zh, fr, es, ...)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    infer(cfg, args.image, args.question, lang=args.lang)


if __name__ == "__main__":
    main()
