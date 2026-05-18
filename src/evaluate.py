"""Evaluation: generate answers, compute exact-match accuracy and token-level F1."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.data import VQADataset, collate_fn
from src.modeling import build_model
from src.utils import exact_match, load_config, normalise_answer, token_f1


def evaluate(cfg):
    val_ds = VQADataset(cfg["val_file"], image_root=cfg.get("image_root", "."))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = build_model(cfg)

    ckpt_path = os.path.join(cfg["output_dir"], "checkpoints", "last.pt")
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No checkpoint found, evaluating with untrained model.")

    model.eval_mode()

    predictions = []
    total_em, total_f1, count = 0.0, 0.0, 0

    for batch in val_loader:
        pred_answers = model.generate(batch)
        for i, pred in enumerate(pred_answers):
            gold = batch["answers"][i]
            pred_norm = normalise_answer(pred)
            em = exact_match(pred_norm, gold)
            f1 = token_f1(pred_norm, gold)
            total_em += int(em)
            total_f1 += f1
            count += 1
            predictions.append({
                "image": batch["meta"][i]["image_path"],
                "question": batch["meta"][i]["question"],
                "gold_answer": gold,
                "pred_answer": pred_norm,
                "exact_match": em,
            })

    acc = total_em / max(count, 1)
    avg_f1 = total_f1 / max(count, 1)
    print(f"Exact-match accuracy: {acc:.4f}  ({int(total_em)}/{count})")
    print(f"Token-level F1:      {avg_f1:.4f}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    pred_path = os.path.join(cfg["output_dir"], "predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"Predictions saved to {pred_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
