"""Evaluation: generate answers, compute exact-match accuracy and token-level F1."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.data import MultilingualVQADataset, VQADataset, collate_fn
from src.modeling import build_model
from src.utils import exact_match, load_config, normalise_answer, token_f1


def build_eval_dataset(cfg):
    languages = cfg.get("languages")
    image_root = cfg.get("image_root", ".")

    if languages and cfg.get("backend") == "multilingual_blip2":
        return MultilingualVQADataset(
            base_jsonl=cfg["val_file"],
            image_root=image_root,
            languages=languages,
            split="val",
        )
    else:
        lang = cfg.get("lang", "en")
        return VQADataset(cfg["val_file"], image_root=image_root, lang=lang)


def evaluate(cfg):
    val_ds = build_eval_dataset(cfg)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn
    )

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
    per_lang_stats = {}

    for batch in val_loader:
        pred_answers = model.generate(batch)
        for i, pred in enumerate(pred_answers):
            gold = batch["answers"][i]
            lang = batch["langs"][i] if "langs" in batch else "en"
            pred_norm = normalise_answer(pred)
            em = exact_match(pred_norm, gold)
            f1 = token_f1(pred_norm, gold)
            total_em += int(em)
            total_f1 += f1
            count += 1

            if lang not in per_lang_stats:
                per_lang_stats[lang] = {"em": 0.0, "f1": 0.0, "count": 0}
            per_lang_stats[lang]["em"] += int(em)
            per_lang_stats[lang]["f1"] += f1
            per_lang_stats[lang]["count"] += 1

            predictions.append({
                "image": batch["meta"][i]["image_path"],
                "question": batch["meta"][i]["question"],
                "lang": lang,
                "gold_answer": gold,
                "pred_answer": pred_norm,
                "exact_match": em,
            })

    acc = total_em / max(count, 1)
    avg_f1 = total_f1 / max(count, 1)
    print(f"\n=== Overall ===")
    print(f"Exact-match accuracy: {acc:.4f}  ({int(total_em)}/{count})")
    print(f"Token-level F1:      {avg_f1:.4f}")

    for lang, stats in sorted(per_lang_stats.items()):
        lang_acc = stats["em"] / max(stats["count"], 1)
        lang_f1 = stats["f1"] / max(stats["count"], 1)
        print(f"\n=== {lang} ===")
        print(f"Exact-match accuracy: {lang_acc:.4f}  ({int(stats['em'])}/{stats['count']})")
        print(f"Token-level F1:      {lang_f1:.4f}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    pred_path = os.path.join(cfg["output_dir"], "predictions.jsonl")
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved to {pred_path}")

    scores = {
        "overall_em": acc,
        "overall_f1": avg_f1,
    }
    for lang, stats in per_lang_stats.items():
        scores[f"{lang}_em"] = stats["em"] / max(stats["count"], 1)
        scores[f"{lang}_f1"] = stats["f1"] / max(stats["count"], 1)

    score_path = os.path.join(cfg["output_dir"], "scores.json")
    with open(score_path, "w") as f:
        json.dump(scores, f, indent=4)
    print(f"Scores saved to {score_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
