"""Training loop for VQA model adaptation."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.data import VQADataset, collate_fn
from src.modeling import build_model
from src.utils import load_config, normalise_answer


def train(cfg):
    train_ds = VQADataset(cfg["train_file"], image_root=cfg.get("image_root", "."))
    val_ds = VQADataset(cfg["val_file"], image_root=cfg.get("image_root", "."))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = build_model(cfg)

    all_answers = list({normalise_answer(s["answer"]) for s in train_ds.samples}
                       | {normalise_answer(s["answer"]) for s in val_ds.samples})
    if hasattr(model, "warmup_vocab"):
        model.warmup_vocab(all_answers)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train_mode()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.train_step(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg_train = total_loss / max(steps, 1)

        model.eval_mode()
        val_loss, val_steps = 0.0, 0
        for batch in val_loader:
            with torch.no_grad():
                loss = model.train_step(batch)
            val_loss += loss.item()
            val_steps += 1
        avg_val = val_loss / max(val_steps, 1)

        print(f"Epoch {epoch}/{cfg['num_epochs']}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

    ckpt_path = os.path.join(ckpt_dir, "last.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
