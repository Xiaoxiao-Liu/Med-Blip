"""Generate a synthetic image-QA demo dataset with colored geometric shapes."""

import json
import os
import random

from PIL import Image, ImageDraw

SHAPES = {
    "circle": lambda draw, bbox, color: draw.ellipse(bbox, fill=color),
    "square": lambda draw, bbox, color: draw.rectangle(bbox, fill=color),
    "triangle": lambda draw, bbox, color: draw.polygon(
        [(bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3])],
        fill=color,
    ),
}

COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 50, 220),
    "green": (50, 180, 50),
    "yellow": (220, 220, 50),
    "purple": (150, 50, 200),
    "orange": (240, 150, 30),
}

QUESTIONS = [
    ("What color is the object?", lambda c, s: c),
    ("What shape is shown?", lambda c, s: s),
    ("What color is the shape?", lambda c, s: c),
    ("Describe the shape.", lambda c, s: f"{c} {s}"),
]


def generate_image(color_name: str, shape_name: str, size: int = 128) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 20
    bbox = (margin, margin, size - margin, size - margin)
    SHAPES[shape_name](draw, bbox, COLORS[color_name])
    return img


def main():
    out_dir = os.path.join("data", "demo")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    samples = []
    idx = 0
    for color_name in COLORS:
        for shape_name in SHAPES:
            for question_template, answer_fn in QUESTIONS:
                fname = f"sample_{idx:03d}.png"
                fpath = os.path.join(img_dir, fname)
                img = generate_image(color_name, shape_name)
                img.save(fpath)
                samples.append({
                    "image": os.path.join("data", "demo", "images", fname),
                    "question": question_template,
                    "answer": answer_fn(color_name, shape_name),
                })
                idx += 1

    random.seed(42)
    random.shuffle(samples)
    split = max(1, int(len(samples) * 0.8))
    train_samples = samples[:split]
    val_samples = samples[split:]

    for name, data in [("train.jsonl", train_samples), ("val.jsonl", val_samples)]:
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(data)} samples to {path}")

    print(f"Generated {idx} images in {img_dir}")


if __name__ == "__main__":
    main()
