# Vision-Language Model Adaptation for Image Question Answering

A lightweight research demo that illustrates how to adapt a BLIP-2-style vision-language model for image-question-answer generation. The project implements a complete multimodal pipeline covering image-text data formatting, instruction prompt construction, visual feature preparation, answer normalisation, and shared-task-style evaluation.

## Key Features

- **End-to-end pipeline** – data generation, training, evaluation, and inference in one repository.
- **Two backends** – a fast *smoke* backend (tiny CNN) for instant demo runs, and a *blip2* backend that wraps HuggingFace `Blip2ForConditionalGeneration` for real experiments.
- **Synthetic dataset generator** – creates colored-shape images with QA pairs, no external data download needed.
- **Instruction-style prompts** – formats every sample as `"Question: {q} Answer:"`.
- **Answer normalisation** – lowercase, strip punctuation, collapse whitespace.
- **Standard metrics** – exact-match accuracy and token-level F1.
- **YAML config** – switch backends, hyperparameters, and paths without touching code.

## Repository Structure

```
vlm-vqa-adaptation/
├── README.md
├── requirements.txt
├── configs/
│   ├── demo.yaml          # smoke backend (default)
│   └── blip2.yaml         # real BLIP-2 backend
├── data/
│   └── demo/
│       ├── images/
│       ├── train.jsonl
│       └── val.jsonl
├── outputs/
│   └── .gitkeep
├── scripts/
│   └── run_demo.sh        # one-click demo
├── src/
│   ├── __init__.py
│   ├── data.py            # dataset & prompt formatting
│   ├── modeling.py        # model wrappers (smoke / blip2)
│   ├── train.py           # training loop
│   ├── evaluate.py        # evaluation & metrics
│   ├── infer.py           # single-sample inference
│   └── utils.py           # config, normalisation, metrics
└── tools/
    └── create_demo_dataset.py
```

## Installation

```bash
git clone https://github.com/<your-username>/vlm-vqa-adaptation.git
cd vlm-vqa-adaptation
pip install -r requirements.txt
```

Python 3.9+ is recommended. The default demo requires only CPU.

## Quick Start

Run the full pipeline (data generation → training → evaluation → inference) with one command:

```bash
bash scripts/run_demo.sh
```

This uses the **smoke** backend—a tiny CNN classifier that trains in seconds.

## Dataset Format

Each sample is a JSON line in `.jsonl` files:

```json
{
  "image": "data/demo/images/sample_001.png",
  "question": "What color is the object?",
  "answer": "red"
}
```

Generate the demo dataset:

```bash
python tools/create_demo_dataset.py
```

This creates ~72 samples (colored shapes × question types) split into `train.jsonl` and `val.jsonl`.

## Training

```bash
python src/train.py --config configs/demo.yaml
```

The training loop prints per-epoch train/val loss and saves a checkpoint to `outputs/checkpoints/last.pt`.

## Evaluation

```bash
python src/evaluate.py --config configs/demo.yaml
```

Computes exact-match accuracy and token-level F1, and writes per-sample predictions to `outputs/predictions.jsonl`:

```json
{
  "image": "data/demo/images/sample_005.png",
  "question": "What shape is shown?",
  "gold_answer": "circle",
  "pred_answer": "circle",
  "exact_match": true
}
```

## Inference

```bash
python src/infer.py \
  --config configs/demo.yaml \
  --image data/demo/images/sample_001.png \
  --question "What color is the object?"
```

## How to Switch to the BLIP-2 Backend

If you have a GPU with sufficient memory (~10 GB for `blip2-opt-2.7b`):

```bash
# Install accelerate (optional, for mixed-precision)
pip install accelerate

# Run with the BLIP-2 config
python src/train.py --config configs/blip2.yaml
python src/evaluate.py --config configs/blip2.yaml
```

`configs/blip2.yaml` uses `Salesforce/blip2-opt-2.7b` with frozen vision encoder and language model—only the Q-Former is fine-tuned.

## Example Results (smoke backend)

After 3 epochs on the synthetic dataset:

| Metric | Score |
|---|---|
| Exact-match accuracy | ~0.25–0.40 |
| Token-level F1 | ~0.45–0.60 |

Exact numbers vary by run. The smoke model is a tiny CNN that learns to associate image pixel statistics with answer classes. It demonstrates the pipeline rather than targeting high accuracy.

## Research Motivation

Model performance on vision-language tasks does not depend solely on the backbone architecture. It also depends on:

- **Input format alignment** – whether the image preprocessing and prompt template match what the pre-trained model expects.
- **Answer space definition** – how the target vocabulary is constructed and constrained.
- **Prompt engineering** – the instruction template that bridges the gap between pre-training and the downstream task.
- **Evaluation protocol** – whether answer normalisation and metrics are consistent across experiments.

This project provides a minimal, reproducible pipeline for studying these adaptation factors. By isolating data formatting, prompt construction, and evaluation from the model itself, it becomes straightforward to measure how each component affects downstream accuracy—independent of the backbone's raw capability.

## License

MIT
