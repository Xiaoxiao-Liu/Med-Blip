#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "===== Step 1: Generate demo dataset ====="
python tools/create_demo_dataset.py

echo ""
echo "===== Step 2: Train (smoke backend) ====="
python src/train.py --config configs/demo.yaml

echo ""
echo "===== Step 3: Evaluate ====="
python src/evaluate.py --config configs/demo.yaml

echo ""
echo "===== Step 4: Single-sample inference ====="
python src/infer.py \
  --config configs/demo.yaml \
  --image data/demo/images/sample_001.png \
  --question "What color is the object?"

echo ""
echo "===== Demo complete ====="
