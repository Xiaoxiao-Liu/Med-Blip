#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "===== Multilingual Med-BLIP Training ====="
echo "Config: configs/multilingual.yaml"
echo ""

echo "===== Step 1: Train (multilingual_blip2 backend) ====="
python src/train.py --config configs/multilingual.yaml

echo ""
echo "===== Step 2: Evaluate (all languages) ====="
python src/evaluate.py --config configs/multilingual.yaml

echo ""
echo "===== Step 3: Per-language inference examples ====="

echo "--- English ---"
python src/infer.py \
  --config configs/multilingual.yaml \
  --image data/demo/images/sample_001.png \
  --question "What color is the object?" \
  --lang en

echo ""
echo "--- Chinese ---"
python src/infer.py \
  --config configs/multilingual.yaml \
  --image data/demo/images/sample_001.png \
  --question "这个物体是什么颜色的？" \
  --lang zh

echo ""
echo "--- French ---"
python src/infer.py \
  --config configs/multilingual.yaml \
  --image data/demo/images/sample_001.png \
  --question "Quelle est la couleur de l'objet ?" \
  --lang fr

echo ""
echo "--- Spanish ---"
python src/infer.py \
  --config configs/multilingual.yaml \
  --image data/demo/images/sample_001.png \
  --question "¿De qué color es el objeto?" \
  --lang es

echo ""
echo "===== Multilingual pipeline complete ====="
