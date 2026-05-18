#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "===== MedCon Evaluation (English) ====="
python src/evaluate_medcon.py --config configs/medcon_en.yaml

echo ""
echo "===== MedCon Evaluation (Chinese) ====="
python src/evaluate_medcon.py --config configs/medcon_zh.yaml

echo ""
echo "===== MedCon Evaluation Complete ====="
