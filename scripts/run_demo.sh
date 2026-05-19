#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    echo "===== Demo: Smoke backend (toy model, no GPU needed) ====="
    echo ""

    echo "--- Step 1: Generate demo dataset ---"
    python tools/create_demo_dataset.py

    echo ""
    echo "--- Step 2: Train (smoke backend) ---"
    python src/train.py --config configs/demo.yaml

    echo ""
    echo "--- Step 3: Evaluate ---"
    python src/evaluate.py --config configs/demo.yaml

    echo ""
    echo "--- Step 4: Single-sample inference ---"
    python src/infer.py \
      --config configs/demo.yaml \
      --image data/demo/images/sample_001.png \
      --question "What color is the object?"
    ;;

  multilingual)
    echo "===== Demo: Multilingual Med-BLIP (pipeline.png architecture) ====="
    echo ""

    echo "--- Step 1: Train multilingual model ---"
    python src/train.py --config configs/multilingual.yaml

    echo ""
    echo "--- Step 2: Evaluate (all languages) ---"
    python src/evaluate.py --config configs/multilingual.yaml

    echo ""
    echo "--- Step 3: Per-language inference ---"
    for LANG in en zh fr es; do
      echo ""
      echo "--- Language: $LANG ---"
      python src/infer.py \
        --config configs/multilingual.yaml \
        --image data/demo/images/sample_001.png \
        --question "What color is the object?" \
        --lang "$LANG"
    done
    ;;

  *)
    echo "Usage: $0 [smoke|multilingual]"
    echo ""
    echo "  smoke        - Toy CNN model, no GPU needed (default)"
    echo "  multilingual - Multilingual Med-BLIP (requires GPU + model download)"
    exit 1
    ;;
esac

echo ""
echo "===== Demo complete ====="
