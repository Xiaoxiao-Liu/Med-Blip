"""MedCon evaluation: DeltaBLEU, BERTScore, and UMLS concept F1."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.data import load_json
from src.utils import load_config


# ---------------------------------------------------------------------------
# UMLS MedCon F1
# ---------------------------------------------------------------------------

def load_umls_concepts(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def simple_concept_match(text: str, concept_dict: dict, stop_words: list = None) -> list:
    """Simple substring-based concept extraction (no QuickUMLS dependency).

    For production use, replace with QuickUMLS-based extraction as shown in
    the notebooks under ``data/medcon/``.
    """
    text_lower = text.lower()
    stop_terms = set(stop_words) if stop_words else set()
    found = []
    for term, info in concept_dict.items():
        if term.lower() in stop_terms:
            continue
        if term.lower() in text_lower and info.get("type"):
            found.append({
                "term": term,
                "type": info["type"],
                "status": "present",
            })
    return found


def compute_medcon_f1(pred_concepts: list, ref_concepts: list) -> float:
    """Compute F1 between two concept sets (term, type, status) tuples."""
    pred_set = set((c["term"], c["type"], c.get("status", "present")) for c in pred_concepts)
    ref_set = set((c["term"], c["type"], c.get("status", "present")) for c in ref_concepts)

    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0

    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(ref_set) if ref_set else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_medcon_f1(truth, prediction, concept_dict, lang="en", stop_words=None):
    """Compute MedCon F1 across all encounters."""
    content_key = f"content_{lang}"
    all_f1 = []

    for pred_item, ref_item in zip(prediction, truth):
        assert pred_item["encounter_id"] == ref_item["encounter_id"]

        pred_text = pred_item["responses"][0][content_key]
        pred_concepts = simple_concept_match(pred_text, concept_dict, stop_words)

        max_f1 = 0.0
        for ref_resp in ref_item["responses"]:
            ref_text = ref_resp[content_key]
            ref_concepts = simple_concept_match(ref_text, concept_dict, stop_words)
            f1 = compute_medcon_f1(pred_concepts, ref_concepts)
            max_f1 = max(max_f1, f1)

        all_f1.append(max_f1)

    return float(np.mean(all_f1)) if all_f1 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg):
    lang = cfg.get("lang", "en")
    reference_path = cfg["reference_file"]
    prediction_path = cfg["prediction_file"]
    output_dir = cfg.get("output_dir", "outputs")
    score_path = os.path.join(output_dir, "medcon_scores.json")

    print(f"=== MedCon Evaluation (lang={lang}) ===")
    print(f"Reference: {reference_path}")
    print(f"Prediction: {prediction_path}")

    truth = load_json(reference_path)
    prediction = load_json(prediction_path)

    assert len(truth) == len(prediction), (
        f"Reference ({len(truth)}) and prediction ({len(prediction)}) must have same length"
    )

    scores = {}

    # --- DeltaBLEU + BERTScore via scoring_multilingresp ---
    print("\n--- DeltaBLEU + BERTScore ---")
    nlg_score_path = os.path.join(output_dir, "nlg_scores.json")
    try:
        from src.nlg.scoring_multilingresp import main as score_main
        nlg_scores = score_main(truth, prediction, nlg_score_path, df_userinfo=None)
        scores.update(nlg_scores)
    except Exception as e:
        print(f"WARNING: NLG scoring failed: {e}")
        print("Falling back to DeltaBLEU-only (without BERTScore)...")
        try:
            from src.nlg import sacrebleu_deltableu
            content_key = f"content_{lang}"
            refs = [[r[content_key] for r in item["responses"]] for item in truth]
            hyps = [item["responses"][0][content_key] for item in prediction]
            ref_weights = [[1.0] * len(r) for r in refs]

            kwargs = dict(
                ref_weights=ref_weights,
                lowercase=True,
                use_effective_order=True,
            )
            if lang == "zh":
                kwargs["tokenize"] = "zh"

            deltableu = sacrebleu_deltableu.corpus_bleu_t(hyps, refs, **kwargs)
            scores[f"deltableu_{lang}"] = deltableu.score
            print(f"DeltaBLEU ({lang}): {deltableu.score:.4f}")
        except Exception as e2:
            print(f"WARNING: DeltaBLEU fallback also failed: {e2}")

    # --- MedCon F1 (UMLS concept matching) ---
    print("\n--- MedCon F1 ---")
    umls_path = cfg.get("umls_concepts_file")
    stop_words_path = cfg.get("umls_stop_words_file")

    if umls_path and os.path.exists(umls_path):
        concept_dict = load_umls_concepts(umls_path)
        stop_words = None
        if stop_words_path and os.path.exists(stop_words_path):
            sw_data = load_json(stop_words_path)
            stop_words = sw_data.get("terms", [])

        medcon_f1 = evaluate_medcon_f1(truth, prediction, concept_dict, lang=lang, stop_words=stop_words)
        scores[f"medcon_f1_{lang}"] = medcon_f1
        print(f"MedCon F1 ({lang}): {medcon_f1:.4f}")
    else:
        print("No UMLS concept file found, skipping MedCon F1")

    # --- Save scores ---
    os.makedirs(output_dir, exist_ok=True)
    with open(score_path, 'w') as f:
        json.dump(scores, f, indent=4)
    print(f"\nAll scores saved to {score_path}")
    print(json.dumps(scores, indent=4))

    return scores


def main():
    parser = argparse.ArgumentParser(description="MedCon evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
