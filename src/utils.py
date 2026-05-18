"""Utility helpers: config loading, answer normalisation, metrics."""

import re
import string
from collections import Counter
from typing import Dict

import yaml


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalise_answer(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(pred: str, gold: str) -> bool:
    return normalise_answer(pred) == normalise_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalise_answer(pred).split()
    gold_tokens = normalise_answer(gold).split()
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
