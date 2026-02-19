from __future__ import annotations

import math
import re
from typing import Iterable


def normalize_question(question: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", question.strip().lower())
    return " ".join(tokens)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or not right_values:
        return -1.0

    limit = min(len(left_values), len(right_values))
    left_values = left_values[:limit]
    right_values = right_values[:limit]
    if not left_values or not right_values:
        return -1.0

    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0

    return float(dot / (left_norm * right_norm))
