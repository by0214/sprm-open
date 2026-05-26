from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from sprm.inference import build_sprm_input_text


def extract_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "text", "response", "answer", "completion", "value"):
            if key in value:
                return extract_text_content(value[key])
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := extract_text_content(item))).strip()
    return str(value).strip()


def normalize_steps(raw_steps: Any, mode: str = "auto") -> list[str]:
    if raw_steps is None:
        return []
    if isinstance(raw_steps, list):
        return [step for item in raw_steps if (step := extract_text_content(item))]

    text = extract_text_content(raw_steps)
    if not text:
        return []
    if mode == "single":
        return [text]
    if mode in {"auto", "step"}:
        steps = split_structured_steps(text)
        if steps:
            return steps
    if mode in {"auto", "newline"} and "\n" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
    if mode in {"auto", "sentence"}:
        parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
        if len(parts) > 1:
            return parts
    return [text]


def split_structured_steps(text: str) -> list[str]:
    pattern = re.compile(
        r"(?im)(?:^|\n)\s*(?:step\s+\d+|case\b|claim\b|lemma\b|proof\b|\d+[.)])"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    steps: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        step = text[start:end].strip()
        if step:
            steps.append(step)
    return steps if len(steps) > 1 else []


def aggregate_step_scores(step_scores: Sequence[float], mode: str = "mean") -> float:
    if not step_scores:
        return float("nan")
    values = np.asarray(step_scores, dtype=np.float64)
    if mode == "last":
        return float(values[-1])
    if mode == "min":
        return float(values.min())
    if mode == "max":
        return float(values.max())
    if mode == "product":
        return float(values.prod())
    if mode == "mean":
        return float(values.mean())
    raise ValueError(f"Unknown score aggregation mode: {mode}")


def first_error_index_from_correctness(step_correctness: Sequence[int]) -> int:
    for idx, is_correct in enumerate(step_correctness):
        if int(is_correct) == 0:
            return idx
    return -1


def labels_from_first_error(first_error_idx: int, num_steps: int) -> list[int]:
    if num_steps <= 0:
        return []
    labels = [1] * int(num_steps)
    if first_error_idx is None or int(first_error_idx) < 0:
        return labels
    for idx in range(int(first_error_idx), int(num_steps)):
        labels[idx] = 0
    return labels


def predict_first_error_from_scores(step_scores: Sequence[float], threshold: float = 0.5) -> int:
    for idx, score in enumerate(step_scores):
        if float(score) < float(threshold):
            return idx
    return -1


__all__ = [
    "aggregate_step_scores",
    "build_sprm_input_text",
    "extract_text_content",
    "first_error_index_from_correctness",
    "labels_from_first_error",
    "normalize_steps",
    "predict_first_error_from_scores",
    "split_structured_steps",
]
