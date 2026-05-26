from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_processbench_metrics(labels: Sequence[int], preds: Sequence[int]) -> dict[str, float]:
    """Compute first-error localization metrics.

    `-1` means no erroneous step; non-negative values are zero-based first-error
    indices.
    """

    if len(labels) == 0:
        return {}
    if len(labels) != len(preds):
        raise ValueError(f"labels and preds length mismatch: {len(labels)} != {len(preds)}")

    labels_arr = np.asarray(labels, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    has_error = labels_arr != -1
    no_error = labels_arr == -1

    error_acc = (
        float(np.mean(preds_arr[has_error] == labels_arr[has_error])) if np.any(has_error) else 0.0
    )
    correct_acc = float(np.mean(preds_arr[no_error] == -1)) if np.any(no_error) else 0.0
    f1 = (
        2.0 * error_acc * correct_acc / (error_acc + correct_acc)
        if (error_acc + correct_acc) > 0
        else 0.0
    )
    return {"error_acc": error_acc, "correct_acc": correct_acc, "f1": f1}


def compute_step_classification_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
) -> dict[str, float]:
    """Compute binary step-level metrics for PRM-style evaluation."""

    if len(labels) == 0:
        return {}
    if len(labels) != len(preds):
        raise ValueError(f"labels and preds length mismatch: {len(labels)} != {len(preds)}")

    labels_arr = np.asarray(labels, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    if np.any((labels_arr != 0) & (labels_arr != 1)):
        raise ValueError("step labels must be binary 0/1 values")
    if np.any((preds_arr != 0) & (preds_arr != 1)):
        raise ValueError("step predictions must be binary 0/1 values")

    correct_mask = labels_arr == 1
    error_mask = labels_arr == 0
    correct_acc = float(np.mean(preds_arr[correct_mask] == 1)) if np.any(correct_mask) else 0.0
    error_acc = float(np.mean(preds_arr[error_mask] == 0)) if np.any(error_mask) else 0.0

    tp = float(np.sum((preds_arr == 1) & (labels_arr == 1)))
    fp = float(np.sum((preds_arr == 1) & (labels_arr == 0)))
    tn = float(np.sum((preds_arr == 0) & (labels_arr == 0)))
    fn = float(np.sum((preds_arr == 0) & (labels_arr == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    negative_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    negative_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    negative_f1 = (
        2.0 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if (negative_precision + negative_recall) > 0
        else 0.0
    )

    return {
        "error_acc": error_acc,
        "correct_acc": correct_acc,
        "total_acc": float(np.mean(preds_arr == labels_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_precision": float(negative_precision),
        "negative_recall": float(negative_recall),
        "negative_f1": float(negative_f1),
        "macro_f1": float((f1 + negative_f1) * 0.5),
    }


def compute_pairwise_accuracy(
    chosen_scores: Sequence[float],
    rejected_scores: Sequence[float],
    *,
    tie_score: float = 0.5,
) -> dict[str, float]:
    if len(chosen_scores) == 0:
        return {}
    if len(chosen_scores) != len(rejected_scores):
        raise ValueError("chosen_scores and rejected_scores must have the same length")

    chosen = np.asarray(chosen_scores, dtype=np.float64)
    rejected = np.asarray(rejected_scores, dtype=np.float64)
    wins = chosen > rejected
    ties = chosen == rejected
    losses = chosen < rejected
    margins = chosen - rejected
    return {
        "total": int(len(chosen_scores)),
        "pair_accuracy": float((wins.astype(float) + ties.astype(float) * tie_score).mean()),
        "win_rate": float(wins.mean()),
        "tie_rate": float(ties.mean()),
        "loss_rate": float(losses.mean()),
        "avg_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
    }
