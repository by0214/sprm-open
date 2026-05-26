"""Evaluation utilities and benchmark entrypoints for SPRM."""

from .benchmark_common import (
    aggregate_step_scores,
    first_error_index_from_correctness,
    labels_from_first_error,
    normalize_steps,
    predict_first_error_from_scores,
)
from .metrics import (
    compute_pairwise_accuracy,
    compute_processbench_metrics,
    compute_step_classification_metrics,
)

__all__ = [
    "aggregate_step_scores",
    "compute_pairwise_accuracy",
    "compute_processbench_metrics",
    "compute_step_classification_metrics",
    "first_error_index_from_correctness",
    "labels_from_first_error",
    "normalize_steps",
    "predict_first_error_from_scores",
]
