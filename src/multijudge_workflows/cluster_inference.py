from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _binary_vector(values: Sequence[int | bool], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not set(np.unique(result)).issubset({0, 1}):
        raise ValueError(f"{name} must be binary")
    return result


def weighted_classification_metrics(
    targets: Sequence[int | bool],
    predictions: Sequence[int | bool],
    weights: Sequence[float] | None = None,
) -> dict[str, float | int]:
    y = _binary_vector(targets, "targets")
    guess = _binary_vector(predictions, "predictions")
    if y.shape != guess.shape:
        raise ValueError("Targets and predictions must align")
    if weights is None:
        sample_weights = np.ones(len(y), dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != y.shape:
            raise ValueError("Weights must align with targets")
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights <= 0):
            raise ValueError("Weights must be finite and positive")

    tp = float(sample_weights[np.logical_and(y == 1, guess == 1)].sum())
    tn = float(sample_weights[np.logical_and(y == 0, guess == 0)].sum())
    fp = float(sample_weights[np.logical_and(y == 0, guess == 1)].sum())
    fn = float(sample_weights[np.logical_and(y == 1, guess == 0)].sum())
    total = tp + tn + fp + fn
    unsafe_f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    safe_f1 = 0.0 if 2 * tn + fn + fp == 0 else 2 * tn / (2 * tn + fn + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)
    denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator
    return {
        "sample_count": int(len(y)),
        "weight_sum": total,
        "accuracy": (tp + tn) / total,
        "macro_f1": (unsafe_f1 + safe_f1) / 2,
        "unsafe_recall": recall,
        "safe_specificity": specificity,
        "false_safe_rate": 1.0 - recall,
        "mcc": mcc,
        "unsafe_true_positive": tp,
        "safe_true_positive": tn,
        "unsafe_false_positive": fp,
        "unsafe_false_negative": fn,
    }


def probability_metrics(
    targets: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    y = _binary_vector(targets, "targets").astype(np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape or not np.all(np.isfinite(p)):
        raise ValueError("Probabilities must be finite and align with targets")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if weights is None:
        sample_weights = np.ones(len(y), dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64)
        if sample_weights.shape != y.shape:
            raise ValueError("Weights must align with targets")
        if not np.all(np.isfinite(sample_weights)) or np.any(sample_weights <= 0):
            raise ValueError("Weights must be finite and positive")
    normalized = sample_weights / sample_weights.sum()
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    brier = float(np.dot(normalized, np.square(p - y)))
    log_loss = float(
        -np.dot(normalized, y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
    )
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    populated = 0
    for index in range(ece_bins):
        upper_inclusive = index == ece_bins - 1
        members = (p >= edges[index]) & (
            (p <= edges[index + 1]) if upper_inclusive else (p < edges[index + 1])
        )
        if not np.any(members):
            continue
        populated += 1
        bin_weights = sample_weights[members]
        bin_weights = bin_weights / bin_weights.sum()
        confidence = float(np.dot(bin_weights, p[members]))
        frequency = float(np.dot(bin_weights, y[members]))
        mass = float(sample_weights[members].sum() / sample_weights.sum())
        ece += mass * abs(confidence - frequency)
    return {
        "sample_count": int(len(y)),
        "brier_score": brier,
        "log_loss": log_loss,
        "ece": float(ece),
        "ece_bins": int(ece_bins),
        "populated_ece_bins": populated,
    }


def _component_indices(groups: Sequence[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    if not groups:
        raise ValueError("At least one component is required")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(groups):
        grouped[str(value)].append(index)
    component_ids = sorted(grouped)
    return component_ids, {
        component: np.asarray(indices, dtype=np.int64)
        for component, indices in grouped.items()
    }


def paired_component_bootstrap(
    *,
    targets: Sequence[int | bool],
    groups: Sequence[str],
    left_predictions: Sequence[int | bool],
    right_predictions: Sequence[int | bool],
    replicates: int = 10000,
    seed: int = 20260724,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    y = _binary_vector(targets, "targets")
    left = _binary_vector(left_predictions, "left_predictions")
    right = _binary_vector(right_predictions, "right_predictions")
    if not (len(y) == len(left) == len(right) == len(groups)):
        raise ValueError("Bootstrap inputs must align")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    component_ids, grouped = _component_indices(groups)
    rng = np.random.default_rng(seed)
    names = ("macro_f1", "unsafe_recall", "safe_specificity", "mcc")
    distributions = {
        name: np.empty(replicates, dtype=np.float64) for name in names
    }
    for replicate in range(replicates):
        sampled = rng.choice(component_ids, size=len(component_ids), replace=True)
        indices = np.concatenate([grouped[component] for component in sampled])
        left_metrics = weighted_classification_metrics(y[indices], left[indices])
        right_metrics = weighted_classification_metrics(y[indices], right[indices])
        for name in names:
            distributions[name][replicate] = float(left_metrics[name]) - float(
                right_metrics[name]
            )
    alpha = 1.0 - confidence_level
    interval_key = f"ci{int(round(confidence_level * 100))}"
    return {
        "schema": "paired-component-bootstrap-v1",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "component_count": len(component_ids),
        "difference_direction": "left_minus_right",
        "metrics": {
            name: {
                "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)),
                interval_key: np.quantile(
                    values, [alpha / 2.0, 1.0 - alpha / 2.0]
                ).tolist(),
                "positive_fraction": float(np.mean(values > 0)),
                "nonnegative_fraction": float(np.mean(values >= 0)),
            }
            for name, values in distributions.items()
        },
    }


def paired_component_randomization(
    *,
    targets: Sequence[int | bool],
    groups: Sequence[str],
    left_predictions: Sequence[int | bool],
    right_predictions: Sequence[int | bool],
    replicates: int = 10000,
    seed: int = 20260724,
) -> dict[str, Any]:
    y = _binary_vector(targets, "targets")
    left = _binary_vector(left_predictions, "left_predictions")
    right = _binary_vector(right_predictions, "right_predictions")
    if not (len(y) == len(left) == len(right) == len(groups)):
        raise ValueError("Randomization inputs must align")
    component_ids, grouped = _component_indices(groups)
    observed = float(
        weighted_classification_metrics(y, left)["macro_f1"]
        - weighted_classification_metrics(y, right)["macro_f1"]
    )
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        swap = set(
            component
            for component, selected in zip(
                component_ids, rng.integers(0, 2, size=len(component_ids))
            )
            if selected
        )
        permuted_left = left.copy()
        permuted_right = right.copy()
        for component in swap:
            indices = grouped[component]
            permuted_left[indices], permuted_right[indices] = (
                right[indices],
                left[indices],
            )
        null[replicate] = float(
            weighted_classification_metrics(y, permuted_left)["macro_f1"]
            - weighted_classification_metrics(y, permuted_right)["macro_f1"]
        )
    return {
        "schema": "paired-component-randomization-v1",
        "replicates": replicates,
        "seed": seed,
        "component_count": len(component_ids),
        "metric": "macro_f1",
        "observed_delta": observed,
        "one_sided_p_value": float((1 + np.sum(null >= observed)) / (replicates + 1)),
        "two_sided_p_value": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (replicates + 1)
        ),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), str(name)) for name, value in p_values.items())
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (value, name) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
