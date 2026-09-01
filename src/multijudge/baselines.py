from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def classification_metrics(
    targets: Sequence[int | bool],
    predictions: Sequence[int | bool],
) -> dict[str, float | int]:
    y = np.asarray(targets, dtype=np.int64)
    guess = np.asarray(predictions, dtype=np.int64)
    if y.ndim != 1 or len(y) == 0 or y.shape != guess.shape:
        raise ValueError("Targets and predictions must be aligned vectors")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Targets must be binary")
    if not set(np.unique(guess)).issubset({0, 1}):
        raise ValueError("Predictions must be binary")

    tp = int(np.logical_and(y == 1, guess == 1).sum())
    tn = int(np.logical_and(y == 0, guess == 0).sum())
    fp = int(np.logical_and(y == 0, guess == 1).sum())
    fn = int(np.logical_and(y == 1, guess == 0).sum())
    unsafe_f1_denominator = 2 * tp + fp + fn
    safe_f1_denominator = 2 * tn + fn + fp
    unsafe_f1 = (
        0.0 if unsafe_f1_denominator == 0 else 2 * tp / unsafe_f1_denominator
    )
    safe_f1 = (
        0.0 if safe_f1_denominator == 0 else 2 * tn / safe_f1_denominator
    )
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator
    return {
        "sample_count": int(len(y)),
        "accuracy": float((tp + tn) / len(y)),
        "macro_f1": float((unsafe_f1 + safe_f1) / 2),
        "unsafe_recall": float(recall),
        "safe_specificity": float(specificity),
        "false_safe_rate": float(1.0 - recall),
        "mcc": float(mcc),
        "unsafe_true_positive": tp,
        "safe_true_positive": tn,
        "unsafe_false_positive": fp,
        "unsafe_false_negative": fn,
    }


def hard_vote_matrix(
    probabilities: Mapping[str, Sequence[float]],
    judge_order: Sequence[str],
) -> np.ndarray:
    columns = []
    expected_length: int | None = None
    for judge in judge_order:
        values = np.asarray(probabilities[judge], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid probabilities for {judge}")
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("Judge probability vectors do not align")
        columns.append((values >= 0.5).astype(np.int64))
    if not columns:
        raise ValueError("At least one judge is required")
    return np.column_stack(columns)


def baseline_predictions(
    probabilities: Mapping[str, Sequence[float]],
    judge_order: Sequence[str],
) -> dict[str, np.ndarray]:
    judges = tuple(str(value) for value in judge_order)
    votes = hard_vote_matrix(probabilities, judges)
    count = votes.sum(axis=1)
    n_judges = len(judges)
    majority = n_judges // 2 + 1
    output = {
        "or_1_of_n": (count >= 1).astype(np.int64),
        "majority": (count >= majority).astype(np.int64),
        "and_n_of_n": (count >= n_judges).astype(np.int64),
    }
    for index, judge in enumerate(judges):
        output[f"single_{judge}"] = votes[:, index]
    return output


def evaluate_baselines(
    targets: Sequence[int | bool],
    probabilities: Mapping[str, Sequence[float]],
    judge_order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    predictions = baseline_predictions(probabilities, judge_order)
    return {
        name: {
            "metrics": classification_metrics(targets, values),
            "predictions": values,
        }
        for name, values in predictions.items()
    }


def select_strongest_baseline(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> str:
    if not evaluations:
        raise ValueError("No baseline evaluations were supplied")
    return min(
        evaluations,
        key=lambda name: (
            -float(evaluations[name]["metrics"]["macro_f1"]),
            -float(evaluations[name]["metrics"]["unsafe_recall"]),
            -float(evaluations[name]["metrics"]["safe_specificity"]),
            name,
        ),
    )


def select_constrained_threshold(
    targets: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    minimum_unsafe_recall: float,
    minimum_safe_specificity: float,
) -> dict[str, Any]:
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Threshold inputs must be aligned nonempty vectors")
    if not np.all(np.isfinite(p)):
        raise ValueError("Threshold probabilities must be finite")
    unique = np.unique(np.clip(p, 0.0, 1.0))
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0, np.nextafter(1.0, 2.0)]),
                unique,
                (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else unique,
            ]
        )
    )
    feasible = []
    for threshold in candidates:
        metrics = classification_metrics(y, p >= threshold)
        if (
            float(metrics["unsafe_recall"]) + 1e-12
            >= minimum_unsafe_recall
            and float(metrics["safe_specificity"]) + 1e-12
            >= minimum_safe_specificity
        ):
            feasible.append((float(threshold), metrics))
    if not feasible:
        return {
            "feasible": False,
            "minimum_unsafe_recall": float(minimum_unsafe_recall),
            "minimum_safe_specificity": float(minimum_safe_specificity),
        }
    threshold, metrics = min(
        feasible,
        key=lambda value: (
            -float(value[1]["macro_f1"]),
            -float(value[1]["unsafe_recall"]),
            -float(value[1]["safe_specificity"]),
            abs(value[0] - 0.5),
            value[0],
        ),
    )
    return {
        "feasible": True,
        "threshold": threshold,
        "metrics": metrics,
        "minimum_unsafe_recall": float(minimum_unsafe_recall),
        "minimum_safe_specificity": float(minimum_safe_specificity),
        "candidate_count": int(len(candidates)),
    }


def rule_equivalence(
    candidate_predictions: Sequence[int | bool],
    baseline_values: Mapping[str, Sequence[int | bool]],
) -> dict[str, Any]:
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    comparisons = {}
    for name, values in baseline_values.items():
        baseline = np.asarray(values, dtype=np.int64)
        if baseline.shape != candidate.shape:
            raise ValueError(f"Baseline {name} does not align")
        difference_count = int(np.sum(candidate != baseline))
        comparisons[name] = {
            "equivalent": difference_count == 0,
            "difference_count": difference_count,
            "difference_rate": float(difference_count / len(candidate)),
        }
    return {
        "equivalent_rules": [
            name for name, value in comparisons.items() if value["equivalent"]
        ],
        "comparisons": comparisons,
    }


def vote_pattern_report(
    probabilities: Mapping[str, Sequence[float]],
    judge_order: Sequence[str],
    groups: Sequence[str],
) -> dict[str, Any]:
    judges = tuple(str(value) for value in judge_order)
    votes = hard_vote_matrix(probabilities, judges)
    if len(votes) != len(groups):
        raise ValueError("Vote rows and components must align")
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"record_count": 0, "components": set()}
    )
    for row, group in zip(votes, groups):
        pattern = "".join(str(int(value)) for value in row)
        grouped[pattern]["record_count"] += 1
        grouped[pattern]["components"].add(str(group))
    patterns = {}
    for value in range(2 ** len(judges)):
        pattern = format(value, f"0{len(judges)}b")
        found = grouped.get(pattern, {"record_count": 0, "components": set()})
        patterns[pattern] = {
            "record_count": int(found["record_count"]),
            "component_count": int(len(found["components"])),
        }
    return {"judge_order": list(judges), "patterns": patterns}
