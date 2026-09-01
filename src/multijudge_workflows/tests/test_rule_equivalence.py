from __future__ import annotations

import numpy as np

from multijudge.baselines import (
    baseline_predictions,
    classification_metrics,
    rule_equivalence,
    select_constrained_threshold,
    vote_pattern_report,
)


def test_binary_metrics_include_safety_operating_points() -> None:
    metrics = classification_metrics(
        [1, 1, 0, 0],
        [1, 0, 1, 0],
    )
    assert metrics["unsafe_recall"] == 0.5
    assert metrics["safe_specificity"] == 0.5
    assert metrics["false_safe_rate"] == 0.5
    assert metrics["mcc"] == 0.0


def test_three_judge_baselines_match_vote_counts() -> None:
    probabilities = {
        "qwen": [0.9, 0.9, 0.1, 0.1],
        "safework": [0.9, 0.1, 0.9, 0.1],
        "minicpm": [0.9, 0.1, 0.1, 0.9],
    }
    values = baseline_predictions(probabilities, tuple(probabilities))
    assert values["or_1_of_n"].tolist() == [1, 1, 1, 1]
    assert values["majority"].tolist() == [1, 0, 0, 0]
    assert values["and_n_of_n"].tolist() == [1, 0, 0, 0]


def test_rule_equivalence_detects_exact_majority() -> None:
    baselines = {
        "majority": np.asarray([1, 0, 1, 0]),
        "or": np.asarray([1, 1, 1, 0]),
    }
    audit = rule_equivalence([1, 0, 1, 0], baselines)
    assert audit["equivalent_rules"] == ["majority"]
    assert audit["comparisons"]["or"]["difference_count"] == 1


def test_constrained_threshold_obeys_both_bounds() -> None:
    targets = np.asarray([1, 1, 1, 0, 0, 0])
    probabilities = np.asarray([0.9, 0.8, 0.4, 0.7, 0.2, 0.1])
    result = select_constrained_threshold(
        targets,
        probabilities,
        minimum_unsafe_recall=2 / 3,
        minimum_safe_specificity=2 / 3,
    )
    assert result["feasible"] is True
    assert result["metrics"]["unsafe_recall"] >= 2 / 3
    assert result["metrics"]["safe_specificity"] >= 2 / 3


def test_vote_pattern_report_includes_unobserved_patterns() -> None:
    probabilities = {
        "qwen": [0.9, 0.1],
        "safework": [0.9, 0.1],
        "minicpm": [0.1, 0.1],
    }
    report = vote_pattern_report(
        probabilities, tuple(probabilities), ["a", "b"]
    )
    assert len(report["patterns"]) == 8
    assert report["patterns"]["110"]["record_count"] == 1
    assert report["patterns"]["000"]["record_count"] == 1
    assert report["patterns"]["111"]["record_count"] == 0
