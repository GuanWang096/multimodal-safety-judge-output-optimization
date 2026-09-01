from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from analyze_external_small_budget_adaptation import (  # noqa: E402
    build_parser,
    evaluate_repeat,
    prepare_dataset,
    select_group_split,
    summarize_repeats,
)


def synthetic_rows(count: int = 80) -> list[dict]:
    rows = []
    for index in range(count):
        unsafe = index % 3 == 0
        label = "unsafe" if unsafe else "safe"
        base = 0.78 if unsafe else 0.22
        calibrated = {
            "qwen": min(0.99, max(0.01, base + ((index % 5) - 2) * 0.03)),
            "safework": min(0.99, max(0.01, base + ((index % 7) - 3) * 0.02)),
            "minicpm": min(0.99, max(0.01, 0.65 if unsafe else 0.35)),
        }
        raw = dict(calibrated)
        majority = sum(value >= 0.5 for value in raw.values()) >= 2
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "group_id": f"group-{index}",
                "expected_label": label,
                "calibrated_p_unsafe": calibrated,
                "raw_p_unsafe": raw,
                "predictions": {
                    "majority": "unsafe" if majority else "safe",
                },
            }
        )
    return rows


def dataset_config() -> dict:
    return {
        "id": "synthetic",
        "label_field": "expected_label",
        "group_field": "group_id",
        "majority_prediction_key": "majority",
    }


def test_group_split_is_deterministic_and_disjoint() -> None:
    prepared = prepare_dataset(
        synthetic_rows(), dataset_config(), ("qwen", "safework", "minicpm")
    )
    first = select_group_split(
        prepared["groups"], prepared["targets"], 16, seed=20260802
    )
    second = select_group_split(
        prepared["groups"], prepared["targets"], 16, seed=20260802
    )
    assert first[2] == second[2]
    assert np.array_equal(first[0], second[0])
    assert not np.any(first[0] & first[1])
    assert np.all(first[0] | first[1])
    assert set(np.unique(prepared["targets"][first[0]])) == {0, 1}
    assert set(np.unique(prepared["targets"][first[1]])) == {0, 1}


def test_repeat_evaluation_returns_all_fixed_comparators() -> None:
    prepared = prepare_dataset(
        synthetic_rows(), dataset_config(), ("qwen", "safework", "minicpm")
    )
    adaptation, evaluation, _, _ = select_group_split(
        prepared["groups"], prepared["targets"], 24, seed=20260803
    )
    result = evaluate_repeat(
        prepared, adaptation, evaluation, c_value=0.03, seed=20260803
    )
    assert set(result["metrics"]) == {
        "adapted_soft_logits",
        "same_pool_majority",
        "adapted_binary_judges",
        "adapted_calibrated_soft_mean",
    }
    assert set(result["primary_minus_comparator"]) == {
        "same_pool_majority",
        "adapted_binary_judges",
        "adapted_calibrated_soft_mean",
    }
    for metrics in result["metrics"].values():
        assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_summary_labels_split_variability_without_inferential_claim() -> None:
    prepared = prepare_dataset(
        synthetic_rows(), dataset_config(), ("qwen", "safework", "minicpm")
    )
    repeats = []
    for seed in (20260804, 20260805, 20260806):
        adaptation, evaluation, _, _ = select_group_split(
            prepared["groups"], prepared["targets"], 20, seed=seed
        )
        repeats.append(
            evaluate_repeat(prepared, adaptation, evaluation, 0.03, seed)
        )
    summary = summarize_repeats(repeats)
    delta = summary["primary_minus_comparator"]["same_pool_majority"][
        "macro_f1"
    ]
    assert "split_variability_interval" in delta
    assert "ci95" not in delta
    assert len(delta["split_variability_interval"]) == 2


def test_cli_argument_names_match_run_signature() -> None:
    arguments = vars(build_parser().parse_args([]))
    assert set(arguments) == {"protocol_path", "output_root"}
