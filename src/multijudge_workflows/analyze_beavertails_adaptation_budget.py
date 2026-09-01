from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


V15_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from analyze_beavertails_target_adaptation import (  # noqa: E402
    classification_metrics,
    feature_matrices,
    fit_model,
    load_jsonl,
    select_threshold,
)
from multijudge.schema import canonical_json, sha256_text  # noqa: E402


CALIBRATION_GROUP_BUDGETS = (25, 50, 100, 150, 200)
REPEATS = 30
METHODS = {
    "soft_logits": 0.03,
    "soft_plus_vote": 0.03,
}


def run() -> dict[str, Any]:
    path = (
        V15_ROOT
        / "results/external/beavertails_v/predictions.jsonl"
    )
    rows = load_jsonl(path)
    targets = np.asarray(
        [1 if row["expected_label"] == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )
    groups = np.asarray([str(row["group_id"]) for row in rows])
    unique_groups = np.unique(groups)
    all_features = feature_matrices(rows)
    majority = np.asarray(
        [1 if row["predictions"]["majority"] == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )

    method_results = {}
    for family, c_value in METHODS.items():
        features = all_features[family]
        budgets = {}
        for budget in CALIBRATION_GROUP_BUDGETS:
            repeat_rows = []
            for repeat in range(REPEATS):
                rng = np.random.default_rng(20261000 + budget * 100 + repeat)
                selected_groups = set(
                    rng.choice(unique_groups, size=budget, replace=False).tolist()
                )
                training = np.asarray(
                    [group in selected_groups for group in groups], dtype=bool
                )
                validation = ~training
                model = fit_model(features[training], targets[training], c_value)
                training_scores = model.predict_proba(features[training])[:, 1]
                threshold = select_threshold(targets[training], training_scores)
                validation_scores = model.predict_proba(features[validation])[:, 1]
                adapted_metrics = classification_metrics(
                    targets[validation], validation_scores >= threshold
                )
                majority_metrics = classification_metrics(
                    targets[validation], majority[validation]
                )
                repeat_rows.append(
                    {
                        "repeat": repeat,
                        "calibration_group_count": budget,
                        "calibration_record_count": int(training.sum()),
                        "evaluation_record_count": int(validation.sum()),
                        "threshold": threshold,
                        "adapted_metrics": adapted_metrics,
                        "majority_metrics": majority_metrics,
                        "macro_f1_delta": (
                            float(adapted_metrics["macro_f1"])
                            - float(majority_metrics["macro_f1"])
                        ),
                    }
                )
            deltas = np.asarray([row["macro_f1_delta"] for row in repeat_rows])
            adapted_f1 = np.asarray(
                [row["adapted_metrics"]["macro_f1"] for row in repeat_rows]
            )
            budgets[str(budget)] = {
                "repeat_count": REPEATS,
                "mean_adapted_macro_f1": float(adapted_f1.mean()),
                "mean_macro_f1_delta": float(deltas.mean()),
                "median_macro_f1_delta": float(np.median(deltas)),
                "delta_interval_across_splits": [
                    float(np.quantile(deltas, 0.025)),
                    float(np.quantile(deltas, 0.975)),
                ],
                "positive_delta_fraction": float(np.mean(deltas > 0.0)),
                "repeats": repeat_rows,
            }
        method_results[family] = {
            "C": c_value,
            "budgets": budgets,
        }

    output = {
        "schema": "beavertails-v-target-adaptation-budget-diagnostic-v1",
        "role": "posthoc_supervised_target_adaptation_budget_exploration",
        "sample_count": len(rows),
        "group_count": len(unique_groups),
        "fixed_methods": {
            family: {
                "feature_family": family,
                "penalty": "l2",
                "C": c_value,
            }
            for family, c_value in METHODS.items()
        },
        "shared_threshold_selection": (
            "maximum calibration-subset Macro-F1"
        ),
        "methods": method_results,
        "interpretation_constraints": [
            "Calibration subsets and held-out evaluation subsets are group-disjoint within each repeat.",
            "Thresholds are selected in-sample on each calibration subset and assessed only on its held-out subset.",
            "Repeated-split intervals describe split variability and are not confirmatory confidence intervals.",
            "The fixed method was chosen after observing BeaverTails-V aggregate results and requires prospective confirmation.",
        ],
    }
    output["content_sha256"] = sha256_text(canonical_json(output))
    output_path = (
        V15_ROOT
        / "results/external/beavertails_v/adaptation_budget_diagnostic.json"
    )
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    result = run()
    summary = {
        family: {
            budget: {
                key: value[key]
                for key in (
                    "mean_adapted_macro_f1",
                    "mean_macro_f1_delta",
                    "delta_interval_across_splits",
                    "positive_delta_fraction",
                )
            }
            for budget, value in method["budgets"].items()
        }
        for family, method in result["methods"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
