from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import binomtest


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_inference import paired_component_bootstrap, weighted_classification_metrics  # noqa: E402
from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reliability(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    order = np.argsort(probabilities, kind="stable")
    partitions = np.array_split(order, bins)
    rows = []
    ece = 0.0
    for index, members in enumerate(partitions):
        mean_probability = float(probabilities[members].mean())
        observed = float(targets[members].mean())
        mass = float(len(members) / len(targets))
        ece += mass * abs(mean_probability - observed)
        rows.append(
            {
                "bin": index + 1,
                "record_count": int(len(members)),
                "minimum_probability": float(probabilities[members].min()),
                "maximum_probability": float(probabilities[members].max()),
                "mean_probability": mean_probability,
                "observed_unsafe_rate": observed,
                "calibration_gap": mean_probability - observed,
            }
        )
    return {"equal_frequency_ece": float(ece), "bins": rows}


def _threshold_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    thresholds = np.linspace(0.0, 1.0, 201)
    return [
        {
            "threshold": float(threshold),
            "metrics": weighted_classification_metrics(
                targets, probabilities >= threshold
            ),
        }
        for threshold in thresholds
    ]


def _stable_fold(group: str, fold_count: int) -> int:
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def _fit_intercept(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    y = targets.astype(np.float64)
    base = logit(np.clip(probabilities, 1e-6, 1.0 - 1e-6))

    def objective(intercept: float) -> float:
        adjusted = np.clip(expit(base + intercept), 1e-6, 1.0 - 1e-6)
        return float(-np.mean(y * np.log(adjusted) + (1.0 - y) * np.log1p(-adjusted)))

    result = minimize_scalar(objective, bounds=(-10.0, 10.0), method="bounded")
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("Intercept-only diagnostic calibration failed")
    return float(result.x)


def _cross_fitted_intercept(
    targets: np.ndarray,
    probabilities: np.ndarray,
    groups: Sequence[str],
    *,
    fold_count: int = 5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    folds = np.asarray([_stable_fold(str(group), fold_count) for group in groups])
    adjusted = np.full(len(targets), np.nan, dtype=np.float64)
    records = []
    for fold in range(fold_count):
        validation = folds == fold
        training = ~validation
        intercept = _fit_intercept(targets[training], probabilities[training])
        adjusted[validation] = expit(
            logit(np.clip(probabilities[validation], 1e-6, 1.0 - 1e-6))
            + intercept
        )
        records.append(
            {
                "fold": fold,
                "training_count": int(training.sum()),
                "validation_count": int(validation.sum()),
                "intercept": intercept,
            }
        )
    if not np.all(np.isfinite(adjusted)):
        raise RuntimeError("Cross-fitted intercept diagnostic is incomplete")
    return adjusted, records


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve()
    results_path = args.results.resolve()
    plan_path = args.analysis_plan.resolve()
    output_path = args.output.resolve()
    rows = _load_jsonl(predictions_path)
    results = _load_json(results_path)
    plan = _load_json(plan_path)
    if len(rows) != 1180 or results.get("integrity_passed") is not True:
        raise RuntimeError("External transfer results are incomplete")
    targets = np.asarray(
        [1 if row["expected_label"] == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )
    groups = [str(row["group_id"]) for row in rows]
    fusion = np.asarray([float(row["fusion_p_unsafe"]) for row in rows])
    frozen_threshold = float(plan["transfer_contract"]["fusion_threshold"])
    frozen_prediction = (fusion >= frozen_threshold).astype(np.int64)
    majority = np.asarray(
        [1 if row["predictions"]["majority"] == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )
    curve = _threshold_curve(targets, fusion)
    best = max(
        curve,
        key=lambda value: (
            float(value["metrics"]["macro_f1"]),
            float(value["metrics"]["unsafe_recall"]),
            float(value["metrics"]["safe_specificity"]),
        ),
    )
    majority_unsafe_count = int(majority.sum())
    matched_threshold = float(
        np.sort(fusion)[max(0, len(fusion) - majority_unsafe_count)]
    )
    matched_prediction = (fusion >= matched_threshold).astype(np.int64)

    adjusted, intercept_folds = _cross_fitted_intercept(targets, fusion, groups)
    adjusted_prediction = (adjusted >= frozen_threshold).astype(np.int64)
    majority_correct = majority == targets
    fusion_correct = frozen_prediction == targets
    majority_only = int(np.logical_and(majority_correct, ~fusion_correct).sum())
    fusion_only = int(np.logical_and(~majority_correct, fusion_correct).sum())
    discordant = majority_only + fusion_only
    transition = {
        "both_correct": int(np.logical_and(majority_correct, fusion_correct).sum()),
        "majority_only_correct": majority_only,
        "fusion_only_correct": fusion_only,
        "both_wrong": int(np.logical_and(~majority_correct, ~fusion_correct).sum()),
        "exact_mcnemar_two_sided_p": (
            None
            if discordant == 0
            else float(binomtest(fusion_only, discordant, 0.5).pvalue)
        ),
    }
    diagnostic_metrics = {
        "frozen_fusion": weighted_classification_metrics(targets, frozen_prediction),
        "majority": weighted_classification_metrics(targets, majority),
        "posthoc_full_data_best_threshold": best["metrics"],
        "posthoc_majority_rate_matched_threshold": weighted_classification_metrics(
            targets, matched_prediction
        ),
        "posthoc_cross_fitted_intercept": weighted_classification_metrics(
            targets, adjusted_prediction
        ),
    }
    intercept_bootstrap = paired_component_bootstrap(
        targets=targets,
        groups=groups,
        left_predictions=adjusted_prediction,
        right_predictions=majority,
        replicates=10000,
        seed=20260812,
    )
    result = {
        "schema": "beavertails-v-posthoc-shift-diagnostic-v1",
        "status": "completed",
        "role": "posthoc_diagnostic_not_confirmatory_evidence",
        "sample_count": len(rows),
        "group_count": len(set(groups)),
        "inputs": {
            "predictions_sha256": sha256_file(predictions_path),
            "results_sha256": sha256_file(results_path),
            "analysis_plan_sha256": str(plan["analysis_plan_sha256"]),
        },
        "unsafe_rates": {
            "observed": float(targets.mean()),
            "majority": float(majority.mean()),
            "frozen_fusion": float(frozen_prediction.mean()),
            "posthoc_cross_fitted_intercept": float(adjusted_prediction.mean()),
        },
        "frozen_threshold": frozen_threshold,
        "posthoc_full_data_best_threshold": float(best["threshold"]),
        "posthoc_majority_rate_matched_threshold": matched_threshold,
        "posthoc_intercept_folds": intercept_folds,
        "metrics": diagnostic_metrics,
        "cross_fitted_intercept_vs_majority_bootstrap": intercept_bootstrap,
        "majority_fusion_error_transition": transition,
        "fusion_reliability": _reliability(targets, fusion),
        "cross_fitted_intercept_reliability": _reliability(targets, adjusted),
        "threshold_curve": curve,
        "interpretation_constraints": [
            "Threshold and intercept diagnostics were designed after observing the frozen-transfer result.",
            "They cannot replace or relabel the prespecified negative external comparison.",
            "They may motivate a future preregistered deployment-calibration experiment.",
        ],
    }
    result["content_sha256"] = sha256_text(canonical_json(result))
    _write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    base = (
        V15_ROOT
        / "results"
        / "external"
        / "beavertails_v"
    )
    parser.add_argument(
        "--predictions", type=Path, default=base / "evaluation" / "predictions.jsonl"
    )
    parser.add_argument("--results", type=Path, default=base / "evaluation" / "results.json")
    parser.add_argument("--analysis-plan", type=Path, default=base / "analysis_plan.json")
    parser.add_argument(
        "--output", type=Path, default=base / "evaluation" / "shift_diagnostic.json"
    )
    return parser


if __name__ == "__main__":
    result = analyze(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "unsafe_rates": result["unsafe_rates"],
                "metrics": result["metrics"],
            },
            indent=2,
        )
    )
