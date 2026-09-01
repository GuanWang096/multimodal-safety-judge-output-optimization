from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cluster_inference import weighted_classification_metrics  # noqa: E402
from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


EPSILON = 1e-6
METRIC_KEYS = ("macro_f1", "unsafe_recall", "safe_specificity", "mcc")
METHODS = (
    "adapted_soft_logits",
    "same_pool_majority",
    "adapted_binary_judges",
    "adapted_calibrated_soft_mean",
)
COMPARATORS = (
    "same_pool_majority",
    "adapted_binary_judges",
    "adapted_calibrated_soft_mean",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    metrics = weighted_classification_metrics(targets, predictions)
    return {
        key: float(metrics[key])
        for key in (
            "accuracy",
            "macro_f1",
            "unsafe_recall",
            "safe_specificity",
            "mcc",
            "unsafe_true_positive",
            "unsafe_false_positive",
            "unsafe_false_negative",
        )
    }


def metric_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        metrics["macro_f1"],
        metrics["unsafe_recall"],
        metrics["safe_specificity"],
    )


def select_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(
        np.concatenate(
            ([0.0], np.quantile(scores, np.linspace(0.0, 1.0, 401)), [1.0])
        )
    )
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        current = classification_metrics(targets, scores >= threshold)
        current_key = metric_key(current)
        if current_key > best_key:
            best_key = current_key
            best_threshold = float(threshold)
    return best_threshold


def fit_model(
    features: np.ndarray, targets: np.ndarray, c_value: float, seed: int
) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-9,
        random_state=seed,
    )
    model.fit(features, targets)
    return model


def prepare_dataset(
    rows: Sequence[dict[str, Any]], dataset: dict[str, Any], judges: Sequence[str]
) -> dict[str, Any]:
    label_field = str(dataset["label_field"])
    group_field = str(dataset["group_field"])
    majority_key = str(dataset["majority_prediction_key"])
    if not rows:
        raise ValueError(f"Dataset {dataset['id']} has no records")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Dataset {dataset['id']} contains duplicate sample IDs")
    targets = np.asarray(
        [1 if str(row[label_field]) == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError(f"Dataset {dataset['id']} must contain both labels")
    groups = np.asarray([str(row[group_field]) for row in rows])
    calibrated = np.column_stack(
        [
            [float(row["calibrated_p_unsafe"][judge]) for row in rows]
            for judge in judges
        ]
    )
    raw = np.column_stack(
        [
            [float(row["raw_p_unsafe"][judge]) for row in rows]
            for judge in judges
        ]
    )
    if not np.all(np.isfinite(calibrated)) or not np.all(np.isfinite(raw)):
        raise ValueError(f"Dataset {dataset['id']} contains non-finite scores")
    if (
        np.any(calibrated < 0.0)
        or np.any(calibrated > 1.0)
        or np.any(raw < 0.0)
        or np.any(raw > 1.0)
    ):
        raise ValueError(f"Dataset {dataset['id']} contains invalid probabilities")
    majority = np.asarray(
        [
            1 if str(row["predictions"][majority_key]) == "unsafe" else 0
            for row in rows
        ],
        dtype=np.int64,
    )
    return {
        "targets": targets,
        "groups": groups,
        "soft_features": logit(
            np.clip(calibrated, EPSILON, 1.0 - EPSILON)
        ),
        "hard_features": (raw >= 0.5).astype(float),
        "soft_mean": calibrated.mean(axis=1),
        "majority": majority,
    }


def select_group_split(
    groups: np.ndarray,
    targets: np.ndarray,
    adaptation_group_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    unique_groups = np.unique(groups)
    if adaptation_group_count < 2 or adaptation_group_count >= len(unique_groups):
        raise ValueError("Adaptation group count must leave at least one evaluation group")
    rng = np.random.default_rng(seed)
    for attempt in range(10000):
        selected = sorted(
            str(value)
            for value in rng.choice(
                unique_groups, size=adaptation_group_count, replace=False
            ).tolist()
        )
        adaptation = np.isin(groups, selected)
        evaluation = ~adaptation
        if set(np.unique(targets[adaptation])) != {0, 1}:
            continue
        if set(np.unique(targets[evaluation])) != {0, 1}:
            continue
        return adaptation, evaluation, selected, attempt
    raise RuntimeError("Could not construct a two-class group-disjoint split")


def evaluate_repeat(
    prepared: dict[str, Any],
    adaptation: np.ndarray,
    evaluation: np.ndarray,
    c_value: float,
    seed: int,
) -> dict[str, Any]:
    targets = prepared["targets"]
    soft_model = fit_model(
        prepared["soft_features"][adaptation], targets[adaptation], c_value, seed
    )
    hard_model = fit_model(
        prepared["hard_features"][adaptation], targets[adaptation], c_value, seed
    )
    soft_training_scores = soft_model.predict_proba(
        prepared["soft_features"][adaptation]
    )[:, 1]
    hard_training_scores = hard_model.predict_proba(
        prepared["hard_features"][adaptation]
    )[:, 1]
    soft_threshold = select_threshold(targets[adaptation], soft_training_scores)
    hard_threshold = select_threshold(targets[adaptation], hard_training_scores)
    mean_threshold = select_threshold(
        targets[adaptation], prepared["soft_mean"][adaptation]
    )
    predictions = {
        "adapted_soft_logits": (
            soft_model.predict_proba(prepared["soft_features"][evaluation])[:, 1]
            >= soft_threshold
        ),
        "same_pool_majority": prepared["majority"][evaluation].astype(bool),
        "adapted_binary_judges": (
            hard_model.predict_proba(prepared["hard_features"][evaluation])[:, 1]
            >= hard_threshold
        ),
        "adapted_calibrated_soft_mean": (
            prepared["soft_mean"][evaluation] >= mean_threshold
        ),
    }
    metrics = {
        method: classification_metrics(targets[evaluation], prediction)
        for method, prediction in predictions.items()
    }
    deltas = {
        comparator: {
            metric: (
                metrics["adapted_soft_logits"][metric]
                - metrics[comparator][metric]
            )
            for metric in METRIC_KEYS
        }
        for comparator in COMPARATORS
    }
    return {
        "adaptation_record_count": int(adaptation.sum()),
        "evaluation_record_count": int(evaluation.sum()),
        "adaptation_class_counts": {
            "safe": int((targets[adaptation] == 0).sum()),
            "unsafe": int((targets[adaptation] == 1).sum()),
        },
        "evaluation_class_counts": {
            "safe": int((targets[evaluation] == 0).sum()),
            "unsafe": int((targets[evaluation] == 1).sum()),
        },
        "thresholds": {
            "adapted_soft_logits": soft_threshold,
            "adapted_binary_judges": hard_threshold,
            "adapted_calibrated_soft_mean": mean_threshold,
        },
        "soft_model": {
            "intercept": float(soft_model.intercept_[0]),
            "coefficients": [float(value) for value in soft_model.coef_[0]],
        },
        "hard_model": {
            "intercept": float(hard_model.intercept_[0]),
            "coefficients": [float(value) for value in hard_model.coef_[0]],
        },
        "metrics": metrics,
        "primary_minus_comparator": deltas,
    }


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "split_variability_interval": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "positive_fraction": float(np.mean(array > 0.0)),
        "nonnegative_fraction": float(np.mean(array >= 0.0)),
    }


def summarize_repeats(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method_metrics": {
            method: {
                metric: summarize_values(
                    [row["metrics"][method][metric] for row in repeats]
                )
                for metric in METRIC_KEYS
            }
            for method in METHODS
        },
        "primary_minus_comparator": {
            comparator: {
                metric: summarize_values(
                    [
                        row["primary_minus_comparator"][comparator][metric]
                        for row in repeats
                    ]
                )
                for metric in METRIC_KEYS
            }
            for comparator in COMPARATORS
        },
    }


def evaluate_gate(
    dataset_result: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    budget_key = f"{float(protocol['primary_budget_fraction']):.2f}"
    summary = dataset_result["budgets"][budget_key]["summary"]
    gate = protocol["claim_gate"]
    majority = summary["primary_minus_comparator"]["same_pool_majority"][
        "macro_f1"
    ]
    hard = summary["primary_minus_comparator"]["adapted_binary_judges"][
        "macro_f1"
    ]
    checks = {
        "mean_delta_vs_majority": (
            majority["mean"]
            > float(gate["mean_macro_f1_delta_vs_majority_must_exceed"])
        ),
        "positive_fraction_vs_majority": (
            majority["positive_fraction"]
            >= float(gate["positive_repeat_fraction_vs_majority_must_reach"])
        ),
        "mean_delta_vs_adapted_binary": (
            hard["mean"]
            > float(gate["mean_macro_f1_delta_vs_adapted_binary_must_exceed"])
        ),
        "positive_fraction_vs_adapted_binary": (
            hard["positive_fraction"]
            >= float(
                gate["positive_repeat_fraction_vs_adapted_binary_must_reach"]
            )
        ),
    }
    return {
        "budget_fraction": float(protocol["primary_budget_fraction"]),
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "mean_macro_f1_delta_vs_majority": majority["mean"],
            "positive_repeat_fraction_vs_majority": majority[
                "positive_fraction"
            ],
            "mean_macro_f1_delta_vs_adapted_binary": hard["mean"],
            "positive_repeat_fraction_vs_adapted_binary": hard[
                "positive_fraction"
            ],
        },
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# External Small-Budget Adaptation Diagnostic",
        "",
        "This is retrospective exploratory evidence. Both datasets' labels had been opened before the unified budget protocol was frozen. Split intervals report repeated-holdout variability, not confidence intervals.",
        "",
        "## Primary 10% Budget Gate",
        "",
        "| Dataset | Soft Macro-F1 | Majority Macro-F1 | Delta | Positive repeats | Soft minus adapted hard | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset_id, dataset in result["datasets"].items():
        primary = dataset["budgets"]["0.10"]["summary"]
        soft = primary["method_metrics"]["adapted_soft_logits"]["macro_f1"][
            "mean"
        ]
        majority = primary["method_metrics"]["same_pool_majority"]["macro_f1"][
            "mean"
        ]
        delta = primary["primary_minus_comparator"]["same_pool_majority"][
            "macro_f1"
        ]
        hard_delta = primary["primary_minus_comparator"][
            "adapted_binary_judges"
        ]["macro_f1"]
        lines.append(
            f"| {dataset_id} | {soft:.4f} | {majority:.4f} | "
            f"{delta['mean']:+.4f} | {delta['positive_fraction']:.2f} | "
            f"{hard_delta['mean']:+.4f} | "
            f"{'PASS' if dataset['primary_gate']['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Overall predefined gate: **{'PASS' if result['overall_gate']['passed'] else 'FAIL'}**.",
            "",
            "## Budget Curves",
            "",
            "| Dataset | Label budget | Mean soft Macro-F1 | Mean majority Macro-F1 | Mean delta | 95% split-variability interval | Positive repeats |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_id, dataset in result["datasets"].items():
        for budget_key, budget in dataset["budgets"].items():
            summary = budget["summary"]
            soft = summary["method_metrics"]["adapted_soft_logits"]["macro_f1"][
                "mean"
            ]
            majority = summary["method_metrics"]["same_pool_majority"][
                "macro_f1"
            ]["mean"]
            delta = summary["primary_minus_comparator"]["same_pool_majority"][
                "macro_f1"
            ]
            interval = delta["split_variability_interval"]
            lines.append(
                f"| {dataset_id} | {float(budget_key):.0%} | {soft:.4f} | "
                f"{majority:.4f} | {delta['mean']:+.4f} | "
                f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
                f"{delta['positive_fraction']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Passing the gate supports a limited-label adaptation claim, not zero-shot or universal superiority.",
            "- Failing the gate means the existing external scores do not support the planned small-budget claim.",
            "- The manuscript should use these results only after a claim audit because this protocol is retrospective.",
            "",
        ]
    )
    return "\n".join(lines)


def run(protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    output_root = output_root.resolve()
    protocol = load_json(protocol_path)
    if protocol.get("schema") != "external-small-budget-adaptation-protocol-v1":
        raise ValueError("Unsupported small-budget protocol schema")
    if protocol.get("protocol_status") != "frozen-before-budget-sweep-execution":
        raise ValueError("Small-budget protocol is not frozen")
    judges = [str(value) for value in protocol["judges"]]
    repeat_count = int(protocol["repeat_count"])
    base_seed = int(protocol["random_seed"])
    c_value = float(protocol["primary_method"]["C"])
    result: dict[str, Any] = {
        "schema": "external-small-budget-adaptation-results-v1",
        "role": protocol["evidence_role"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "datasets": {},
    }
    for dataset_index, dataset in enumerate(protocol["datasets"]):
        predictions_path = (V15_ROOT / str(dataset["predictions"])).resolve()
        rows = load_jsonl(predictions_path)
        prepared = prepare_dataset(rows, dataset, judges)
        unique_groups = np.unique(prepared["groups"])
        dataset_result: dict[str, Any] = {
            "prediction_path": str(predictions_path),
            "prediction_sha256": sha256_file(predictions_path),
            "sample_count": len(rows),
            "group_count": len(unique_groups),
            "class_counts": {
                "safe": int((prepared["targets"] == 0).sum()),
                "unsafe": int((prepared["targets"] == 1).sum()),
            },
            "budgets": {},
        }
        for fraction in protocol["budget_fractions"]:
            fraction = float(fraction)
            group_count = max(2, int(math.floor(len(unique_groups) * fraction + 0.5)))
            group_count = min(group_count, len(unique_groups) - 1)
            repeats = []
            for repeat in range(repeat_count):
                seed = (
                    base_seed
                    + dataset_index * 1_000_000
                    + int(round(fraction * 10000)) * 1000
                    + repeat
                )
                adaptation, evaluation, selected_groups, resample_attempt = (
                    select_group_split(
                        prepared["groups"],
                        prepared["targets"],
                        group_count,
                        seed,
                    )
                )
                record = evaluate_repeat(
                    prepared, adaptation, evaluation, c_value, seed
                )
                record.update(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "resample_attempt": resample_attempt,
                        "adaptation_group_count": group_count,
                        "evaluation_group_count": len(unique_groups) - group_count,
                        "adaptation_group_set_sha256": sha256_text(
                            canonical_json(selected_groups)
                        ),
                    }
                )
                repeats.append(record)
            budget_key = f"{fraction:.2f}"
            dataset_result["budgets"][budget_key] = {
                "budget_fraction": fraction,
                "adaptation_group_count": group_count,
                "repeat_count": repeat_count,
                "summary": summarize_repeats(repeats),
                "repeats": repeats,
            }
        dataset_result["primary_gate"] = evaluate_gate(dataset_result, protocol)
        result["datasets"][str(dataset["id"])] = dataset_result
    result["overall_gate"] = {
        "all_datasets_required": bool(protocol["claim_gate"]["all_datasets_required"]),
        "passed": all(
            dataset["primary_gate"]["passed"]
            for dataset in result["datasets"].values()
        ),
        "dataset_status": {
            dataset_id: dataset["primary_gate"]["passed"]
            for dataset_id, dataset in result["datasets"].items()
        },
    }
    result["interpretation_constraints"] = list(protocol["anti_claims"])
    result["content_sha256"] = sha256_text(canonical_json(result))
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "results.json"
    report_path = output_root / "REPORT.md"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen cross-dataset small-label-budget diagnostic."
    )
    parser.add_argument(
        "--protocol",
        dest="protocol_path",
        type=Path,
        default=SCRIPT_ROOT / "configs" / "external_small_budget_protocol_v1.json",
    )
    parser.add_argument(
        "--output-root",
        dest="output_root",
        type=Path,
        default=(
            V15_ROOT
            / "results/external/small-budget-adaptation"
        ),
    )
    return parser


if __name__ == "__main__":
    completed = run(**vars(build_parser().parse_args()))
    print(
        json.dumps(
            {
                "passed": completed["overall_gate"]["passed"],
                "dataset_status": completed["overall_gate"]["dataset_status"],
                "content_sha256": completed["content_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
