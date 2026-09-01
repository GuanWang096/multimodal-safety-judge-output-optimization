from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cluster_inference import (  # noqa: E402
    holm_adjust,
    paired_component_bootstrap,
    paired_component_randomization,
    probability_metrics,
    weighted_classification_metrics,
)
from multijudge.baselines import baseline_predictions, rule_equivalence  # noqa: E402
from multijudge.fusion import (  # noqa: E402
    apply_calibrators,
    fusion_model_from_dict,
    predict_calibrated_fusion,
)


JUDGES = ("qwen", "safework", "minicpm")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            values.append(value)
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def verify_embedded_hash(value: Mapping[str, Any], field: str) -> str:
    expected = str(value[field])
    unhashed = dict(value)
    unhashed.pop(field, None)
    if stable_json_sha256(unhashed) != expected:
        raise RuntimeError(f"Embedded hash mismatch: {field}")
    return expected


def mcnemar_exact(
    targets: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict[str, Any]:
    left_correct = left == targets
    right_correct = right == targets
    left_only = int(np.logical_and(left_correct, ~right_correct).sum())
    right_only = int(np.logical_and(~left_correct, right_correct).sum())
    discordant = left_only + right_only
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(left_only, discordant, 0.5).pvalue)
    )
    return {
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant_count": discordant,
        "two_sided_exact_p_value": p_value,
        "role": "record_level_compatibility_appendix_only",
    }


def subgroup_metrics(
    metadata: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(metadata):
        grouped[str(value.get(field) or "unspecified")].append(index)
    output = {}
    for name, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=np.int64)
        output[name] = {
            "record_count": len(indices),
            "unsafe_count": int(targets[selected].sum()),
            "metrics": {
                method: weighted_classification_metrics(
                    targets[selected], values[selected]
                )
                for method, values in predictions.items()
            },
        }
    return output


def comparison(
    *,
    name: str,
    targets: np.ndarray,
    groups: Sequence[str],
    left: np.ndarray,
    right: np.ndarray,
    bootstrap_replicates: int,
    randomization_replicates: int,
    seed: int,
) -> dict[str, Any]:
    left_metrics = weighted_classification_metrics(targets, left)
    right_metrics = weighted_classification_metrics(targets, right)
    bootstrap = paired_component_bootstrap(
        targets=targets,
        groups=groups,
        left_predictions=left,
        right_predictions=right,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    randomization = paired_component_randomization(
        targets=targets,
        groups=groups,
        left_predictions=left,
        right_predictions=right,
        replicates=randomization_replicates,
        seed=seed + 100000,
    )
    return {
        "name": name,
        "difference_direction": "fusion_minus_comparator",
        "point_differences": {
            metric: float(left_metrics[metric]) - float(right_metrics[metric])
            for metric in ("macro_f1", "unsafe_recall", "safe_specificity", "mcc")
        },
        "bootstrap": bootstrap,
        "randomization": randomization,
        "mcnemar": mcnemar_exact(targets, left, right),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "status.json", {"state": "running", "stage": "SF4"})

    manifest_path = args.manifest.resolve()
    labels_path = args.labels.resolve()
    result_root = args.result_root.resolve()
    raw_freeze_path = args.raw_freeze.resolve()
    fusion_config_path = args.fusion_config.resolve()
    analysis_plan_path = args.analysis_plan.resolve()
    manifest = load_jsonl(manifest_path)
    labels = load_jsonl(labels_path)
    raw_freeze = load_json(raw_freeze_path)
    fusion_config = load_json(fusion_config_path)
    analysis_plan = load_json(analysis_plan_path)
    raw_freeze_hash = verify_embedded_hash(raw_freeze, "content_sha256")
    fusion_hash = verify_embedded_hash(fusion_config, "config_sha256")
    analysis_hash = verify_embedded_hash(analysis_plan, "analysis_plan_sha256")
    if raw_freeze.get("status") != "frozen" or raw_freeze.get("labels_accessed") is not False:
        raise RuntimeError("Raw outputs were not frozen before label-aware evaluation")
    if fusion_hash != analysis_plan.get("fusion_config_sha256"):
        raise RuntimeError("Analysis plan does not reference the frozen fusion config")
    if sha256_file(manifest_path) != analysis_plan.get("confirmatory_manifest_sha256"):
        raise RuntimeError("Confirmatory manifest differs from the analysis freeze")

    ordered_ids = [str(value["sample_id"]) for value in manifest]
    manifest_by_id = {str(value["sample_id"]): value for value in manifest}
    labels_by_id = {str(value["sample_id"]): value for value in labels}
    if len(ordered_ids) != 513 or len(manifest_by_id) != 513:
        raise RuntimeError("Unexpected confirmatory manifest membership")
    if set(labels_by_id) != set(ordered_ids) or len(labels_by_id) != len(labels):
        raise RuntimeError("Confirmatory labels do not exactly cover the frozen manifest")
    for sample_id in ordered_ids:
        if str(labels_by_id[sample_id]["component_id"]) != str(
            manifest_by_id[sample_id]["component_id"]
        ):
            raise RuntimeError(f"Component mismatch for {sample_id}")

    raw_probabilities: dict[str, np.ndarray] = {}
    judgment_hashes = {}
    for judge in JUDGES:
        path = result_root / judge / "judgments.jsonl"
        records = load_jsonl(path)
        by_id = {str(value["sample_id"]): value for value in records}
        if len(by_id) != len(records) or [str(value["sample_id"]) for value in records] != ordered_ids:
            raise RuntimeError(f"{judge} outputs do not exactly match confirm order")
        expected_hash = raw_freeze["judges"][judge]["files"]["judgments"]["sha256"]
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise RuntimeError(f"{judge} judgments changed after raw-output freeze")
        expected_version = str(analysis_plan["adapter_versions"][judge])
        versions = {
            str(value["judge_service"]["adapter_version"]) for value in records
        }
        if versions != {expected_version}:
            raise RuntimeError(f"Unexpected {judge} adapter versions: {versions}")
        values = np.asarray(
            [float(by_id[sample_id]["soft_evidence"]["p_unsafe"]) for sample_id in ordered_ids],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite {judge} probabilities")
        raw_probabilities[judge] = values
        judgment_hashes[judge] = observed_hash

    targets = np.asarray(
        [1 if labels_by_id[sample_id]["assistant_label"] == "unsafe" else 0 for sample_id in ordered_ids],
        dtype=np.int64,
    )
    groups = [str(manifest_by_id[sample_id]["component_id"]) for sample_id in ordered_ids]
    inclusion_probabilities = np.asarray(
        [float(manifest_by_id[sample_id]["component_inclusion_probability"]) for sample_id in ordered_ids],
        dtype=np.float64,
    )
    if np.any(inclusion_probabilities <= 0) or not np.all(np.isfinite(inclusion_probabilities)):
        raise RuntimeError("Invalid component inclusion probabilities")
    design_weights = 1.0 / inclusion_probabilities

    model = fusion_model_from_dict(fusion_config["model"])
    fusion_probabilities = predict_calibrated_fusion(
        raw_probabilities,
        fusion_config["final_development_calibrators"],
        model,
    )
    fusion_threshold = float(fusion_config["threshold"])
    fusion_predictions = (fusion_probabilities >= fusion_threshold).astype(np.int64)
    calibrated = apply_calibrators(
        raw_probabilities,
        fusion_config["final_development_calibrators"],
        JUDGES,
    )
    calibrated_mean_probabilities = np.column_stack(
        [calibrated[judge] for judge in JUDGES]
    ).mean(axis=1)
    calibrated_mean_predictions = (
        calibrated_mean_probabilities
        >= float(analysis_plan["calibrated_soft_mean"]["threshold"])
    ).astype(np.int64)
    hard_baselines = baseline_predictions(raw_probabilities, JUDGES)

    predictions: dict[str, np.ndarray] = {
        "fusion": fusion_predictions,
        **{name: np.asarray(values, dtype=np.int64) for name, values in hard_baselines.items()},
        "calibrated_soft_mean": calibrated_mean_predictions,
    }
    probabilities: dict[str, np.ndarray] = {
        "fusion": fusion_probabilities,
        "calibrated_soft_mean": calibrated_mean_probabilities,
        **raw_probabilities,
    }

    leave_one_out = {}
    for omitted in JUDGES:
        config = analysis_plan["leave_one_judge_out"][omitted]
        reduced_model = fusion_model_from_dict(config["model"])
        reduced_probabilities = predict_calibrated_fusion(
            raw_probabilities, config["calibrators"], reduced_model
        )
        reduced_predictions = (
            reduced_probabilities >= float(config["threshold"])
        ).astype(np.int64)
        predictions[f"without_{omitted}"] = reduced_predictions
        probabilities[f"without_{omitted}"] = reduced_probabilities

    metrics = {
        name: weighted_classification_metrics(targets, values)
        for name, values in predictions.items()
    }
    probability_results = {
        name: probability_metrics(targets, values)
        for name, values in probabilities.items()
    }
    design_weighted = {
        name: weighted_classification_metrics(targets, values, design_weights)
        for name, values in predictions.items()
    }

    bootstrap_replicates = int(analysis_plan["statistics"]["bootstrap_replicates"])
    randomization_replicates = int(
        analysis_plan["statistics"]["randomization_replicates"]
    )
    comparators = [
        "majority",
        str(analysis_plan["strongest_simple_baseline"]),
        "calibrated_soft_mean",
        "single_qwen",
        "single_safework",
        "single_minicpm",
    ]
    comparisons = {}
    for offset, comparator in enumerate(dict.fromkeys(comparators)):
        comparisons[comparator] = comparison(
            name=f"fusion_vs_{comparator}",
            targets=targets,
            groups=groups,
            left=fusion_predictions,
            right=predictions[comparator],
            bootstrap_replicates=bootstrap_replicates,
            randomization_replicates=randomization_replicates,
            seed=int(analysis_plan["seed_base"]) + 410000 + offset,
        )
    adjusted = holm_adjust(
        {
            name: value["randomization"]["one_sided_p_value"]
            for name, value in comparisons.items()
            if name != "majority"
        }
    )
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_one_sided_p_value"] = value

    reference_name = str(analysis_plan["risk_constraints"]["reference"])
    reference_metrics = metrics[reference_name]
    risk_gate = {
        "unsafe_recall": float(metrics["fusion"]["unsafe_recall"])
        >= float(reference_metrics["unsafe_recall"])
        - float(
            analysis_plan["risk_constraints"][
                "unsafe_recall_margin_below_reference"
            ]
        ),
        "safe_specificity": float(metrics["fusion"]["safe_specificity"])
        >= float(reference_metrics["safe_specificity"])
        - float(
            analysis_plan["risk_constraints"][
                "safe_specificity_margin_below_reference"
            ]
        ),
    }
    risk_gate["passed"] = all(risk_gate.values())

    primary_ci = comparisons["majority"]["bootstrap"]["metrics"]["macro_f1"]["ci95"]
    strongest_ci = comparisons[reference_name]["bootstrap"]["metrics"]["macro_f1"]["ci95"]
    mean_ci = comparisons["calibrated_soft_mean"]["bootstrap"]["metrics"]["macro_f1"]["ci95"]
    majority_gate = bool(primary_ci[0] > 0 and risk_gate["passed"])
    strongest_gate = bool(majority_gate and strongest_ci[0] > 0)
    soft_mean_gate = bool(mean_ci[0] > 0)

    marginal = {}
    marginal_config = analysis_plan["marginal_gate"]
    for offset, omitted in enumerate(JUDGES):
        name = f"without_{omitted}"
        comparison_value = comparison(
            name=f"fusion_vs_{name}",
            targets=targets,
            groups=groups,
            left=fusion_predictions,
            right=predictions[name],
            bootstrap_replicates=bootstrap_replicates,
            randomization_replicates=randomization_replicates,
            seed=int(analysis_plan["seed_base"]) + 420000 + offset,
        )
        change_rate = float(np.mean(fusion_predictions != predictions[name]))
        point = comparison_value["point_differences"]
        positive_fraction = float(
            comparison_value["bootstrap"]["metrics"]["macro_f1"][
                "positive_fraction"
            ]
        )
        gate = {
            "changes_at_least_one_percent": change_rate
            >= float(marginal_config["minimum_hard_prediction_change_rate"]),
            "positive_in_required_bootstraps": positive_fraction
            >= float(
                marginal_config["minimum_positive_component_bootstrap_fraction"]
            ),
            "nonnegative_point_unsafe_recall": float(point["unsafe_recall"]) >= 0,
        }
        gate["passed"] = all(gate.values())
        marginal[omitted] = {
            "hard_prediction_change_rate": change_rate,
            "comparison": comparison_value,
            "gate": gate,
        }
    marginal_gate = all(value["gate"]["passed"] for value in marginal.values())

    equivalence = rule_equivalence(
        fusion_predictions,
        {
            **hard_baselines,
            "calibrated_soft_mean": calibrated_mean_predictions,
        },
    )
    nondegenerate_gate = not equivalence["equivalent_rules"]

    metadata = [manifest_by_id[sample_id] for sample_id in ordered_ids]
    subgroup = {
        field: subgroup_metrics(metadata, targets, predictions, field)
        for field in (
            "target_model",
            "image_count_bucket",
            "data_type",
            "coarse_stratum",
        )
    }
    prediction_rows = []
    for index, sample_id in enumerate(ordered_ids):
        prediction_rows.append(
            {
                "schema": "softfusion-confirmatory-prediction-v1",
                "sample_id": sample_id,
                "component_id": groups[index],
                "expected_label": "unsafe" if targets[index] else "safe",
                "raw_p_unsafe": {
                    judge: float(raw_probabilities[judge][index]) for judge in JUDGES
                },
                "calibrated_p_unsafe": {
                    judge: float(calibrated[judge][index]) for judge in JUDGES
                },
                "fusion_p_unsafe": float(fusion_probabilities[index]),
                "fusion_prediction": "unsafe" if fusion_predictions[index] else "safe",
                "baseline_predictions": {
                    name: "unsafe" if values[index] else "safe"
                    for name, values in predictions.items()
                    if name != "fusion"
                },
                "component_inclusion_probability": float(
                    inclusion_probabilities[index]
                ),
            }
        )
    write_jsonl(output_root / "confirmatory_predictions.jsonl", prediction_rows)

    if majority_gate and strongest_gate and soft_mean_gate and marginal_gate and nondegenerate_gate:
        claim_status = "supports_improvement_over_majority_and_prespecified_simple_baselines"
    elif majority_gate:
        claim_status = "supports_improvement_over_majority_only"
    else:
        claim_status = "does_not_support_optimization_claim"
    result: dict[str, Any] = {
        "schema": "softfusion-confirmatory-results-v1",
        "status": "completed",
        "analysis_locked": True,
        "confirmatory_record_count": len(targets),
        "confirmatory_component_count": len(set(groups)),
        "label_counts": {
            "safe": int((targets == 0).sum()),
            "unsafe": int((targets == 1).sum()),
        },
        "metrics": metrics,
        "probability_metrics": probability_results,
        "design_weighted_sensitivity": design_weighted,
        "comparisons": comparisons,
        "risk_gate": risk_gate,
        "hierarchical_gates": {
            "fusion_vs_majority": majority_gate,
            "fusion_vs_strongest_simple": strongest_gate,
            "fusion_vs_calibrated_soft_mean": soft_mean_gate,
            "all_judges_positive_marginal_contribution": marginal_gate,
            "not_equivalent_to_simple_rule": nondegenerate_gate,
        },
        "leave_one_judge_out": marginal,
        "rule_equivalence": equivalence,
        "subgroup_diagnostics": subgroup,
        "claim_status": claim_status,
        "claim_constraints": [
            "The confirmatory split is an analysis-locked MMDS train component sample, not a new public benchmark.",
            "Multi-view consistency and pairwise interaction claims remain excluded.",
            "Subgroup analyses are descriptive and do not replace the primary endpoint.",
        ],
        "provenance": {
            "manifest_sha256": sha256_file(manifest_path),
            "labels_sha256": sha256_file(labels_path),
            "raw_output_freeze_content_sha256": raw_freeze_hash,
            "fusion_config_sha256": fusion_hash,
            "analysis_plan_sha256": analysis_hash,
            "judgment_sha256": judgment_hashes,
        },
    }
    result["result_content_sha256"] = stable_json_sha256(result)
    write_json(output_root / "confirmatory_results.json", result)
    write_json(
        output_root / "status.json",
        {
            "state": "completed",
            "stage": "SF4",
            "claim_status": claim_status,
            "result_content_sha256": result["result_content_sha256"],
        },
    )
    report = [
        "# SF4 Confirmatory Evaluation",
        "",
        f"**Claim status:** `{claim_status}`",
        "",
        "| Method | Macro-F1 | Unsafe recall | Specificity | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "fusion",
        "majority",
        reference_name,
        "calibrated_soft_mean",
        "single_qwen",
        "single_safework",
        "single_minicpm",
    ):
        value = metrics[name]
        report.append(
            f"| {name} | {value['macro_f1']:.4f} | {value['unsafe_recall']:.4f} | {value['safe_specificity']:.4f} | {value['mcc']:.4f} |"
        )
    report.extend(
        [
            "",
            "## Frozen gates",
            "",
            *[
                f"- `{name}`: `{value}`"
                for name, value in result["hierarchical_gates"].items()
            ],
            "",
            "All confidence intervals use paired component resampling.",
            "",
        ]
    )
    (output_root / "CONFIRMATORY_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    base = V15_ROOT / "results" / "mmds"
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=base / "data" / "confirm-unlabeled.jsonl")
    parser.add_argument("--labels", type=Path, default=base / "data" / "confirm-labels.jsonl")
    parser.add_argument("--result-root", type=Path, default=base / "confirm")
    parser.add_argument("--raw-freeze", type=Path, default=base / "confirm" / "raw_output_freeze.json")
    parser.add_argument("--fusion-config", type=Path, default=base / "fit" / "fusion_config.json")
    parser.add_argument("--analysis-plan", type=Path, default=base / "fit" / "confirmatory_analysis_plan.json")
    parser.add_argument("--output-root", type=Path, default=base / "evaluation")
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "claim_status": result["claim_status"],
                "record_count": result["confirmatory_record_count"],
                "component_count": result["confirmatory_component_count"],
            }
        )
    )
