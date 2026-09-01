from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_inference import (  # noqa: E402
    holm_adjust,
    paired_component_bootstrap,
    paired_component_randomization,
    probability_metrics,
    weighted_classification_metrics,
)
from multijudge.baselines import baseline_predictions  # noqa: E402
from multijudge.fusion import (  # noqa: E402
    apply_calibrators,
    fusion_model_from_dict,
    predict_calibrated_fusion,
)
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


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")
    os.replace(temporary, path)


def _verify_embedded_hash(value: dict[str, Any], field: str) -> str:
    observed = str(value.get(field))
    unhashed = dict(value)
    unhashed.pop(field, None)
    if observed != sha256_text(canonical_json(unhashed)):
        raise RuntimeError(f"Invalid embedded hash: {field}")
    return observed


def _probability_metrics(targets: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    result = probability_metrics(targets, values)
    result.update(
        {
            "auroc": float(roc_auc_score(targets, values)),
            "average_precision": float(average_precision_score(targets, values)),
        }
    )
    return result


def _comparison(
    *,
    targets: np.ndarray,
    groups: Sequence[str],
    fusion: np.ndarray,
    comparator: np.ndarray,
    name: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left = weighted_classification_metrics(targets, fusion)
    right = weighted_classification_metrics(targets, comparator)
    return {
        "name": name,
        "difference_direction": "frozen_fusion_minus_comparator",
        "point_differences": {
            metric: float(left[metric]) - float(right[metric])
            for metric in ("macro_f1", "unsafe_recall", "safe_specificity", "mcc")
        },
        "bootstrap": paired_component_bootstrap(
            targets=targets,
            groups=groups,
            left_predictions=fusion,
            right_predictions=comparator,
            replicates=replicates,
            seed=seed,
        ),
        "randomization": paired_component_randomization(
            targets=targets,
            groups=groups,
            left_predictions=fusion,
            right_predictions=comparator,
            replicates=replicates,
            seed=seed + 100000,
        ),
    }


def _subgroup_metrics(
    metadata: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(metadata):
        grouped[str(value[field])].append(index)
    output = {}
    for group, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=np.int64)
        output[group] = {
            "record_count": len(indices),
            "unsafe_count": int(targets[selected].sum()),
            "metrics": {
                name: weighted_classification_metrics(
                    targets[selected], values[selected]
                )
                for name, values in predictions.items()
            },
        }
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    labels_path = args.labels.resolve()
    result_root = args.result_root.resolve()
    fusion_path = args.fusion_config.resolve()
    plan_path = args.analysis_plan.resolve()
    freeze_path = args.raw_freeze.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = _load_jsonl(manifest_path)
    labels = _load_jsonl(labels_path)
    fusion_config = _load_json(fusion_path)
    plan = _load_json(plan_path)
    raw_freeze = _load_json(freeze_path)
    plan_hash = _verify_embedded_hash(plan, "analysis_plan_sha256")
    fusion_hash = _verify_embedded_hash(fusion_config, "config_sha256")
    freeze_hash = _verify_embedded_hash(raw_freeze, "content_sha256")
    if raw_freeze.get("status") != "frozen" or raw_freeze.get("labels_accessed") is not False:
        raise RuntimeError("External raw outputs were not frozen before evaluation")
    if plan.get("fusion_config_sha256") != fusion_hash:
        raise RuntimeError("External plan does not reference the frozen fusion config")
    if raw_freeze.get("analysis_plan_sha256") != plan_hash:
        raise RuntimeError("Raw-output freeze does not reference the external plan")
    if plan.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("External manifest differs from frozen plan")
    if plan.get("labels_sha256") != sha256_file(labels_path):
        raise RuntimeError("External labels differ from frozen plan")

    ordered_ids = [str(value["sample_id"]) for value in manifest]
    labels_by_id = {str(value["sample_id"]): value for value in labels}
    if len(ordered_ids) != 1180 or set(ordered_ids) != set(labels_by_id):
        raise RuntimeError("External labels do not exactly cover the frozen manifest")
    raw_probabilities: dict[str, np.ndarray] = {}
    judgment_hashes = {}
    for judge in plan["judge_order"]:
        path = result_root / str(judge) / "judgments.jsonl"
        records = _load_jsonl(path)
        if [str(value["sample_id"]) for value in records] != ordered_ids:
            raise RuntimeError(f"External output order mismatch for {judge}")
        expected_hash = raw_freeze["judges"][judge]["files"]["judgments"]["sha256"]
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"External judgments changed after freeze for {judge}")
        raw_probabilities[str(judge)] = np.asarray(
            [float(value["soft_evidence"]["p_unsafe"]) for value in records],
            dtype=np.float64,
        )
        judgment_hashes[str(judge)] = expected_hash

    targets = np.asarray(
        [1 if labels_by_id[sample_id]["assistant_label"] == "unsafe" else 0 for sample_id in ordered_ids],
        dtype=np.int64,
    )
    groups = [str(value["group_id"]) for value in manifest]
    model = fusion_model_from_dict(fusion_config["model"])
    fusion_probabilities = predict_calibrated_fusion(
        raw_probabilities,
        fusion_config["final_development_calibrators"],
        model,
    )
    fusion_predictions = (
        fusion_probabilities >= float(plan["transfer_contract"]["fusion_threshold"])
    ).astype(np.int64)
    calibrated = apply_calibrators(
        raw_probabilities,
        fusion_config["final_development_calibrators"],
        plan["judge_order"],
    )
    soft_mean_probabilities = np.column_stack(
        [calibrated[judge] for judge in plan["judge_order"]]
    ).mean(axis=1)
    soft_mean_predictions = (
        soft_mean_probabilities
        >= float(plan["transfer_contract"]["soft_mean_threshold"])
    ).astype(np.int64)
    hard = baseline_predictions(raw_probabilities, plan["judge_order"])
    predictions = {
        "frozen_fusion": fusion_predictions,
        "calibrated_soft_mean": soft_mean_predictions,
        **hard,
    }
    probabilities = {
        "frozen_fusion": fusion_probabilities,
        "calibrated_soft_mean": soft_mean_probabilities,
        **raw_probabilities,
    }
    metrics = {
        name: weighted_classification_metrics(targets, values)
        for name, values in predictions.items()
    }
    probability_results = {
        name: _probability_metrics(targets, values)
        for name, values in probabilities.items()
    }

    comparisons = {}
    comparison_names = ("majority", "or_1_of_n", "calibrated_soft_mean")
    replicates = int(plan["statistics"]["bootstrap_replicates"])
    seed = int(plan["statistics"]["seed"])
    for offset, name in enumerate(comparison_names):
        comparisons[name] = _comparison(
            targets=targets,
            groups=groups,
            fusion=fusion_predictions,
            comparator=predictions[name],
            name=f"frozen_fusion_vs_{name}",
            replicates=replicates,
            seed=seed + offset * 1000,
        )
    adjusted = holm_adjust(
        {
            name: float(value["randomization"]["one_sided_p_value"])
            for name, value in comparisons.items()
        }
    )
    for name, adjusted_value in adjusted.items():
        comparisons[name]["randomization"]["holm_one_sided_p_value"] = adjusted_value

    primary_delta = float(comparisons["majority"]["point_differences"]["macro_f1"])
    primary_ci = comparisons["majority"]["bootstrap"]["metrics"]["macro_f1"]["ci95"]
    prediction_rows = []
    for index, entry in enumerate(manifest):
        sample_id = ordered_ids[index]
        prediction_rows.append(
            {
                "schema": "beavertails-v-frozen-transfer-prediction-v1",
                "sample_id": sample_id,
                "group_id": entry["group_id"],
                "category": entry["category"],
                "image_severity": entry["image_severity"],
                "expected_label": "unsafe" if targets[index] else "safe",
                "raw_p_unsafe": {
                    judge: float(raw_probabilities[judge][index])
                    for judge in plan["judge_order"]
                },
                "calibrated_p_unsafe": {
                    judge: float(calibrated[judge][index])
                    for judge in plan["judge_order"]
                },
                "fusion_p_unsafe": float(fusion_probabilities[index]),
                "predictions": {
                    name: "unsafe" if values[index] else "safe"
                    for name, values in predictions.items()
                },
            }
        )
    _write_jsonl(output_root / "predictions.jsonl", prediction_rows)
    result = {
        "schema": "beavertails-v-frozen-transfer-results-v1",
        "status": "completed",
        "integrity_passed": True,
        "dataset": "PKU-Alignment/BeaverTails-V",
        "sample_count": len(targets),
        "paired_group_count": len(set(groups)),
        "label_counts": {
            "safe": int((targets == 0).sum()),
            "unsafe": int((targets == 1).sum()),
        },
        "zero_shot_transfer": {
            "external_refitting": False,
            "external_recalibration": False,
            "external_threshold_selection": False,
            "fusion_config_sha256": fusion_hash,
        },
        "metrics": metrics,
        "probability_metrics": probability_results,
        "comparisons": comparisons,
        "category_metrics": _subgroup_metrics(
            manifest, targets, predictions, "category"
        ),
        "severity_metrics": _subgroup_metrics(
            manifest, targets, predictions, "image_severity"
        ),
        "primary_assessment": {
            "comparison": "frozen_fusion_vs_majority",
            "macro_f1_point_delta": primary_delta,
            "macro_f1_cluster_bootstrap_ci95": primary_ci,
            "support_rule_passed": primary_delta > 0.0,
            "strong_support_rule_passed": float(primary_ci[0]) > 0.0,
        },
        "provenance": {
            "manifest_sha256": sha256_file(manifest_path),
            "labels_sha256": sha256_file(labels_path),
            "analysis_plan_sha256": plan_hash,
            "raw_output_freeze_sha256": freeze_hash,
            "fusion_config_sha256": fusion_hash,
            "judgment_sha256": judgment_hashes,
        },
        "labels_accessed_after_raw_output_freeze": True,
        "external_labels_used_for_model_selection": False,
    }
    result["content_sha256"] = sha256_text(canonical_json(result))
    _write_json(output_root / "results.json", result)
    _write_json(
        output_root / "status.json",
        {
            "state": "completed",
            "integrity_passed": True,
            "support_rule_passed": primary_delta > 0.0,
            "strong_support_rule_passed": float(primary_ci[0]) > 0.0,
        },
    )
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
        "--manifest",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "beavertails_v_external_unlabeled.jsonl"
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=base / "preparation" / "beavertails_v_external_labels.jsonl",
    )
    parser.add_argument("--result-root", type=Path, default=base / "raw")
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=(
            V15_ROOT
            / "results"
            / "mmds"
            / "fit"
            / "fusion_config.json"
        ),
    )
    parser.add_argument("--analysis-plan", type=Path, default=base / "analysis_plan.json")
    parser.add_argument("--raw-freeze", type=Path, default=base / "raw_output_freeze.json")
    parser.add_argument("--output-root", type=Path, default=base)
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "integrity_passed": result["integrity_passed"],
                "sample_count": result["sample_count"],
                "primary_assessment": result["primary_assessment"],
            },
            indent=2,
        )
    )
