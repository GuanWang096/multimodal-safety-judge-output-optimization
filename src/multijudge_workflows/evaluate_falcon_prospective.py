from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


PRIMARY_METHOD = "adapted_soft_logits"
PRIMARY_COMPARATORS = {
    "FALCON-C1": "same_pool_majority",
    "FALCON-C2": "adapted_hard_logits",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


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


def _create_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")
    os.replace(temporary, path)


def _verify_embedded_hash(value: dict[str, Any], field: str) -> str:
    observed = str(value.get(field, ""))
    unhashed = dict(value)
    unhashed.pop(field, None)
    if observed != sha256_text(canonical_json(unhashed)):
        raise RuntimeError(f"Invalid embedded hash: {field}")
    return observed


def _comparison(
    targets: np.ndarray,
    groups: Sequence[str],
    left: np.ndarray,
    right: np.ndarray,
    bootstrap_replicates: int,
    randomization_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    left_metrics = weighted_classification_metrics(targets, left)
    right_metrics = weighted_classification_metrics(targets, right)
    return {
        "difference_direction": "adapted_soft_logits_minus_comparator",
        "point_differences": {
            metric: float(left_metrics[metric]) - float(right_metrics[metric])
            for metric in ("macro_f1", "unsafe_recall", "safe_specificity", "mcc")
        },
        "bootstrap": paired_component_bootstrap(
            targets=targets,
            groups=groups,
            left_predictions=left,
            right_predictions=right,
            replicates=bootstrap_replicates,
            seed=seed,
            confidence_level=confidence_level,
        ),
        "randomization": paired_component_randomization(
            targets=targets,
            groups=groups,
            left_predictions=left,
            right_predictions=right,
            replicates=randomization_replicates,
            seed=seed + 100000,
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = args.predictions.resolve()
    model_freeze_path = args.model_freeze.resolve()
    model_freeze_audit_path = args.model_freeze_audit.resolve()
    confirmation_labels_path = args.confirmation_labels.resolve()
    confirmation_seal_path = args.confirmation_seal.resolve()
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = _load_jsonl(predictions_path)
    model_freeze = _load_json(model_freeze_path)
    model_freeze_audit = _load_json(model_freeze_audit_path)
    seal = _load_json(confirmation_seal_path)
    protocol = _load_json(protocol_path)
    model_freeze_hash = _verify_embedded_hash(model_freeze, "content_sha256")
    model_freeze_audit_hash = _verify_embedded_hash(
        model_freeze_audit, "content_sha256"
    )
    seal_hash = _verify_embedded_hash(seal, "seal_sha256")
    if model_freeze.get("status") != "frozen-before-confirmation-label-access":
        raise RuntimeError("Falcon adapted models were not frozen")
    if model_freeze_audit.get("passed") is not True:
        raise RuntimeError("Falcon adapted model freeze audit did not pass")
    if model_freeze_audit.get("confirmation_labels_accessed") is not False:
        raise RuntimeError("Falcon model-freeze audit reports confirmation access")
    if model_freeze_audit.get("adapted_model_freeze_sha256") != model_freeze_hash:
        raise RuntimeError("Falcon model-freeze audit references another freeze")
    if model_freeze.get("confirmation_labels_opened") is not False:
        raise RuntimeError("Falcon model freeze reports prior confirmation access")
    opening_path = output_root / "confirmation_opening.json"
    if opening_path.exists():
        raise RuntimeError("Falcon confirmation labels already have an opening record")
    if seal.get("status") != "sealed-not-opened" or seal.get("opened") is not False:
        raise RuntimeError("Falcon confirmation-label seal is not closed")
    if model_freeze.get("confirmation_label_seal_sha256") != seal_hash:
        raise RuntimeError("Falcon model freeze references a different label seal")
    if model_freeze.get("predictions_file_sha256") != sha256_file(predictions_path):
        raise RuntimeError("Falcon confirmation predictions changed after freeze")
    if model_freeze.get("protocol_file_sha256") != sha256_file(protocol_path):
        raise RuntimeError("Falcon protocol changed after model freeze")
    if seal.get("confirmation_label_file_sha256") != sha256_file(
        confirmation_labels_path
    ):
        raise RuntimeError("Falcon confirmation labels differ from the sealed file")

    opened_at = datetime.now(timezone.utc).isoformat()
    opening_record = {
        "schema": "falcon-confirmation-opening-v2",
        "status": "opening-consumed-before-label-read",
        "opened_at_utc": opened_at,
        "confirmation_label_seal_sha256": seal_hash,
        "confirmation_labels_sha256": seal["confirmation_label_file_sha256"],
        "adapted_model_freeze_sha256": model_freeze_hash,
        "adapted_model_freeze_audit_sha256": model_freeze_audit_hash,
        "frozen_predictions_sha256": sha256_file(predictions_path),
        "expected_confirmation_sample_count": int(seal["confirmation_sample_count"]),
        "opening_count": 1,
    }
    opening_record["content_sha256"] = sha256_text(canonical_json(opening_record))
    _create_json_exclusive(opening_path, opening_record)

    consumed_seal = dict(seal)
    consumed_seal.pop("seal_sha256", None)
    consumed_seal.update(
        {
            "status": "opened-for-prespecified-evaluation",
            "opened": True,
            "opened_at_utc": opened_at,
            "opening_count": 1,
            "closed_seal_sha256": seal_hash,
            "opening_record_sha256": opening_record["content_sha256"],
        }
    )
    consumed_seal["seal_sha256"] = sha256_text(canonical_json(consumed_seal))
    _write_json(confirmation_seal_path, consumed_seal)

    labels = _load_jsonl(confirmation_labels_path)
    label_by_id = {str(record["sample_id"]): record for record in labels}
    prediction_ids = [str(record["sample_id"]) for record in predictions]
    if len(label_by_id) != len(labels) or set(label_by_id) != set(prediction_ids):
        raise RuntimeError("Falcon confirmation labels and frozen predictions do not align")
    if any(
        record.get("prospective_partition") != "confirmation" for record in labels
    ):
        raise RuntimeError("Non-confirmation label found after opening the seal")

    targets = np.asarray(
        [
            1 if label_by_id[sample_id]["assistant_label"] == "unsafe" else 0
            for sample_id in prediction_ids
        ],
        dtype=np.int64,
    )
    groups = [str(record["group_id"]) for record in predictions]
    method_names = tuple(predictions[0]["predictions"])
    method_predictions = {
        method: np.asarray(
            [
                1 if record["predictions"][method] == "unsafe" else 0
                for record in predictions
            ],
            dtype=np.int64,
        )
        for method in method_names
    }
    probability_names = tuple(predictions[0]["probabilities"])
    probabilities = {
        method: np.asarray(
            [float(record["probabilities"][method]) for record in predictions],
            dtype=np.float64,
        )
        for method in probability_names
    }
    metrics = {
        method: weighted_classification_metrics(targets, values)
        for method, values in method_predictions.items()
    }
    probability_results = {}
    for method, values in probabilities.items():
        current = probability_metrics(targets, values)
        current["auroc"] = float(roc_auc_score(targets, values))
        current["average_precision"] = float(average_precision_score(targets, values))
        probability_results[method] = current

    statistics = protocol["claims_and_endpoints"]
    bootstrap_replicates = int(statistics["bootstrap_replicates"])
    randomization_replicates = int(statistics["randomization_replicates"])
    confidence_level = float(statistics["confidence_level"])
    seed = int(statistics["statistical_seed"])
    comparisons = {
        claim_id: _comparison(
            targets,
            groups,
            method_predictions[PRIMARY_METHOD],
            method_predictions[comparator],
            bootstrap_replicates,
            randomization_replicates,
            confidence_level,
            seed + index * 1000,
        )
        for index, (claim_id, comparator) in enumerate(PRIMARY_COMPARATORS.items())
    }
    adjusted = holm_adjust(
        {
            claim_id: float(value["randomization"]["one_sided_p_value"])
            for claim_id, value in comparisons.items()
        }
    )
    assessments = {}
    for claim_id, comparison in comparisons.items():
        adjusted_p = float(adjusted[claim_id])
        comparison["randomization"]["holm_one_sided_p_value"] = adjusted_p
        delta = float(comparison["point_differences"]["macro_f1"])
        ci95 = comparison["bootstrap"]["metrics"]["macro_f1"]["ci95"]
        assessments[claim_id] = {
            "comparator": PRIMARY_COMPARATORS[claim_id],
            "macro_f1_point_delta": delta,
            "macro_f1_cluster_bootstrap_ci95": ci95,
            "holm_one_sided_p_value": adjusted_p,
            "support_rule_passed": delta > 0.0 and adjusted_p < 0.05,
            "strong_support_rule_passed": (
                delta > 0.0 and adjusted_p < 0.05 and float(ci95[0]) > 0.0
            ),
        }

    labeled_predictions = []
    for record in predictions:
        sample_id = str(record["sample_id"])
        labeled_predictions.append(
            {
                **record,
                "expected_label": label_by_id[sample_id]["assistant_label"],
                "harmful_category": label_by_id[sample_id]["harmful_category"],
            }
        )
    labeled_predictions_path = output_root / "confirmation_predictions_labeled.jsonl"
    _write_jsonl(labeled_predictions_path, labeled_predictions)

    result = {
        "schema": "falcon-prospective-confirmation-results-v1",
        "status": "completed",
        "integrity_passed": True,
        "dataset": "zhangrjjj/Falcon",
        "subset": "Falcon-test",
        "confirmation_sample_count": len(targets),
        "confirmation_group_count": len(set(groups)),
        "label_counts": dict(
            sorted(Counter("unsafe" if value else "safe" for value in targets).items())
        ),
        "metrics": metrics,
        "probability_metrics": probability_results,
        "primary_comparisons": comparisons,
        "primary_assessments": assessments,
        "provenance": {
            "protocol_file_sha256": sha256_file(protocol_path),
            "adapted_model_freeze_sha256": model_freeze_hash,
            "adapted_model_freeze_audit_sha256": model_freeze_audit_hash,
            "confirmation_label_seal_sha256": seal_hash,
            "confirmation_labels_sha256": sha256_file(confirmation_labels_path),
            "frozen_predictions_sha256": sha256_file(predictions_path),
            "labeled_predictions_sha256": sha256_file(labeled_predictions_path),
            "confirmation_opening_record_sha256": opening_record["content_sha256"],
            "consumed_confirmation_seal_sha256": consumed_seal["seal_sha256"],
        },
        "confirmation_labels_opened_after_prediction_freeze": True,
        "confirmation_opening_count": 1,
    }
    result["content_sha256"] = sha256_text(canonical_json(result))
    _write_json(output_root / "results.json", result)
    _write_json(
        output_root / "status.json",
        {
            "state": "completed",
            "integrity_passed": True,
            "primary_assessments": assessments,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    experiment_root = Path(__file__).resolve().parent
    base = (
        V15_ROOT
        / "results"
        / "external"
        / "falcon_test"
    )
    model_root = base
    parser.add_argument(
        "--predictions",
        type=Path,
        default=model_root / "confirmation_predictions.frozen.jsonl",
    )
    parser.add_argument(
        "--model-freeze",
        type=Path,
        default=model_root / "adapted_model_freeze.json",
    )
    parser.add_argument(
        "--model-freeze-audit",
        type=Path,
        default=model_root / "model_freeze_audit.json",
    )
    parser.add_argument(
        "--confirmation-labels",
        type=Path,
        default=base
        / "preparation"
        / "falcon_confirmation_labels.sealed.jsonl",
    )
    parser.add_argument(
        "--confirmation-seal",
        type=Path,
        default=base / "preparation" / "confirmation_label_seal.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=experiment_root / "configs" / "falcon_prospective_protocol_v1.json",
    )
    parser.add_argument("--output-root", type=Path, default=base)
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(
        json.dumps(
            {
                "integrity_passed": result["integrity_passed"],
                "confirmation_sample_count": result["confirmation_sample_count"],
                "primary_assessments": result["primary_assessments"],
            },
            indent=2,
        )
    )
