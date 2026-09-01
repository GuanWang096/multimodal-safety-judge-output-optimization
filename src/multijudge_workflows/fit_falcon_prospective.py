from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.baselines import baseline_predictions  # noqa: E402
from multijudge.fusion import (  # noqa: E402
    apply_calibrators,
    fusion_model_from_dict,
    predict_calibrated_fusion,
)
from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


JUDGES = ("qwen", "safework", "minicpm")
EPSILON = 1e-6


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


def _fit(features: np.ndarray, targets: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        C=0.03,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-9,
        class_weight=None,
        random_state=20260802,
    )
    model.fit(features, targets)
    return model


def _model_record(model: LogisticRegression, feature_names: Sequence[str]) -> dict[str, Any]:
    return {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "feature_names": list(feature_names),
        "penalty": "l2",
        "C": 0.03,
        "solver": "lbfgs",
        "max_iter": 10000,
        "tol": 1e-9,
        "class_weight": None,
        "random_state": 20260802,
        "decision_threshold": 0.5,
        "intercept": float(model.intercept_[0]),
        "coefficients": model.coef_[0].tolist(),
        "n_iter": int(model.n_iter_[0]),
    }


def fit_and_freeze(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    adaptation_labels_path = args.adaptation_labels.resolve()
    confirmation_seal_path = args.confirmation_seal.resolve()
    raw_freeze_path = args.raw_freeze.resolve()
    result_root = args.result_root.resolve()
    fusion_config_path = args.fusion_config.resolve()
    protocol_path = args.protocol.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = _load_jsonl(manifest_path)
    adaptation_labels = _load_jsonl(adaptation_labels_path)
    confirmation_seal = _load_json(confirmation_seal_path)
    raw_freeze = _load_json(raw_freeze_path)
    fusion_config = _load_json(fusion_config_path)
    protocol = _load_json(protocol_path)
    raw_freeze_hash = _verify_embedded_hash(raw_freeze, "content_sha256")
    confirmation_seal_hash = _verify_embedded_hash(confirmation_seal, "seal_sha256")
    fusion_hash = _verify_embedded_hash(fusion_config, "config_sha256")

    if raw_freeze.get("status") != "frozen-before-adaptation-label-access":
        raise RuntimeError("Falcon raw outputs are not frozen")
    if raw_freeze.get("labels_accessed") is not False:
        raise RuntimeError("Falcon raw-output freeze reports label access")
    if confirmation_seal.get("status") != "sealed-not-opened":
        raise RuntimeError("Falcon confirmation-label seal is not closed")
    if confirmation_seal.get("opened") is not False:
        raise RuntimeError("Falcon confirmation labels were already opened")
    if raw_freeze.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Falcon manifest differs from raw-output freeze")
    if raw_freeze.get("protocol_file_sha256") != sha256_file(protocol_path):
        raise RuntimeError("Falcon protocol differs from raw-output freeze")
    if tuple(raw_freeze.get("judge_order", ())) != JUDGES:
        raise RuntimeError("Falcon raw-output judge order changed")
    if tuple(fusion_config.get("judge_order", ())) != JUDGES:
        raise RuntimeError("MMDS calibrator judge order is incompatible")

    ordered_ids = [str(record["sample_id"]) for record in manifest]
    partition_by_id = {
        str(record["sample_id"]): str(record["prospective_partition"])
        for record in manifest
    }
    group_by_id = {
        str(record["sample_id"]): str(record["group_id"])
        for record in manifest
    }
    adaptation_ids = [
        sample_id
        for sample_id in ordered_ids
        if partition_by_id[sample_id] == "adaptation"
    ]
    confirmation_ids = [
        sample_id
        for sample_id in ordered_ids
        if partition_by_id[sample_id] == "confirmation"
    ]
    labels_by_id = {
        str(record["sample_id"]): record for record in adaptation_labels
    }
    if set(labels_by_id) != set(adaptation_ids) or len(labels_by_id) != len(adaptation_ids):
        raise RuntimeError("Adaptation labels do not exactly cover the frozen partition")
    if any(
        record.get("prospective_partition") != "adaptation"
        for record in adaptation_labels
    ):
        raise RuntimeError("Non-adaptation label found in the adaptation-label file")

    raw_probabilities: dict[str, np.ndarray] = {}
    judgment_hashes: dict[str, str] = {}
    for judge in JUDGES:
        path = result_root / judge / "judgments.jsonl"
        records = _load_jsonl(path)
        if [str(record["sample_id"]) for record in records] != ordered_ids:
            raise RuntimeError(f"Falcon judgment order mismatch for {judge}")
        expected_hash = str(
            raw_freeze["judges"][judge]["files"]["judgments"]["sha256"]
        )
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Falcon judgments changed after freeze for {judge}")
        raw_probabilities[judge] = np.asarray(
            [float(record["soft_evidence"]["p_unsafe"]) for record in records],
            dtype=np.float64,
        )
        judgment_hashes[judge] = expected_hash

    calibrated = apply_calibrators(
        raw_probabilities,
        fusion_config["final_development_calibrators"],
        JUDGES,
    )
    calibrated_matrix = np.column_stack([calibrated[judge] for judge in JUDGES])
    soft_features = logit(
        np.clip(calibrated_matrix, EPSILON, 1.0 - EPSILON)
    )
    hard_features = np.column_stack(
        [(raw_probabilities[judge] >= 0.5).astype(float) for judge in JUDGES]
    )
    index_by_id = {sample_id: index for index, sample_id in enumerate(ordered_ids)}
    adaptation_indices = np.asarray(
        [index_by_id[sample_id] for sample_id in adaptation_ids], dtype=np.int64
    )
    confirmation_indices = np.asarray(
        [index_by_id[sample_id] for sample_id in confirmation_ids], dtype=np.int64
    )
    adaptation_targets = np.asarray(
        [
            1 if labels_by_id[sample_id]["assistant_label"] == "unsafe" else 0
            for sample_id in adaptation_ids
        ],
        dtype=np.int64,
    )
    if set(np.unique(adaptation_targets)) != {0, 1}:
        raise RuntimeError("Falcon adaptation partition does not contain both classes")

    soft_model = _fit(soft_features[adaptation_indices], adaptation_targets)
    hard_model = _fit(hard_features[adaptation_indices], adaptation_targets)
    soft_probability = soft_model.predict_proba(soft_features)[:, 1]
    hard_probability = hard_model.predict_proba(hard_features)[:, 1]
    soft_prediction = (soft_probability >= 0.5).astype(np.int64)
    hard_prediction = (hard_probability >= 0.5).astype(np.int64)

    logical = baseline_predictions(raw_probabilities, JUDGES)
    soft_mean_probability = calibrated_matrix.mean(axis=1)
    soft_mean_prediction = (soft_mean_probability >= 0.5).astype(np.int64)
    frozen_model = fusion_model_from_dict(fusion_config["model"])
    frozen_probability = predict_calibrated_fusion(
        raw_probabilities,
        fusion_config["final_development_calibrators"],
        frozen_model,
    )
    frozen_prediction = (
        frozen_probability >= float(fusion_config["threshold"])
    ).astype(np.int64)

    predictions: list[dict[str, Any]] = []
    for index in confirmation_indices:
        sample_id = ordered_ids[int(index)]
        predictions.append(
            {
                "schema": "falcon-prospective-frozen-prediction-v1",
                "sample_id": sample_id,
                "group_id": group_by_id[sample_id],
                "prospective_partition": "confirmation",
                "raw_p_unsafe": {
                    judge: float(raw_probabilities[judge][index]) for judge in JUDGES
                },
                "calibrated_p_unsafe": {
                    judge: float(calibrated[judge][index]) for judge in JUDGES
                },
                "probabilities": {
                    "adapted_soft_logits": float(soft_probability[index]),
                    "adapted_hard_logits": float(hard_probability[index]),
                    "mmds_frozen_fusion": float(frozen_probability[index]),
                    "calibrated_soft_mean": float(soft_mean_probability[index]),
                    **{
                        f"single_{judge}": float(raw_probabilities[judge][index])
                        for judge in JUDGES
                    },
                },
                "predictions": {
                    "adapted_soft_logits": "unsafe" if soft_prediction[index] else "safe",
                    "adapted_hard_logits": "unsafe" if hard_prediction[index] else "safe",
                    "same_pool_majority": "unsafe" if logical["majority"][index] else "safe",
                    "or_1_of_3": "unsafe" if logical["or_1_of_n"][index] else "safe",
                    "and_3_of_3": "unsafe" if logical["and_n_of_n"][index] else "safe",
                    "mmds_frozen_fusion": "unsafe" if frozen_prediction[index] else "safe",
                    "calibrated_soft_mean": "unsafe" if soft_mean_prediction[index] else "safe",
                    **{
                        f"single_{judge}": (
                            "unsafe"
                            if logical[f"single_{judge}"][index]
                            else "safe"
                        )
                        for judge in JUDGES
                    },
                },
            }
        )

    predictions_path = output_root / "confirmation_predictions.frozen.jsonl"
    models_path = output_root / "adapted_models.json"
    freeze_path = output_root / "adapted_model_freeze.json"
    _write_jsonl(predictions_path, predictions)
    model_record = {
        "schema": "falcon-prospective-adapted-models-v1",
        "status": "fit-on-adaptation-only",
        "feature_calibrators": "MMDS final-development calibrators",
        "fusion_config_sha256": fusion_hash,
        "adaptation_sample_count": len(adaptation_ids),
        "adaptation_group_count": len(
            {group_by_id[sample_id] for sample_id in adaptation_ids}
        ),
        "primary_soft_model": _model_record(soft_model, JUDGES),
        "mechanism_control_hard_model": _model_record(hard_model, JUDGES),
        "confirmation_labels_accessed": False,
    }
    model_record["content_sha256"] = sha256_text(canonical_json(model_record))
    _write_json(models_path, model_record)

    frozen = {
        "schema": "falcon-prospective-adapted-model-freeze-v1",
        "status": "frozen-before-confirmation-label-access",
        "protocol_file_sha256": sha256_file(protocol_path),
        "raw_output_freeze_sha256": raw_freeze_hash,
        "fusion_config_sha256": fusion_hash,
        "adaptation_labels_sha256": sha256_file(adaptation_labels_path),
        "adaptation_sample_count": len(adaptation_ids),
        "adaptation_class_counts": {
            "safe": int((adaptation_targets == 0).sum()),
            "unsafe": int((adaptation_targets == 1).sum()),
        },
        "confirmation_sample_count": len(confirmation_ids),
        "confirmation_label_seal_sha256": confirmation_seal_hash,
        "sealed_confirmation_labels_sha256": confirmation_seal[
            "confirmation_label_file_sha256"
        ],
        "models_file": str(models_path),
        "models_file_sha256": sha256_file(models_path),
        "predictions_file": str(predictions_path),
        "predictions_file_sha256": sha256_file(predictions_path),
        "judgment_sha256": judgment_hashes,
        "confirmation_labels_opened": False,
        "model_or_threshold_selection_after_adaptation_fit": False,
    }
    frozen["content_sha256"] = sha256_text(canonical_json(frozen))
    if freeze_path.is_file():
        prior = _load_json(freeze_path)
        if prior != frozen:
            raise RuntimeError("Existing Falcon model freeze differs from current fit")
    else:
        _write_json(freeze_path, frozen)
    return frozen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    experiment_root = Path(__file__).resolve().parent
    base = (
        V15_ROOT
        / "results"
        / "external"
        / "falcon_test"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=experiment_root / "configs" / "falcon_external_unlabeled.jsonl",
    )
    parser.add_argument(
        "--adaptation-labels",
        type=Path,
        default=base / "preparation" / "falcon_adaptation_labels.jsonl",
    )
    parser.add_argument(
        "--confirmation-seal",
        type=Path,
        default=base / "preparation" / "confirmation_label_seal.json",
    )
    parser.add_argument(
        "--raw-freeze", type=Path, default=base / "raw_output_freeze.json"
    )
    parser.add_argument("--result-root", type=Path, default=base / "raw")
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=V15_ROOT / "results" / "mmds" / "fit" / "fusion_config.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=experiment_root / "configs" / "falcon_prospective_protocol_v1.json",
    )
    parser.add_argument("--output-root", type=Path, default=base)
    return parser


if __name__ == "__main__":
    result = fit_and_freeze(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "adaptation_sample_count": result["adaptation_sample_count"],
                "confirmation_sample_count": result["confirmation_sample_count"],
                "confirmation_labels_opened": result["confirmation_labels_opened"],
            },
            indent=2,
        )
    )
