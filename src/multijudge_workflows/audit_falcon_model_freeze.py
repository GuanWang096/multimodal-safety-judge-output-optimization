from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


JUDGES = ("qwen", "safework", "minicpm")
FORBIDDEN_KEYS = {
    "assistant_label",
    "expected_label",
    "ground_truth",
    "is_response_safe",
    "label",
    "target",
    "harmful_category",
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_embedded_hash(value: dict[str, Any], field: str) -> str:
    observed = str(value.get(field, ""))
    unhashed = dict(value)
    unhashed.pop(field, None)
    if observed != sha256_text(canonical_json(unhashed)):
        raise RuntimeError(f"Invalid embedded hash: {field}")
    return observed


def _forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if str(key).casefold() in FORBIDDEN_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    protocol_path = args.protocol.resolve()
    seal_path = args.confirmation_seal.resolve()
    predictions_path = args.predictions.resolve()
    models_path = args.models.resolve()
    freeze_path = args.model_freeze.resolve()
    output_path = args.output.resolve()

    manifest = _load_jsonl(manifest_path)
    predictions = _load_jsonl(predictions_path)
    models = _load_json(models_path)
    frozen = _load_json(freeze_path)
    seal = _load_json(seal_path)
    freeze_hash = _verify_embedded_hash(frozen, "content_sha256")
    model_hash = _verify_embedded_hash(models, "content_sha256")
    seal_hash = _verify_embedded_hash(seal, "seal_sha256")

    if frozen.get("status") != "frozen-before-confirmation-label-access":
        raise RuntimeError("Falcon adapted model freeze has the wrong state")
    if frozen.get("confirmation_labels_opened") is not False:
        raise RuntimeError("Falcon model freeze reports confirmation-label access")
    if seal.get("status") != "sealed-not-opened" or seal.get("opened") is not False:
        raise RuntimeError("Falcon confirmation seal is open")
    if frozen.get("confirmation_label_seal_sha256") != seal_hash:
        raise RuntimeError("Falcon model freeze references a different label seal")
    if frozen.get("protocol_file_sha256") != sha256_file(protocol_path):
        raise RuntimeError("Falcon protocol changed after model freeze")
    if frozen.get("models_file_sha256") != sha256_file(models_path):
        raise RuntimeError("Falcon adapted model file changed after freeze")
    if frozen.get("predictions_file_sha256") != sha256_file(predictions_path):
        raise RuntimeError("Falcon confirmation predictions changed after freeze")
    if frozen.get("sealed_confirmation_labels_sha256") != seal.get(
        "confirmation_label_file_sha256"
    ):
        raise RuntimeError("Falcon sealed confirmation-label hash changed")

    confirmation_entries = [
        record
        for record in manifest
        if record.get("prospective_partition") == "confirmation"
    ]
    expected_ids = [str(record["sample_id"]) for record in confirmation_entries]
    observed_ids = [str(record["sample_id"]) for record in predictions]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
        raise RuntimeError("Falcon frozen prediction coverage or order is invalid")
    expected_confirmation_count = int(frozen.get("confirmation_sample_count", -1))
    if len(predictions) != expected_confirmation_count:
        raise RuntimeError("Falcon confirmation prediction count is invalid")
    forbidden = [path for record in predictions for path in _forbidden_paths(record)]
    if forbidden:
        raise RuntimeError(f"Falcon frozen predictions contain labels: {forbidden[:5]}")

    for record in predictions:
        if record.get("prospective_partition") != "confirmation":
            raise RuntimeError("Non-confirmation prediction found in frozen output")
        for value in record["probabilities"].values():
            probability = float(value)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise RuntimeError("Invalid Falcon frozen probability")
        if set(record["predictions"].values()) - {"safe", "unsafe"}:
            raise RuntimeError("Invalid Falcon frozen prediction label")

    for key in ("primary_soft_model", "mechanism_control_hard_model"):
        model = models[key]
        expected = {
            "penalty": "l2",
            "C": 0.03,
            "solver": "lbfgs",
            "max_iter": 10000,
            "tol": 1e-9,
            "class_weight": None,
            "random_state": 20260802,
            "decision_threshold": 0.5,
            "feature_names": list(JUDGES),
        }
        for field, value in expected.items():
            if model.get(field) != value:
                raise RuntimeError(f"Unexpected {key} configuration: {field}")
        coefficients = [float(value) for value in model["coefficients"]]
        if len(coefficients) != len(JUDGES) or not all(
            math.isfinite(value) for value in coefficients
        ):
            raise RuntimeError(f"Invalid coefficients for {key}")
        if not math.isfinite(float(model["intercept"])):
            raise RuntimeError(f"Invalid intercept for {key}")

    result = {
        "schema": "falcon-prospective-model-freeze-audit-v1",
        "passed": True,
        "adapted_model_freeze_sha256": freeze_hash,
        "adapted_models_content_sha256": model_hash,
        "confirmation_label_seal_sha256": seal_hash,
        "confirmation_prediction_count": len(predictions),
        "confirmation_group_count": len(
            {str(record["group_id"]) for record in predictions}
        ),
        "prediction_order_matches_manifest": True,
        "predictions_contain_labels": False,
        "model_configuration_matches_protocol": True,
        "confirmation_labels_accessed": False,
    }
    result["content_sha256"] = sha256_text(canonical_json(result))
    _write_json(output_path, result)
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
        "--manifest",
        type=Path,
        default=experiment_root / "configs" / "falcon_external_unlabeled.jsonl",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=experiment_root / "configs" / "falcon_prospective_protocol_v1.json",
    )
    parser.add_argument(
        "--confirmation-seal",
        type=Path,
        default=base / "preparation" / "confirmation_label_seal.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=model_root / "confirmation_predictions.frozen.jsonl",
    )
    parser.add_argument(
        "--models", type=Path, default=model_root / "adapted_models.json"
    )
    parser.add_argument(
        "--model-freeze",
        type=Path,
        default=model_root / "adapted_model_freeze.json",
    )
    parser.add_argument(
        "--output", type=Path, default=model_root / "model_freeze_audit.json"
    )
    return parser


if __name__ == "__main__":
    result = audit(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
