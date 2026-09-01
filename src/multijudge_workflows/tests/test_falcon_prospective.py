from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FUSION_CONFIG = REPOSITORY_ROOT / "results" / "mmds" / "fit" / "fusion_config.json"


def _load_script(name: str):
    path = SCRIPT_ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIT = _load_script("fit_falcon_prospective.py")
AUDIT = _load_script("audit_falcon_model_freeze.py")
EVALUATE = _load_script("evaluate_falcon_prospective.py")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(value) for value in values) + "\n",
        encoding="utf-8",
    )


def _add_hash(value: dict, field: str) -> dict:
    value[field] = FIT.sha256_text(FIT.canonical_json(value))
    return value


def test_falcon_fit_freezes_predictions_before_confirmation_open(tmp_path: Path) -> None:
    protocol = {
        "schema": "test-falcon-protocol",
        "claims_and_endpoints": {
            "bootstrap_replicates": 100,
            "randomization_replicates": 100,
            "confidence_level": 0.95,
            "statistical_seed": 17,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)

    manifest = []
    adaptation_labels = []
    confirmation_labels = []
    targets = [0, 1] * 6
    for index, target in enumerate(targets):
        partition = "adaptation" if index < 6 else "confirmation"
        sample_id = f"falcon:test:{index}"
        manifest.append(
            {
                "sample_id": sample_id,
                "group_id": f"group-{index}",
                "prospective_partition": partition,
            }
        )
        label = {
            "sample_id": sample_id,
            "group_id": f"group-{index}",
            "prospective_partition": partition,
            "assistant_label": "unsafe" if target else "safe",
            "harmful_category": "test" if target else "",
        }
        if partition == "adaptation":
            adaptation_labels.append(label)
        else:
            confirmation_labels.append(label)

    manifest_path = tmp_path / "manifest.jsonl"
    adaptation_path = tmp_path / "adaptation.jsonl"
    confirmation_path = tmp_path / "confirmation.jsonl"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(adaptation_path, adaptation_labels)
    _write_jsonl(confirmation_path, confirmation_labels)

    seal = _add_hash(
        {
            "schema": "falcon-confirmation-label-seal-v1",
            "status": "sealed-not-opened",
            "confirmation_sample_count": 6,
            "confirmation_label_file": str(confirmation_path),
            "confirmation_label_file_sha256": FIT.sha256_file(confirmation_path),
            "protocol_freeze_record_sha256": "test",
            "opened": False,
            "model_freeze_required_before_open": True,
        },
        "seal_sha256",
    )
    seal_path = tmp_path / "seal.json"
    _write_json(seal_path, seal)

    result_root = tmp_path / "raw"
    raw_freeze = {
        "schema": "falcon-test-external-raw-output-freeze-v1",
        "status": "frozen-before-adaptation-label-access",
        "manifest_sha256": FIT.sha256_file(manifest_path),
        "protocol_file_sha256": FIT.sha256_file(protocol_path),
        "judge_order": list(FIT.JUDGES),
        "judges": {},
        "labels_accessed": False,
    }
    offsets = {"qwen": -0.03, "safework": 0.04, "minicpm": 0.0}
    for judge, offset in offsets.items():
        judgments = []
        for index, target in enumerate(targets):
            probability = (0.82 if target else 0.18) + offset
            judgments.append(
                {
                    "sample_id": f"falcon:test:{index}",
                    "soft_evidence": {"p_unsafe": probability},
                }
            )
        path = result_root / judge / "judgments.jsonl"
        _write_jsonl(path, judgments)
        raw_freeze["judges"][judge] = {
            "files": {"judgments": {"sha256": FIT.sha256_file(path)}}
        }
    _add_hash(raw_freeze, "content_sha256")
    raw_freeze_path = tmp_path / "raw_freeze.json"
    _write_json(raw_freeze_path, raw_freeze)

    model_root = tmp_path / "model-freeze"
    frozen = FIT.fit_and_freeze(
        SimpleNamespace(
            manifest=manifest_path,
            adaptation_labels=adaptation_path,
            confirmation_seal=seal_path,
            raw_freeze=raw_freeze_path,
            result_root=result_root,
            fusion_config=FUSION_CONFIG,
            protocol=protocol_path,
            output_root=model_root,
        )
    )
    assert frozen["status"] == "frozen-before-confirmation-label-access"
    assert frozen["confirmation_labels_opened"] is False
    frozen_rows = FIT._load_jsonl(
        model_root / "confirmation_predictions.frozen.jsonl"
    )
    assert len(frozen_rows) == 6
    assert all("expected_label" not in row for row in frozen_rows)

    audit = AUDIT.audit(
        SimpleNamespace(
            manifest=manifest_path,
            protocol=protocol_path,
            confirmation_seal=seal_path,
            predictions=model_root / "confirmation_predictions.frozen.jsonl",
            models=model_root / "adapted_models.json",
            model_freeze=model_root / "adapted_model_freeze.json",
            output=model_root / "model_freeze_audit.json",
        )
    )
    assert audit["passed"] is True
    assert audit["confirmation_labels_accessed"] is False

    evaluation = EVALUATE.evaluate(
        SimpleNamespace(
            predictions=model_root / "confirmation_predictions.frozen.jsonl",
            model_freeze=model_root / "adapted_model_freeze.json",
            model_freeze_audit=model_root / "model_freeze_audit.json",
            confirmation_labels=confirmation_path,
            confirmation_seal=seal_path,
            protocol=protocol_path,
            output_root=tmp_path / "evaluation",
        )
    )
    assert evaluation["integrity_passed"] is True
    assert evaluation["confirmation_sample_count"] == 6
    assert evaluation["confirmation_opening_count"] == 1
    consumed_seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert consumed_seal["opened"] is True
    assert consumed_seal["opening_count"] == 1
    opening_path = tmp_path / "evaluation" / "confirmation_opening.json"
    assert opening_path.exists()
    assert evaluation["primary_comparisons"]["FALCON-C1"]["bootstrap"][
        "confidence_level"
    ] == 0.95
    assert evaluation["primary_comparisons"]["FALCON-C1"]["randomization"][
        "replicates"
    ] == 100

    with pytest.raises(RuntimeError, match="already have an opening record"):
        EVALUATE.evaluate(
            SimpleNamespace(
                predictions=model_root / "confirmation_predictions.frozen.jsonl",
                model_freeze=model_root / "adapted_model_freeze.json",
                model_freeze_audit=model_root / "model_freeze_audit.json",
                confirmation_labels=confirmation_path,
                confirmation_seal=seal_path,
                protocol=protocol_path,
                output_root=tmp_path / "evaluation",
            )
        )
