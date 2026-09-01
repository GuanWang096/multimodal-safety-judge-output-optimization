from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


V15_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = V15_ROOT / "src"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
for path in (SRC_ROOT, EXPERIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multijudge.evidence_v2 import (
    PACKAGE_SCHEMA,
    build_canonical_evidence,
    recompute_fusion,
    stable_json_sha256,
    verify_package,
)

from mj5_fabric_client import MJ5FabricClient


def fusion_config() -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "softfusion-fusion-config-v1",
        "status": "frozen",
        "judge_order": ["qwen", "safework", "minicpm"],
        "adapter_versions": {
            "qwen": "q-adapter",
            "safework": "s-adapter",
            "minicpm": "m-adapter",
        },
        "calibrator_methods": {
            "qwen": "identity",
            "safework": "temperature",
            "minicpm": "beta",
        },
        "final_development_calibrators": {
            "qwen": {"method": "identity"},
            "safework": {"method": "temperature", "temperature": 2.0},
            "minicpm": {"method": "beta", "a": 1.0, "b": 1.0, "c": 0.0},
        },
        "model": {
            "feature_family": "F1",
            "feature_names": ["qwen", "safework", "minicpm"],
            "judge_order": ["qwen", "safework", "minicpm"],
            "mean": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "coefficients": [0.5, 0.75, -0.25],
            "intercept": -0.1,
            "penalty": "l2",
            "C": 0.3,
            "l1_ratio": None,
        },
        "threshold": 0.4,
    }
    value["config_sha256"] = stable_json_sha256(value)
    return value


def source_record(judge: str, probability: float) -> dict[str, object]:
    return {
        "schema": "soft-evidence-v1",
        "sample_id": "sample-1",
        "dataset": "MMDS",
        "split": "confirm",
        "conversation_sha256": "a" * 64,
        "target_response_sha256": "b" * 64,
        "image_hashes": [{"reference": "images/1.jpg", "sha256": "c" * 64}],
        "judge_service": {
            "canonical_id": f"judge:{judge}",
            "organization": judge,
            "model_id": f"{judge}-model",
            "model_revision": "revision",
            "adapter_version": f"{judge[0]}-adapter",
        },
        "policy_id": "policy",
        "policy_version": "1.0",
        "policy_sha256": "d" * 64,
        "soft_evidence": {
            "score_method": "test-score",
            "p_unsafe": probability,
            "margin_unsafe_minus_safe": probability - 0.5,
            "predicted_label": "unsafe" if probability >= 0.5 else "safe",
        },
    }


def test_evidence_v2_recomputes_and_detects_tampering() -> None:
    config = fusion_config()
    sources = {
        "qwen": source_record("qwen", 0.2),
        "safework": source_record("safework", 0.8),
        "minicpm": source_record("minicpm", 0.3),
    }
    recomputed = recompute_fusion(
        {judge: record["soft_evidence"]["p_unsafe"] for judge, record in sources.items()},
        config,
    )
    stored = {
        "sample_id": "sample-1",
        "raw_p_unsafe": {
            judge: record["soft_evidence"]["p_unsafe"] for judge, record in sources.items()
        },
        "calibrated_p_unsafe": recomputed["calibrated_p_unsafe"],
        "fusion_p_unsafe": recomputed["posterior_unsafe"],
        "fusion_prediction": recomputed["decision"],
    }
    evidence = build_canonical_evidence(
        sample_id="sample-1",
        source_records=sources,
        stored_prediction=stored,
        fusion_config=config,
        timestamp_unix=1_700_000_000,
        raw_output_freeze_sha256="e" * 64,
        result_content_sha256="f" * 64,
    )
    package = {
        "schema": PACKAGE_SCHEMA,
        "canonical_evidence_sha256": stable_json_sha256(evidence),
        "canonical_evidence": evidence,
        "attachments": {"judge_records": sources},
    }
    verify_package(package, config)

    tampered = copy.deepcopy(package)
    tampered["canonical_evidence"]["judges"][0]["calibrated_score"][
        "p_unsafe"
    ] += 0.1
    tampered["canonical_evidence_sha256"] = stable_json_sha256(
        tampered["canonical_evidence"]
    )
    with pytest.raises(ValueError, match="Calibrated score mismatch"):
        verify_package(tampered, config)


def test_fabric_client_builds_batch_anchor_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(MJ5FabricClient)
    client.registrations = {"validator_order": [9, 1, 10, 6]}
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        client,
        "signed_outputs",
        lambda record, decision_id, judge_ids: [
            {"sampleId": record["sample_id"], "decisionId": decision_id}
        ],
    )
    valid_commit = (
        "txid [" + "a" * 64 + "] committed with status (VALID) at localhost:7051"
    )
    monkeypatch.setattr(
        client,
        "invoke",
        lambda function, payload: (
            captured.update(function=function, payload=payload) or 2.0,
            valid_commit,
        ),
    )
    monkeypatch.setattr(
        client, "query", lambda *_args: {"decisionId": "decision-batch"}
    )
    records = [
        {
            "sample_id": "sample-1",
            "evidence_sha256": "b" * 64,
            "provisional_label": "unsafe",
            "posterior_unsafe_ppm": 700_000,
        }
    ]
    snapshot, latency, commit = client.freeze_batch_with_details(
        records,
        "decision-batch",
        "bafy-test",
        "c" * 64,
        {},
        1_700_000_100,
    )
    import json

    payload = json.loads(captured["payload"])
    assert captured["function"] == "FreezeDecisionBatch"
    assert payload["records"][0]["evidenceSha256"] == "b" * 64
    assert payload["evidenceSha256"] == "c" * 64
    assert payload["leaderValidatorIds"] == [
        "validator-09",
        "validator-01",
        "validator-10",
        "validator-06",
    ]
    assert snapshot["decisionId"] == "decision-batch"
    assert latency == 2.0
    assert commit["validation_status"] == "VALID"
