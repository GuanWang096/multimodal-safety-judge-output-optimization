from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


EPSILON = 1e-6
JUDGE_ORDER = ("qwen", "safework", "minicpm")
EVIDENCE_SCHEMA = "multijudge-canonical-evidence-v2"
PACKAGE_SCHEMA = "multijudge-evidence-package-v2"


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return encoded + (b"\n" if trailing_newline else b"")


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_embedded_hash(value: Mapping[str, Any], field: str) -> str:
    expected = str(value[field])
    unhashed = dict(value)
    unhashed.pop(field, None)
    observed = stable_json_sha256(unhashed)
    if observed != expected:
        raise ValueError(f"Embedded hash mismatch for {field}: {observed} != {expected}")
    return expected


def _clip_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Probability must be finite")
    return min(max(float(value), EPSILON), 1.0 - EPSILON)


def _logit(value: float) -> float:
    clipped = _clip_probability(value)
    return math.log(clipped) - math.log1p(-clipped)


def _expit(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def apply_calibrator(parameters: Mapping[str, Any], probability: float) -> float:
    method = str(parameters["method"])
    clipped = _clip_probability(probability)
    if method == "identity":
        return clipped
    if method == "temperature":
        temperature = float(parameters["temperature"])
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        return _expit(_logit(clipped) / temperature)
    if method == "beta":
        return _expit(
            float(parameters["a"]) * math.log(clipped)
            - float(parameters["b"]) * math.log1p(-clipped)
            + float(parameters["c"])
        )
    raise ValueError(f"Unsupported calibrator method: {method}")


def fusion_config_sha256(config: Mapping[str, Any]) -> str:
    unhashed = dict(config)
    claimed = unhashed.pop("config_sha256", None)
    observed = stable_json_sha256(unhashed)
    if claimed is not None and str(claimed) != observed:
        raise ValueError("Frozen fusion config hash does not match its contents")
    return observed


def recompute_fusion(
    raw_probabilities: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    order = tuple(str(value) for value in config["judge_order"])
    if order != JUDGE_ORDER:
        raise ValueError(f"Unexpected frozen judge order: {order}")
    model = config["model"]
    if tuple(str(value) for value in model["judge_order"]) != order:
        raise ValueError("Fusion model and config judge orders differ")
    if str(model["feature_family"]) != "F1":
        raise ValueError("Evidence V2 currently requires the frozen F1 feature family")

    calibrated: dict[str, float] = {}
    feature_logits: dict[str, float] = {}
    standardized: dict[str, float] = {}
    for index, judge in enumerate(order):
        calibrated[judge] = apply_calibrator(
            config["final_development_calibrators"][judge],
            float(raw_probabilities[judge]),
        )
        feature_logits[judge] = _logit(calibrated[judge])
        standardized[judge] = (
            feature_logits[judge] - float(model["mean"][index])
        ) / float(model["scale"][index])

    linear_predictor = float(model["intercept"])
    for index, judge in enumerate(order):
        linear_predictor += float(model["coefficients"][index]) * standardized[judge]
    posterior = _expit(linear_predictor)
    threshold = float(config["threshold"])
    decision = "unsafe" if posterior >= threshold else "safe"
    return {
        "calibrated_p_unsafe": calibrated,
        "feature_logits": feature_logits,
        "standardized_features": standardized,
        "linear_predictor": linear_predictor,
        "posterior_unsafe": posterior,
        "threshold": threshold,
        "decision": decision,
    }


def build_canonical_evidence(
    *,
    sample_id: str,
    source_records: Mapping[str, Mapping[str, Any]],
    stored_prediction: Mapping[str, Any],
    fusion_config: Mapping[str, Any],
    timestamp_unix: int,
    raw_output_freeze_sha256: str,
    result_content_sha256: str,
) -> dict[str, Any]:
    first = source_records[JUDGE_ORDER[0]]
    raw_probabilities = {
        judge: float(source_records[judge]["soft_evidence"]["p_unsafe"])
        for judge in JUDGE_ORDER
    }
    recomputed = recompute_fusion(raw_probabilities, fusion_config)
    judges = []
    for judge in JUDGE_ORDER:
        record = source_records[judge]
        identity = {
            "canonical_id": str(record["judge_service"]["canonical_id"]),
            "organization": str(record["judge_service"]["organization"]),
            "model_id": str(record["judge_service"]["model_id"]),
            "model_revision": str(record["judge_service"]["model_revision"]),
            "adapter_version": str(record["judge_service"]["adapter_version"]),
        }
        score = record["soft_evidence"]
        judges.append(
            {
                "judge_key": judge,
                "identity": identity,
                "identity_sha256": stable_json_sha256(identity),
                "raw_score": {
                    "method": str(score["score_method"]),
                    "p_unsafe": raw_probabilities[judge],
                    "margin_unsafe_minus_safe": float(
                        score["margin_unsafe_minus_safe"]
                    ),
                    "predicted_label": str(score["predicted_label"]),
                },
                "calibrated_score": {
                    "p_unsafe": recomputed["calibrated_p_unsafe"][judge],
                    "method": str(fusion_config["calibrator_methods"][judge]),
                },
                "fusion_feature": {
                    "logit": recomputed["feature_logits"][judge],
                    "standardized": recomputed["standardized_features"][judge],
                },
                "source_record_sha256": stable_json_sha256(record),
            }
        )

    multimedia_record = {
        "conversation_sha256": str(first["conversation_sha256"]),
        "target_response_sha256": str(first["target_response_sha256"]),
        "image_hashes": first["image_hashes"],
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "sample_id": sample_id,
        "dataset": str(first["dataset"]),
        "split": str(first["split"]),
        "multimedia_record": multimedia_record,
        "multimedia_record_sha256": stable_json_sha256(multimedia_record),
        "policy": {
            "policy_id": str(first["policy_id"]),
            "policy_version": str(first["policy_version"]),
            "policy_sha256": str(first["policy_sha256"]),
        },
        "judges": judges,
        "frozen_fusion": {
            "config_sha256": fusion_config_sha256(fusion_config),
            "feature_family": str(fusion_config["model"]["feature_family"]),
            "penalty": str(fusion_config["model"]["penalty"]),
            "c_value": float(fusion_config["model"]["C"]),
            "threshold": recomputed["threshold"],
        },
        "fusion": {
            "linear_predictor": recomputed["linear_predictor"],
            "posterior_unsafe": recomputed["posterior_unsafe"],
            "decision": recomputed["decision"],
        },
        "versioned_decision_state": {
            "score_schema": "soft-evidence-v1",
            "evidence_schema": EVIDENCE_SCHEMA,
            "raw_output_freeze_sha256": raw_output_freeze_sha256,
            "confirmatory_result_sha256": result_content_sha256,
        },
        "timestamp_unix": int(timestamp_unix),
    }
    assert_matches_stored_prediction(evidence, stored_prediction)
    return evidence


def assert_matches_stored_prediction(
    evidence: Mapping[str, Any],
    stored_prediction: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> None:
    by_judge = {str(item["judge_key"]): item for item in evidence["judges"]}
    for judge in JUDGE_ORDER:
        raw = float(by_judge[judge]["raw_score"]["p_unsafe"])
        calibrated = float(by_judge[judge]["calibrated_score"]["p_unsafe"])
        if not math.isclose(
            raw, float(stored_prediction["raw_p_unsafe"][judge]), abs_tol=tolerance
        ):
            raise ValueError(f"Raw score mismatch for {evidence['sample_id']}:{judge}")
        if not math.isclose(
            calibrated,
            float(stored_prediction["calibrated_p_unsafe"][judge]),
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"Calibrated score mismatch for {evidence['sample_id']}:{judge}"
            )
    if not math.isclose(
        float(evidence["fusion"]["posterior_unsafe"]),
        float(stored_prediction["fusion_p_unsafe"]),
        abs_tol=tolerance,
    ):
        raise ValueError(f"Fusion posterior mismatch for {evidence['sample_id']}")
    if str(evidence["fusion"]["decision"]) != str(
        stored_prediction["fusion_prediction"]
    ):
        raise ValueError(f"Fusion decision mismatch for {evidence['sample_id']}")


def verify_evidence(
    evidence: Mapping[str, Any],
    fusion_config: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("Unexpected canonical evidence schema")
    if str(evidence["frozen_fusion"]["config_sha256"]) != fusion_config_sha256(
        fusion_config
    ):
        raise ValueError("Evidence references the wrong frozen fusion config")
    judges = list(evidence["judges"])
    if tuple(str(item["judge_key"]) for item in judges) != JUDGE_ORDER:
        raise ValueError("Evidence judge order differs from the frozen order")
    raw = {
        str(item["judge_key"]): float(item["raw_score"]["p_unsafe"])
        for item in judges
    }
    recomputed = recompute_fusion(raw, fusion_config)
    for item in judges:
        judge = str(item["judge_key"])
        if stable_json_sha256(item["identity"]) != str(item["identity_sha256"]):
            raise ValueError(f"Judge identity hash mismatch: {judge}")
        if not math.isclose(
            float(item["calibrated_score"]["p_unsafe"]),
            recomputed["calibrated_p_unsafe"][judge],
            abs_tol=tolerance,
        ):
            raise ValueError(f"Calibrated score mismatch: {judge}")
        if not math.isclose(
            float(item["fusion_feature"]["standardized"]),
            recomputed["standardized_features"][judge],
            abs_tol=tolerance,
        ):
            raise ValueError(f"Fusion feature mismatch: {judge}")
    if not math.isclose(
        float(evidence["fusion"]["posterior_unsafe"]),
        recomputed["posterior_unsafe"],
        abs_tol=tolerance,
    ):
        raise ValueError("Fusion posterior does not recompute")
    if str(evidence["fusion"]["decision"]) != recomputed["decision"]:
        raise ValueError("Fusion decision does not recompute")
    if stable_json_sha256(evidence["multimedia_record"]) != str(
        evidence["multimedia_record_sha256"]
    ):
        raise ValueError("Multimedia record hash mismatch")
    return recomputed


def verify_package(
    package: Mapping[str, Any],
    fusion_config: Mapping[str, Any],
) -> dict[str, Any]:
    if package.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("Unexpected Evidence V2 package schema")
    evidence = package["canonical_evidence"]
    observed_hash = stable_json_sha256(evidence)
    if observed_hash != str(package["canonical_evidence_sha256"]):
        raise ValueError("Canonical Evidence V2 hash mismatch")
    recomputed = verify_evidence(evidence, fusion_config)
    attachments = package["attachments"]["judge_records"]
    by_judge = {str(item["judge_key"]): item for item in evidence["judges"]}
    for judge in JUDGE_ORDER:
        if stable_json_sha256(attachments[judge]) != str(
            by_judge[judge]["source_record_sha256"]
        ):
            raise ValueError(f"Attached source record hash mismatch: {judge}")
    return recomputed


def forbidden_runtime_keys(value: Any) -> list[str]:
    forbidden = {"expected_label", "assistant_label", "settlement_label"}
    found: list[str] = []

    def walk(current: Any, path: str) -> None:
        if isinstance(current, Mapping):
            for key, item in current.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key).lower() in forbidden:
                    found.append(child)
                walk(item, child)
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            for index, item in enumerate(current):
                walk(item, f"{path}[{index}]")

    walk(value, "")
    return found
