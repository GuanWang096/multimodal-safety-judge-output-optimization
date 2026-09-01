from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.evidence_v2 import (  # noqa: E402
    JUDGE_ORDER,
    PACKAGE_SCHEMA,
    build_canonical_evidence,
    canonical_json_bytes,
    forbidden_runtime_keys,
    stable_json_sha256,
    verify_embedded_hash,
    verify_package,
)

from mj5_common import (  # noqa: E402
    generate_validator_order,
    judge_private_key,
    public_key_fingerprint,
    public_key_hex,
    validator_private_key,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_json_bytes(value, trailing_newline=True))


def source_completion_timestamp(result_root: Path) -> int:
    ended = []
    for judge in JUDGE_ORDER:
        status = read_json(result_root / judge / "status.json")
        if status.get("state") != "completed" or status.get("completed") != 513:
            raise RuntimeError(f"Incomplete confirmatory source for {judge}")
        ended.append(float(status["ended_at_unix"]))
    return math.ceil(max(ended))


def validate_source_alignment(
    ordered_ids: Sequence[str],
    by_judge: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    expected = set(ordered_ids)
    for judge in JUDGE_ORDER:
        if set(by_judge[judge]) != expected:
            raise RuntimeError(f"Confirmatory source IDs do not align for {judge}")
    for sample_id in ordered_ids:
        records = [by_judge[judge][sample_id] for judge in JUDGE_ORDER]
        for field in (
            "sample_id",
            "dataset",
            "split",
            "conversation_sha256",
            "target_response_sha256",
            "image_hashes",
            "policy_id",
            "policy_version",
            "policy_sha256",
        ):
            if any(record[field] != records[0][field] for record in records[1:]):
                raise RuntimeError(f"Source field mismatch for {sample_id}:{field}")


def build_registrations(
    first_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    judge_keys: dict[str, Any] = {}
    judge_registrations: dict[str, Any] = {}
    for judge in JUDGE_ORDER:
        identity = first_records[judge]["judge_service"]
        private_key = judge_private_key(str(identity["canonical_id"]))
        judge_keys[judge] = {
            "private_key_hex": private_key.private_bytes_raw().hex(),
            "public_key_hex": public_key_hex(private_key),
            "public_key_fingerprint": public_key_fingerprint(private_key),
            "purpose": "deterministic Evidence V2 replay-only signing key",
        }
        judge_registrations[judge] = {
            "judgeId": judge,
            "organization": str(identity["organization"]),
            "modelId": str(identity["model_id"]),
            "modelRevision": str(identity["model_revision"]),
            "policySha256": str(identity["policy_sha256"]),
            "adapterVersion": str(identity["adapter_version"]),
            "publicKeyHex": public_key_hex(private_key),
        }

    validator_order, validator_ppm = generate_validator_order(16)
    validator_keys = {}
    validator_registrations = []
    for node_id in range(16):
        private_key = validator_private_key(node_id)
        validator_id = f"validator-{node_id:02d}"
        validator_keys[validator_id] = {
            "node_id": node_id,
            "private_key_hex": private_key.private_bytes_raw().hex(),
            "public_key_hex": public_key_hex(private_key),
            "public_key_fingerprint": public_key_fingerprint(private_key),
            "reliability_ppm": validator_ppm[node_id],
            "version": 1,
            "purpose": "deterministic RGG-PBFT test validator key",
        }
        validator_registrations.append(
            {
                "validatorId": validator_id,
                "publicKeyHex": public_key_hex(private_key),
                "reliabilityPpm": validator_ppm[node_id],
                "version": 1,
            }
        )
    registrations = {
        "schema": "evidence-v2-registration-v1",
        "judge_registrations": judge_registrations,
        "validator_order": validator_order,
        "validator_registrations": validator_registrations,
    }
    keys = {
        "schema": "evidence-v2-replay-key-material-v1",
        "security_scope": "public deterministic experiment keys; never production keys",
        "judge_keys": judge_keys,
        "validator_keys": validator_keys,
    }
    return registrations, keys


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Evidence V2 output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    softfusion = args.softfusion.resolve()
    result_root = softfusion / "confirm"
    evaluation_root = softfusion / "evaluation"
    fusion_config_path = args.fusion_config.resolve()
    fusion_config = read_json(fusion_config_path)
    config_hash = verify_embedded_hash(fusion_config, "config_sha256")
    raw_freeze = read_json(result_root / "raw_output_freeze.json")
    raw_freeze_hash = verify_embedded_hash(raw_freeze, "content_sha256")
    result = read_json(evaluation_root / "confirmatory_results.json")
    result_content_hash = verify_embedded_hash(result, "result_content_sha256")

    predictions = read_jsonl(evaluation_root / "confirmatory_predictions.jsonl")
    ordered_ids = [str(value["sample_id"]) for value in predictions]
    if len(ordered_ids) != 513 or len(set(ordered_ids)) != 513:
        raise RuntimeError("Expected 513 unique confirmatory predictions")
    prediction_by_id = {str(value["sample_id"]): value for value in predictions}

    by_judge: dict[str, dict[str, dict[str, Any]]] = {}
    source_files = {}
    for judge in JUDGE_ORDER:
        path = result_root / judge / "judgments.jsonl"
        rows = read_jsonl(path)
        by_judge[judge] = {str(value["sample_id"]): value for value in rows}
        source_files[judge] = {
            "path": str(path.relative_to(V15_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "record_count": len(rows),
        }
        expected_hash = raw_freeze["judges"][judge]["files"]["judgments"][
            "sha256"
        ]
        if source_files[judge]["sha256"] != expected_hash:
            raise RuntimeError(f"Frozen source hash mismatch for {judge}")
    validate_source_alignment(ordered_ids, by_judge)

    labels = read_jsonl(softfusion / "data" / "confirm-labels.jsonl")
    labels_by_id = {
        str(value["sample_id"]): str(value["assistant_label"]) for value in labels
    }
    if set(labels_by_id) != set(ordered_ids):
        raise RuntimeError("Settlement labels do not exactly cover confirmatory records")

    timestamp_unix = source_completion_timestamp(result_root)
    package_dir = output / "bundles" / "records"
    record_entries = []
    packages = []
    first_records = {
        judge: by_judge[judge][ordered_ids[0]] for judge in JUDGE_ORDER
    }
    registrations, keys = build_registrations(first_records)
    write_json(output / "registrations.json", registrations)
    write_json(output / "replay_keys.json", keys)

    for index, sample_id in enumerate(ordered_ids):
        source_records = {
            judge: by_judge[judge][sample_id] for judge in JUDGE_ORDER
        }
        evidence = build_canonical_evidence(
            sample_id=sample_id,
            source_records=source_records,
            stored_prediction=prediction_by_id[sample_id],
            fusion_config=fusion_config,
            timestamp_unix=timestamp_unix,
            raw_output_freeze_sha256=raw_freeze_hash,
            result_content_sha256=result_content_hash,
        )
        evidence_hash = stable_json_sha256(evidence)
        package = {
            "schema": PACKAGE_SCHEMA,
            "canonical_evidence_sha256": evidence_hash,
            "canonical_evidence": evidence,
            "attachments": {"judge_records": source_records},
            "provenance": {
                "fusion_config_sha256": config_hash,
                "raw_output_freeze_sha256": raw_freeze_hash,
                "confirmatory_result_sha256": result_content_hash,
                "source_judgments": source_files,
            },
        }
        if forbidden := forbidden_runtime_keys(package):
            raise RuntimeError(f"Runtime label leakage in {sample_id}: {forbidden}")
        verify_package(package, fusion_config)
        package_bytes = canonical_json_bytes(package, trailing_newline=True)
        package_path = package_dir / f"{index:04d}.json"
        write_bytes(package_path, package_bytes)
        packages.append(package)
        record_entries.append(
            {
                "index": index,
                "sample_id": sample_id,
                "bundle_path": str(package_path.relative_to(output)).replace("\\", "/"),
                "bundle_bytes": len(package_bytes),
                "bundle_sha256": hashlib.sha256(package_bytes).hexdigest(),
                "evidence_sha256": evidence_hash,
                "predictions": {
                    item["judge_key"]: item["raw_score"]["predicted_label"]
                    for item in evidence["judges"]
                },
                "posterior_unsafe_ppm": round(
                    float(evidence["fusion"]["posterior_unsafe"]) * 1_000_000
                ),
                "provisional_label": str(evidence["fusion"]["decision"]),
            }
        )

    batch_entries = []
    for batch_index, offset in enumerate(range(0, len(packages), args.batch_size)):
        selected_packages = packages[offset : offset + args.batch_size]
        selected_records = record_entries[offset : offset + args.batch_size]
        envelope = {
            "schema": "multijudge-evidence-batch-v2",
            "batch_index": batch_index,
            "fusion_config_sha256": config_hash,
            "record_count": len(selected_packages),
            "packages": selected_packages,
        }
        if forbidden := forbidden_runtime_keys(envelope):
            raise RuntimeError(f"Runtime label leakage in batch {batch_index}: {forbidden}")
        payload = canonical_json_bytes(envelope, trailing_newline=True)
        batch_path = output / "bundles" / "batches" / f"batch-{batch_index:02d}.json"
        write_bytes(batch_path, payload)
        batch_entries.append(
            {
                "batch_index": batch_index,
                "bundle_path": str(batch_path.relative_to(output)).replace("\\", "/"),
                "bundle_bytes": len(payload),
                "bundle_sha256": hashlib.sha256(payload).hexdigest(),
                "record_indices": [value["index"] for value in selected_records],
                "sample_ids": [value["sample_id"] for value in selected_records],
            }
        )

    settlement = {
        "schema": "evidence-v2-independent-settlement-labels-v1",
        "separation_notice": (
            "This file is excluded from Evidence V2 packages and is read only after "
            "certificate admission."
        ),
        "labels": [
            {"sample_id": sample_id, "label": labels_by_id[sample_id]}
            for sample_id in ordered_ids
        ],
    }
    write_json(output / "settlement_labels.json", settlement)

    manifest = {
        "schema": "multijudge-evidence-v2-workload-manifest",
        "status": "built",
        "sample_count": len(record_entries),
        "batch_size": args.batch_size,
        "batch_count": len(batch_entries),
        "judge_order": list(JUDGE_ORDER),
        "timestamp_rule": "ceil(max(confirmatory judge formal completion times))",
        "timestamp_unix": timestamp_unix,
        "fusion_config": {
            "path": str(fusion_config_path.relative_to(V15_ROOT)).replace("\\", "/"),
            "sha256": config_hash,
        },
        "source_files": source_files,
        "records": record_entries,
        "batches": batch_entries,
    }
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    write_json(output / "manifest.json", manifest)
    summary = {
        "schema": "multijudge-evidence-v2-build-summary",
        "status": "completed",
        "sample_count": len(record_entries),
        "batch_count": len(batch_entries),
        "runtime_labels_embedded": False,
        "fusion_config_sha256": config_hash,
        "manifest_sha256": manifest["manifest_sha256"],
        "record_evidence_hash_unique_count": len(
            {value["evidence_sha256"] for value in record_entries}
        ),
        "total_record_package_bytes": sum(
            value["bundle_bytes"] for value in record_entries
        ),
        "total_batch_package_bytes": sum(
            value["bundle_bytes"] for value in batch_entries
        ),
    }
    write_json(output / "build_summary.json", summary)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--softfusion",
        type=Path,
        default=V15_ROOT / "results/mmds",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=V15_ROOT / "results/mmds/fit/fusion_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V15_ROOT / "results/replay/workload",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
