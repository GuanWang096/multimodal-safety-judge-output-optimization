from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.evidence_v2 import (  # noqa: E402
    canonical_json_bytes,
    forbidden_runtime_keys,
    stable_json_sha256,
    verify_embedded_hash,
    verify_package,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value, trailing_newline=True))
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    workload = args.workload.resolve()
    fusion_config = read_json(args.fusion_config.resolve())
    config_hash = verify_embedded_hash(fusion_config, "config_sha256")
    manifest = read_json(workload / "manifest.json")
    manifest_hash = verify_embedded_hash(manifest, "manifest_sha256")
    checks = {
        "manifest_schema": manifest.get("schema")
        == "multijudge-evidence-v2-workload-manifest",
        "sample_count": manifest.get("sample_count") == 513,
        "batch_count": manifest.get("batch_count") == 9,
        "fusion_config_hash": manifest["fusion_config"]["sha256"] == config_hash,
        "record_ids_unique": len(
            {str(value["sample_id"]) for value in manifest["records"]}
        )
        == 513,
    }
    failures = []
    packages_by_index: dict[int, dict[str, Any]] = {}
    for record in manifest["records"]:
        index = int(record["index"])
        path = workload / str(record["bundle_path"])
        if sha256_file(path) != str(record["bundle_sha256"]):
            failures.append(f"record_bundle_hash:{index}")
            continue
        package = read_json(path)
        packages_by_index[index] = package
        if forbidden := forbidden_runtime_keys(package):
            failures.append(f"runtime_label_leakage:{index}:{','.join(forbidden)}")
        try:
            verify_package(package, fusion_config)
        except ValueError as error:
            failures.append(f"record_verify:{index}:{error}")
            continue
        if str(package["canonical_evidence_sha256"]) != str(
            record["evidence_sha256"]
        ):
            failures.append(f"record_evidence_hash:{index}")
        evidence = package["canonical_evidence"]
        if str(evidence["sample_id"]) != str(record["sample_id"]):
            failures.append(f"record_sample_binding:{index}")
        posterior_ppm = round(float(evidence["fusion"]["posterior_unsafe"]) * 1_000_000)
        if posterior_ppm != int(record["posterior_unsafe_ppm"]):
            failures.append(f"record_posterior_binding:{index}")
        if str(evidence["fusion"]["decision"]) != str(record["provisional_label"]):
            failures.append(f"record_decision_binding:{index}")

    seen_indices = []
    for batch in manifest["batches"]:
        path = workload / str(batch["bundle_path"])
        if sha256_file(path) != str(batch["bundle_sha256"]):
            failures.append(f"batch_bundle_hash:{batch['batch_index']}")
            continue
        envelope = read_json(path)
        if forbidden := forbidden_runtime_keys(envelope):
            failures.append(
                f"batch_label_leakage:{batch['batch_index']}:{','.join(forbidden)}"
            )
        if envelope.get("schema") != "multijudge-evidence-batch-v2":
            failures.append(f"batch_schema:{batch['batch_index']}")
        indices = [int(value) for value in batch["record_indices"]]
        seen_indices.extend(indices)
        if len(indices) != int(envelope.get("record_count", -1)):
            failures.append(f"batch_count:{batch['batch_index']}")
        observed = [
            stable_json_sha256(package["canonical_evidence"])
            for package in envelope["packages"]
        ]
        expected = [
            str(manifest["records"][index]["evidence_sha256"]) for index in indices
        ]
        if observed != expected:
            failures.append(f"batch_record_binding:{batch['batch_index']}")
        for index, package in zip(indices, envelope["packages"]):
            if package != packages_by_index.get(index):
                failures.append(f"batch_package_copy:{batch['batch_index']}:{index}")

    checks["batch_partition_exact"] = seen_indices == list(range(513))
    settlement = read_json(workload / "settlement_labels.json")
    settlement_ids = [str(value["sample_id"]) for value in settlement["labels"]]
    checks["settlement_labels_separate_and_exact"] = settlement_ids == [
        str(value["sample_id"]) for value in manifest["records"]
    ]
    checks["runtime_packages_label_free"] = not any(
        failure.startswith(("runtime_label", "batch_label")) for failure in failures
    )
    checks["all_packages_recompute"] = not any(
        failure.startswith("record_verify") for failure in failures
    )
    passed = all(checks.values()) and not failures
    report = {
        "schema": "multijudge-evidence-v2-audit",
        "passed": passed,
        "sample_count": len(packages_by_index),
        "batch_count": len(manifest["batches"]),
        "checks": checks,
        "failures": failures,
        "manifest_sha256": manifest_hash,
        "fusion_config_sha256": config_hash,
        "labels_accessed_for_runtime_verification": False,
    }
    write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Evidence V2 audit failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        type=Path,
        default=V15_ROOT / "results/replay/workload",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=V15_ROOT / "results/mmds/fit/fusion_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V15_ROOT / "results/replay/workload/audit.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
