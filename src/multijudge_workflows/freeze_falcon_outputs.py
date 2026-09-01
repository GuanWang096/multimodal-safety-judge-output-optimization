from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402
from multijudge.soft_scores import validate_soft_record  # noqa: E402


JUDGE_ORDER = ("qwen", "safework", "minicpm")
ADAPTER_VERSIONS = {
    "qwen": "qwen3-vl-soft-score-v1",
    "safework": "safework-rm-safety-native-sequence-sum-v3",
    "minicpm": "minicpm-v4.5-soft-score-v1",
}
FORBIDDEN_FIELDS = {
    "assistant_label",
    "expected_label",
    "ground_truth",
    "is_response_safe",
    "label",
    "target",
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


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    result_root = args.result_root.resolve()
    protocol_path = args.protocol.resolve()
    protocol_freeze_path = args.protocol_freeze.resolve()
    preparation_path = args.preparation.resolve()
    output_path = args.output.resolve()

    manifest = _load_jsonl(manifest_path)
    protocol = _load_json(protocol_path)
    protocol_freeze = _load_json(protocol_freeze_path)
    preparation = _load_json(preparation_path)
    protocol_freeze_hash = _verify_embedded_hash(
        protocol_freeze, "freeze_record_sha256"
    )
    preparation_hash = _verify_embedded_hash(preparation, "content_sha256")
    if protocol_freeze.get("status") != "frozen-before-bulk-csv-read":
        raise RuntimeError("Falcon protocol freeze is invalid")
    if protocol_freeze.get("protocol_file_sha256") != sha256_file(protocol_path):
        raise RuntimeError("Falcon protocol changed after freeze")
    if preparation.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Falcon manifest differs from preparation record")
    if len(manifest) != 1778:
        raise RuntimeError(f"Expected 1778 Falcon records, found {len(manifest)}")
    if any(
        key.casefold() in FORBIDDEN_FIELDS
        for record in manifest
        for key in record
    ):
        raise RuntimeError("Falcon inference manifest contains labels")
    expected_ids = [str(record["sample_id"]) for record in manifest]

    judges: dict[str, Any] = {}
    for judge in JUDGE_ORDER:
        formal = result_root / judge
        paths = {
            "judgments": formal / "judgments.jsonl",
            "config": formal / "config.json",
            "status": formal / "status.json",
            "coverage_audit": formal / "coverage_audit.json",
            "model_fingerprint": formal / "model_fingerprint.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Falcon outputs for {judge}: {missing}")
        records = _load_jsonl(paths["judgments"])
        if [str(record["sample_id"]) for record in records] != expected_ids:
            raise RuntimeError(f"Falcon output order mismatch for {judge}")
        if any(
            key.casefold() in FORBIDDEN_FIELDS
            for record in records
            for key in record
        ):
            raise RuntimeError(f"Falcon output contains labels for {judge}")
        for record in records:
            validate_soft_record(record)
        versions = {
            str(record["judge_service"]["adapter_version"])
            for record in records
        }
        if versions != {ADAPTER_VERSIONS[judge]}:
            raise RuntimeError(f"Unexpected Falcon adapter version for {judge}: {versions}")
        status = _load_json(paths["status"])
        coverage = _load_json(paths["coverage_audit"])
        if status.get("state") != "completed" or coverage.get("passed") is not True:
            raise RuntimeError(f"Falcon output audit did not pass for {judge}")
        if coverage.get("labels_accessed") is not False:
            raise RuntimeError(f"Falcon output audit reports label access for {judge}")
        judges[judge] = {
            "adapter_version": ADAPTER_VERSIONS[judge],
            "record_count": len(records),
            "files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        }

    frozen = {
        "schema": "falcon-test-external-raw-output-freeze-v1",
        "status": "frozen-before-adaptation-label-access",
        "dataset": "zhangrjjj/Falcon",
        "subset": "Falcon-test",
        "sample_count": len(expected_ids),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_content_sha256": sha256_text(canonical_json(protocol)),
        "protocol_freeze_record_sha256": protocol_freeze_hash,
        "preparation_record_sha256": preparation_hash,
        "judge_order": list(JUDGE_ORDER),
        "judges": judges,
        "labels_accessed": False,
        "adaptation_model_fitting_performed": False,
        "confirmation_labels_opened": False,
    }
    frozen["content_sha256"] = sha256_text(canonical_json(frozen))
    if output_path.is_file():
        prior = _load_json(output_path)
        if prior != frozen:
            raise RuntimeError("Existing Falcon raw-output freeze differs from current outputs")
    else:
        _write_json(output_path, frozen)
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
    parser.add_argument("--result-root", type=Path, default=base / "raw")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=experiment_root / "configs" / "falcon_prospective_protocol_v1.json",
    )
    parser.add_argument(
        "--protocol-freeze",
        type=Path,
        default=base / "protocol_freeze.json",
    )
    parser.add_argument(
        "--preparation",
        type=Path,
        default=base / "preparation_summary.json",
    )
    parser.add_argument(
        "--output", type=Path, default=base / "raw_output_freeze.json"
    )
    return parser


if __name__ == "__main__":
    result = freeze(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "sample_count": result["sample_count"],
                "labels_accessed": result["labels_accessed"],
            }
        )
    )
