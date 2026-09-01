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


EXPECTED_SOURCE_SHA256 = (
    "cc40bc880fb948f96494daac5e4b72f50984d3541e2f75a101aeca2a8d2ec0f4"
)
EXPECTED_REVISION = "53341884f7760e434481d88da02d50668b19d566"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source.resolve()
    download_manifest_path = args.download_manifest.resolve()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()

    download = _load_object(download_manifest_path)
    protocol = _load_object(protocol_path)
    if download.get("sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Download manifest source hash is not the approved Falcon-test hash")
    if download.get("source_revision") != EXPECTED_REVISION:
        raise RuntimeError("Download manifest revision is not the approved Falcon revision")
    if download.get("bulk_labels_inspected") is not False:
        raise RuntimeError("Download manifest no longer records label blinding")
    if protocol.get("status") != "specified-before-bulk-csv-read":
        raise RuntimeError("Protocol status is not eligible for prospective freeze")
    if protocol.get("dataset", {}).get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Protocol source hash does not match the approved dataset")

    actual_source_sha256 = sha256_file(source_path)
    if actual_source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Falcon-test source hash mismatch: {actual_source_sha256}"
        )

    record = {
        "schema": "falcon-prospective-protocol-freeze-v1",
        "status": "frozen-before-bulk-csv-read",
        "frozen_on": "2026-08-02",
        "source_file": str(source_path),
        "source_file_bytes": source_path.stat().st_size,
        "source_sha256": actual_source_sha256,
        "source_revision": EXPECTED_REVISION,
        "download_manifest": str(download_manifest_path),
        "download_manifest_sha256": sha256_file(download_manifest_path),
        "protocol": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_content_sha256": sha256_text(canonical_json(protocol)),
        "bulk_csv_read_before_freeze": False,
        "labels_accessed_before_freeze": False,
        "external_predictions_generated": False,
        "confirmation_labels_opened": False,
    }
    record["freeze_record_sha256"] = sha256_text(canonical_json(record))
    _write_json(output_path, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    dataset_root = V15_ROOT.parent / "datasets" / "Falcon"
    parser.add_argument(
        "--source", type=Path, default=dataset_root / "Falcon_test.csv"
    )
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=dataset_root / "download_manifest.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "falcon_prospective_protocol_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            V15_ROOT
            / "results"
            / "external"
            / "falcon_test"
            / "protocol_freeze.json"
        ),
    )
    return parser


if __name__ == "__main__":
    print(json.dumps(freeze(build_parser().parse_args()), ensure_ascii=False, indent=2))
