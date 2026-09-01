from __future__ import annotations

import argparse
import base64
import csv
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

from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


EXPECTED_SOURCE_SHA256 = (
    "cc40bc880fb948f96494daac5e4b72f50984d3541e2f75a101aeca2a8d2ec0f4"
)
EXPECTED_COLUMNS = (
    "id",
    "query",
    "response",
    "Instruction Judgment",
    "Image Judgment",
    "Response Judgment",
    "Harmful Category",
    "image",
)
ALLOWED_IMAGE_PREFIXES = {"data:image;base64", "data:image/png;base64"}


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


def _decode_image(value: str) -> bytes:
    prefix, separator, payload = value.partition(",")
    if not separator or prefix.casefold() not in ALLOWED_IMAGE_PREFIXES:
        raise ValueError(f"Unsupported Falcon image prefix: {prefix!r}")
    try:
        decoded = base64.b64decode("".join(payload.split()), validate=True)
    except ValueError as exc:
        raise ValueError("Invalid Falcon base64 image payload") from exc
    if not decoded:
        raise ValueError("Decoded Falcon image is empty")
    return decoded


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source.resolve()
    manifest_path = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    output_path = args.output.resolve()
    if sha256_file(source_path) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Downloaded Falcon_test.csv failed its frozen SHA256 check")

    manifest = _load_jsonl(manifest_path)
    if len(manifest) != 1778:
        raise RuntimeError(f"Expected 1778 Falcon manifest entries, found {len(manifest)}")
    expected_by_raw_id = {int(record["raw_id"]): record for record in manifest}
    if len(expected_by_raw_id) != len(manifest):
        raise RuntimeError("Falcon manifest contains duplicate raw IDs")
    forbidden = {
        "assistant_label",
        "expected_label",
        "ground_truth",
        "label",
        "target",
        "Instruction Judgment",
        "Image Judgment",
        "Response Judgment",
        "Harmful Category",
    }
    if any(forbidden.intersection(record) for record in manifest):
        raise RuntimeError("Falcon inference manifest contains label fields")

    csv.field_size_limit(2**31 - 1)
    observed_ids: set[int] = set()
    written = 0
    reused = 0
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader))
        if header != EXPECTED_COLUMNS:
            raise RuntimeError(f"Unexpected Falcon CSV columns: {header!r}")
        for source_row, values in enumerate(reader, start=2):
            if len(values) != len(EXPECTED_COLUMNS):
                raise RuntimeError(f"Malformed Falcon CSV row {source_row}")
            raw_id = int(values[0].strip())
            expected = expected_by_raw_id.get(raw_id)
            if expected is None:
                continue
            if raw_id in observed_ids:
                raise RuntimeError(f"Duplicate Falcon raw ID: {raw_id}")
            observed_ids.add(raw_id)

            query = values[1].strip()
            response = values[2].strip()
            image_bytes = _decode_image(values[7])
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            if image_sha256 != expected["image_sha256"]:
                raise RuntimeError(f"Falcon image hash mismatch for raw ID {raw_id}")
            references = expected.get("image_references") or []
            if len(references) != 1:
                raise RuntimeError(f"Unexpected image reference count for raw ID {raw_id}")
            turns = [
                {
                    "role": "user",
                    "content": query,
                    "image_references": references,
                },
                {
                    "role": "assistant",
                    "content": response,
                    "image_references": [],
                },
            ]
            if sha256_text(canonical_json(turns)) != expected["conversation_sha256"]:
                raise RuntimeError(f"Falcon conversation hash mismatch for raw ID {raw_id}")

            image_path = dataset_root / str(references[0])
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.is_file():
                if sha256_file(image_path) != image_sha256:
                    raise RuntimeError(f"Existing Falcon image changed: {image_path}")
                reused += 1
            else:
                temporary = image_path.with_suffix(image_path.suffix + ".tmp")
                temporary.write_bytes(image_bytes)
                os.replace(temporary, image_path)
                written += 1

    if observed_ids != set(expected_by_raw_id):
        missing = sorted(set(expected_by_raw_id) - observed_ids)
        raise RuntimeError(f"Falcon source did not cover the manifest: {missing[:10]}")

    audit = {
        "schema": "falcon-autodl-image-materialization-audit-v1",
        "passed": True,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_record_count": len(manifest),
        "materialized_record_count": len(observed_ids),
        "new_image_file_count": written,
        "reused_image_reference_count": reused,
        "source_label_columns_accessed": False,
        "runtime_manifest_contains_labels": False,
    }
    audit["content_sha256"] = sha256_text(canonical_json(audit))
    _write_json(output_path, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "falcon_external_unlabeled.jsonl"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=V15_ROOT.parent / "datasets")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            V15_ROOT
            / "results"
            / "external"
            / "falcon_test"
            / "image_materialization_audit.json"
        ),
    )
    return parser


if __name__ == "__main__":
    result = materialize(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
