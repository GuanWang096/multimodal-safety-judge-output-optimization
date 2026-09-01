from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


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
FORBIDDEN_RUNTIME_FIELDS = {
    "assistant_label",
    "expected_label",
    "ground_truth",
    "is_response_safe",
    "label",
    "target",
    "Instruction Judgment",
    "Image Judgment",
    "Response Judgment",
    "Harmful Category",
}
FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")
    os.replace(temporary, path)


def _verified_freeze(
    source_path: Path,
    protocol_path: Path,
    freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_object(protocol_path)
    freeze = _load_object(freeze_path)
    frozen_hash = str(freeze.get("freeze_record_sha256", ""))
    unhashed_freeze = dict(freeze)
    unhashed_freeze.pop("freeze_record_sha256", None)
    if frozen_hash != sha256_text(canonical_json(unhashed_freeze)):
        raise RuntimeError("Falcon protocol freeze record hash is invalid")
    if freeze.get("status") != "frozen-before-bulk-csv-read":
        raise RuntimeError("Falcon protocol is not prospectively frozen")
    if freeze.get("protocol_file_sha256") != sha256_file(protocol_path):
        raise RuntimeError("Falcon protocol changed after freeze")
    if freeze.get("source_sha256") != sha256_file(source_path):
        raise RuntimeError("Falcon source file changed after freeze")
    if freeze.get("confirmation_labels_opened") is not False:
        raise RuntimeError("Confirmation labels were already marked as opened")
    return protocol, freeze


def _decode_image_data_url(value: str) -> bytes:
    prefix, separator, payload = value.partition(",")
    normalized_prefix = prefix.casefold()
    allowed_prefixes = {"data:image;base64", "data:image/png;base64"}
    if not separator or normalized_prefix not in allowed_prefixes:
        raise ValueError(f"Unsupported image data URL prefix: {prefix!r}")
    compact_payload = "".join(payload.split())
    try:
        image_bytes = base64.b64decode(compact_payload, validate=True)
    except ValueError as exc:
        raise ValueError("Invalid base64 image payload") from exc
    if not image_bytes:
        raise ValueError("Decoded image is empty")
    return image_bytes


def _image_properties(image_bytes: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image_format = str(image.format or "").upper()
        dimensions = (int(image.width), int(image.height))
    extension = FORMAT_EXTENSIONS.get(image_format)
    if extension is None:
        raise ValueError(f"Unsupported embedded image format: {image_format!r}")
    return extension, dimensions


def _assistant_label(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized == "benign":
        return "safe"
    if normalized == "harmful":
        return "unsafe"
    raise ValueError(f"Unexpected Falcon response judgment: {value!r}")


def _partition_groups(
    group_ids: Iterable[str], seed_text: str, adaptation_fraction: float
) -> dict[str, str]:
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required for a prospective split")
    if not 0.0 < adaptation_fraction < 1.0:
        raise ValueError("Adaptation fraction must be between zero and one")
    ranked = sorted(
        unique_groups,
        key=lambda group: hashlib.sha256(
            f"{seed_text}|{group}".encode("utf-8")
        ).hexdigest(),
    )
    adaptation_count = math.floor(adaptation_fraction * len(ranked))
    if adaptation_count == 0 or adaptation_count == len(ranked):
        raise ValueError("Prospective split produced an empty partition")
    adaptation = set(ranked[:adaptation_count])
    return {
        group: ("adaptation" if group in adaptation else "confirmation")
        for group in ranked
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source.resolve()
    dataset_root = args.dataset_root.resolve()
    protocol_path = args.protocol.resolve()
    freeze_path = args.freeze_record.resolve()
    manifest_path = args.manifest.resolve()
    adaptation_labels_path = args.adaptation_labels.resolve()
    confirmation_labels_path = args.confirmation_labels.resolve()
    partition_path = args.partition_manifest.resolve()
    seal_path = args.confirmation_seal.resolve()
    summary_path = args.summary.resolve()
    image_root = dataset_root / "Falcon" / "external-images"
    image_root.mkdir(parents=True, exist_ok=True)

    protocol, freeze = _verified_freeze(source_path, protocol_path, freeze_path)
    expected_records = int(protocol["dataset"]["expected_source_records"])
    excluded_ids = {int(value) for value in protocol["exclusions"]["raw_ids"]}
    split_contract = protocol["grouping_and_split"]
    adaptation_fraction = float(split_contract["adaptation_fraction"])
    split_seed_text = str(split_contract["split_seed_text"])

    csv.field_size_limit(2**31 - 1)
    records: list[dict[str, Any]] = []
    seen_raw_ids: set[int] = set()
    unique_image_hashes: set[str] = set()
    image_widths: list[int] = []
    image_heights: list[int] = []
    group_sizes: Counter[str] = Counter()

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"Unexpected Falcon-test columns: {reader.fieldnames!r}"
            )
        for source_row, row in enumerate(reader, start=2):
            try:
                raw_id = int(str(row["id"]).strip())
            except ValueError as exc:
                raise RuntimeError(f"Invalid Falcon ID at CSV row {source_row}") from exc
            if raw_id in seen_raw_ids:
                raise RuntimeError(f"Duplicate Falcon raw ID: {raw_id}")
            seen_raw_ids.add(raw_id)

            query = str(row["query"]).strip()
            response = str(row["response"]).strip()
            if not query or not response:
                raise RuntimeError(f"Empty query or response for Falcon ID {raw_id}")
            image_bytes = _decode_image_data_url(str(row["image"]))
            extension, dimensions = _image_properties(image_bytes)
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()

            if raw_id in excluded_ids:
                continue

            image_path = image_root / f"{image_sha256}{extension}"
            if image_path.is_file():
                if sha256_file(image_path) != image_sha256:
                    raise RuntimeError(f"Existing image hash mismatch: {image_path}")
            else:
                image_path.write_bytes(image_bytes)
            image_reference = image_path.relative_to(dataset_root).as_posix()

            group_payload = {
                "image_sha256": image_sha256,
                "query": query,
            }
            group_id = f"falcongrp-{sha256_text(canonical_json(group_payload))[:16]}"
            sample_id = f"falcon:test:{raw_id}"
            turns = [
                {
                    "role": "user",
                    "content": query,
                    "image_references": [image_reference],
                },
                {
                    "role": "assistant",
                    "content": response,
                    "image_references": [],
                },
            ]
            records.append(
                {
                    "raw_id": raw_id,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "image_sha256": image_sha256,
                    "image_reference": image_reference,
                    "turns": turns,
                    "assistant_label": _assistant_label(row["Response Judgment"]),
                    "instruction_judgment": str(row["Instruction Judgment"]).strip(),
                    "image_judgment": str(row["Image Judgment"]).strip(),
                    "response_judgment": str(row["Response Judgment"]).strip(),
                    "harmful_category": str(row["Harmful Category"]).strip(),
                }
            )
            unique_image_hashes.add(image_sha256)
            image_widths.append(dimensions[0])
            image_heights.append(dimensions[1])
            group_sizes[group_id] += 1

    if len(seen_raw_ids) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} Falcon rows, found {len(seen_raw_ids)}"
        )
    if seen_raw_ids != set(range(expected_records)):
        raise RuntimeError("Falcon raw IDs are not the expected contiguous 0-based range")
    expected_selected = expected_records - len(excluded_ids)
    if len(records) != expected_selected:
        raise RuntimeError(
            f"Expected {expected_selected} eligible Falcon records, found {len(records)}"
        )

    partitions = _partition_groups(
        (record["group_id"] for record in records),
        split_seed_text,
        adaptation_fraction,
    )
    entries: list[dict[str, Any]] = []
    adaptation_labels: list[dict[str, Any]] = []
    confirmation_labels: list[dict[str, Any]] = []
    partition_sample_counts: Counter[str] = Counter()
    partition_group_counts: Counter[str] = Counter(partitions.values())

    for line_number, record in enumerate(records, start=1):
        partition = partitions[record["group_id"]]
        entry = {
            "manifest_format": "canonical-explicit-v1",
            "sample_id": record["sample_id"],
            "dataset": "Falcon-test",
            "split": f"external-{partition}",
            "prospective_partition": partition,
            "raw_id": str(record["raw_id"]),
            "jsonl_line": line_number,
            "source_row": record["raw_id"] + 2,
            "group_id": record["group_id"],
            "image_sha256": record["image_sha256"],
            "image_references": [record["image_reference"]],
            "conversation_sha256": sha256_text(canonical_json(record["turns"])),
            "turns": record["turns"],
        }
        forbidden = FORBIDDEN_RUNTIME_FIELDS.intersection(entry)
        if forbidden:
            raise RuntimeError(f"Runtime entry contains labels: {sorted(forbidden)}")
        entries.append(entry)
        label_record = {
            "schema": "falcon-test-sealed-label-v1",
            "sample_id": record["sample_id"],
            "group_id": record["group_id"],
            "prospective_partition": partition,
            "assistant_label": record["assistant_label"],
            "instruction_judgment": record["instruction_judgment"],
            "image_judgment": record["image_judgment"],
            "response_judgment": record["response_judgment"],
            "harmful_category": record["harmful_category"],
        }
        if partition == "adaptation":
            adaptation_labels.append(label_record)
        else:
            confirmation_labels.append(label_record)
        partition_sample_counts[partition] += 1

    if len(entries) != len({entry["sample_id"] for entry in entries}):
        raise RuntimeError("Duplicate Falcon sample IDs after preparation")
    if set(partitions.values()) != {"adaptation", "confirmation"}:
        raise RuntimeError("Falcon prospective split is incomplete")
    if any(
        key in entry
        for entry in entries
        for key in FORBIDDEN_RUNTIME_FIELDS
    ):
        raise RuntimeError("Falcon GPU manifest contains a forbidden label field")

    _write_jsonl(manifest_path, entries)
    _write_jsonl(adaptation_labels_path, adaptation_labels)
    _write_jsonl(confirmation_labels_path, confirmation_labels)
    partition_manifest = {
        "schema": "falcon-prospective-partition-manifest-v1",
        "protocol_file_sha256": sha256_file(protocol_path),
        "split_seed_text": split_seed_text,
        "adaptation_fraction": adaptation_fraction,
        "group_assignment": dict(sorted(partitions.items())),
        "partition_group_counts": dict(sorted(partition_group_counts.items())),
        "partition_sample_counts": dict(sorted(partition_sample_counts.items())),
        "labels_used_for_split": False,
        "model_outputs_used_for_split": False,
    }
    partition_manifest["content_sha256"] = sha256_text(
        canonical_json(partition_manifest)
    )
    _write_json(partition_path, partition_manifest)

    confirmation_seal = {
        "schema": "falcon-confirmation-label-seal-v1",
        "status": "sealed-not-opened",
        "confirmation_sample_count": len(confirmation_labels),
        "confirmation_label_file": str(confirmation_labels_path),
        "confirmation_label_file_sha256": sha256_file(confirmation_labels_path),
        "protocol_freeze_record_sha256": freeze["freeze_record_sha256"],
        "opened": False,
        "model_freeze_required_before_open": True,
    }
    confirmation_seal["seal_sha256"] = sha256_text(
        canonical_json(confirmation_seal)
    )
    _write_json(seal_path, confirmation_seal)

    summary = {
        "schema": "falcon-prospective-preparation-v1",
        "status": "prepared-labels-separated",
        "dataset": "zhangrjjj/Falcon",
        "subset": "Falcon-test",
        "source_revision": protocol["dataset"]["source_revision"],
        "source_sha256": sha256_file(source_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_freeze_record_sha256": freeze["freeze_record_sha256"],
        "source_record_count": len(seen_raw_ids),
        "excluded_preview_record_count": len(excluded_ids),
        "eligible_sample_count": len(entries),
        "group_count": len(partitions),
        "unique_image_count": len(unique_image_hashes),
        "partition_group_counts": dict(sorted(partition_group_counts.items())),
        "partition_sample_counts": dict(sorted(partition_sample_counts.items())),
        "group_size_counts": {
            str(size): count
            for size, count in sorted(Counter(group_sizes.values()).items())
        },
        "image_width_range": [min(image_widths), max(image_widths)],
        "image_height_range": [min(image_heights), max(image_heights)],
        "unique_image_dimension_count": len(
            set(zip(image_widths, image_heights, strict=True))
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "partition_manifest_path": str(partition_path),
        "partition_manifest_sha256": sha256_file(partition_path),
        "adaptation_labels_path": str(adaptation_labels_path),
        "adaptation_labels_sha256": sha256_file(adaptation_labels_path),
        "confirmation_seal_path": str(seal_path),
        "confirmation_seal_sha256": sha256_file(seal_path),
        "runtime_labels_embedded": False,
        "label_counts_reported_before_model_freeze": False,
        "confirmation_labels_opened": False,
    }
    summary["content_sha256"] = sha256_text(canonical_json(summary))
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    dataset_root = V15_ROOT.parent / "datasets"
    experiment_root = Path(__file__).resolve().parent
    result_root = (
        V15_ROOT
        / "results"
        / "external"
        / "falcon_test"
    )
    preparation_root = result_root
    parser.add_argument(
        "--source",
        type=Path,
        default=dataset_root / "Falcon" / "Falcon_test.csv",
    )
    parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=experiment_root / "configs" / "falcon_prospective_protocol_v1.json",
    )
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=result_root / "protocol_freeze.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=experiment_root / "configs" / "falcon_external_unlabeled.jsonl",
    )
    parser.add_argument(
        "--adaptation-labels",
        type=Path,
        default=preparation_root / "falcon_adaptation_labels.jsonl",
    )
    parser.add_argument(
        "--confirmation-labels",
        type=Path,
        default=preparation_root / "falcon_confirmation_labels.sealed.jsonl",
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=preparation_root / "partition_manifest.json",
    )
    parser.add_argument(
        "--confirmation-seal",
        type=Path,
        default=preparation_root / "confirmation_label_seal.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=preparation_root / "preparation_summary.json",
    )
    return parser


if __name__ == "__main__":
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
