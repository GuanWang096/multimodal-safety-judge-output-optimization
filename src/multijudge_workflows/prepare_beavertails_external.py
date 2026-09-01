from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


FORBIDDEN_RUNTIME_FIELDS = {
    "assistant_label",
    "expected_label",
    "ground_truth",
    "is_response_safe",
    "label",
    "target",
}
FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")
    os.replace(temporary, path)


def _image_extension(image_bytes: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image_format = str(image.format or "").upper()
        size = (int(image.width), int(image.height))
    extension = FORMAT_EXTENSIONS.get(image_format)
    if extension is None:
        raise ValueError(f"Unsupported embedded image format: {image_format!r}")
    return extension, size


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    dataset_root = args.dataset_root.resolve()
    manifest_path = args.manifest.resolve()
    labels_path = args.labels.resolve()
    summary_path = args.summary.resolve()
    image_root = dataset_root / "BeaverTails-V" / "external-images"
    image_root.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(source_root.glob("*/evaluation.parquet"))
    if len(parquet_files) != 20:
        raise RuntimeError(f"Expected 20 evaluation parquet files, found {len(parquet_files)}")

    entries: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    source_files = []
    category_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    image_sizes: Counter[str] = Counter()
    image_hashes: set[str] = set()
    group_counts: Counter[str] = Counter()

    for parquet_path in parquet_files:
        source_config = parquet_path.parent.name
        frame = pd.read_parquet(parquet_path)
        expected_columns = {
            "question",
            "image",
            "category",
            "image_severity",
            "response",
            "is_response_safe",
        }
        if set(frame.columns) != expected_columns:
            raise RuntimeError(f"Unexpected columns in {parquet_path}")
        source_files.append(
            {
                "path": parquet_path.relative_to(source_root.parent.parent).as_posix(),
                "sha256": sha256_file(parquet_path),
                "record_count": int(len(frame)),
            }
        )
        for source_row, row in frame.iterrows():
            category = str(row["category"]).strip()
            if category != source_config:
                raise RuntimeError(
                    f"Category/config mismatch in {parquet_path}:{source_row}"
                )
            raw_label = str(row["is_response_safe"]).strip().lower()
            if raw_label not in {"yes", "no"}:
                raise RuntimeError(f"Unexpected response label: {raw_label!r}")
            assistant_label = "safe" if raw_label == "yes" else "unsafe"
            image_bytes = row["image"]
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise RuntimeError(f"Missing image bytes in {parquet_path}:{source_row}")
            extension, dimensions = _image_extension(image_bytes)
            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            image_path = image_root / f"{image_sha256}{extension}"
            if image_path.is_file():
                if sha256_file(image_path) != image_sha256:
                    raise RuntimeError(f"Existing image hash mismatch: {image_path}")
            else:
                image_path.write_bytes(image_bytes)
            image_reference = image_path.relative_to(dataset_root).as_posix()

            question = str(row["question"]).strip()
            response = str(row["response"]).strip()
            if not question or not response:
                raise RuntimeError(f"Empty question/response in {parquet_path}:{source_row}")
            group_payload = {
                "category": category,
                "image_sha256": image_sha256,
                "question": question,
            }
            group_id = f"btvgrp-{sha256_text(canonical_json(group_payload))[:16]}"
            sample_id = (
                f"beavertails-v:evaluation:{source_config}:{int(source_row):04d}"
            )
            turns = [
                {
                    "role": "user",
                    "content": question,
                    "image_references": [image_reference],
                },
                {
                    "role": "assistant",
                    "content": response,
                    "image_references": [],
                },
            ]
            entry = {
                "manifest_format": "canonical-explicit-v1",
                "sample_id": sample_id,
                "dataset": "BeaverTails-V",
                "split": "external-evaluation",
                "raw_id": f"{source_config}:{int(source_row)}",
                "jsonl_line": len(entries) + 1,
                "source_config": source_config,
                "source_row": int(source_row),
                "group_id": group_id,
                "category": category,
                "image_severity": int(row["image_severity"]),
                "image_sha256": image_sha256,
                "image_references": [image_reference],
                "conversation_sha256": sha256_text(canonical_json(turns)),
                "turns": turns,
            }
            forbidden = FORBIDDEN_RUNTIME_FIELDS.intersection(entry)
            if forbidden:
                raise RuntimeError(f"Runtime entry contains labels: {sorted(forbidden)}")
            entries.append(entry)
            labels.append(
                {
                    "sample_id": sample_id,
                    "assistant_label": assistant_label,
                    "category": category,
                    "image_severity": int(row["image_severity"]),
                    "group_id": group_id,
                }
            )
            category_counts[category] += 1
            label_counts[assistant_label] += 1
            severity_counts[str(int(row["image_severity"]))] += 1
            image_sizes[f"{dimensions[0]}x{dimensions[1]}"] += 1
            image_hashes.add(image_sha256)
            group_counts[group_id] += 1

    sample_ids = [entry["sample_id"] for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate external sample IDs")
    if set(group_counts.values()) != {2}:
        raise RuntimeError(
            f"Expected exactly two candidate responses per group, found {Counter(group_counts.values())}"
        )
    if len(entries) != 1180 or label_counts != Counter({"safe": 741, "unsafe": 439}):
        raise RuntimeError(
            f"Unexpected frozen evaluation composition: {len(entries)}, {dict(label_counts)}"
        )

    _write_jsonl(manifest_path, entries)
    _write_jsonl(labels_path, labels)
    summary = {
        "schema": "beavertails-v-external-preparation-v1",
        "status": "frozen",
        "dataset": "PKU-Alignment/BeaverTails-V",
        "source_split": "evaluation",
        "license": "CC-BY-NC-4.0",
        "sample_count": len(entries),
        "group_count": len(group_counts),
        "category_count": len(category_counts),
        "unique_image_count": len(image_hashes),
        "label_counts": dict(sorted(label_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "group_size_counts": {
            str(size): count
            for size, count in sorted(Counter(group_counts.values()).items())
        },
        "image_dimension_counts": dict(sorted(image_sizes.items())),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "labels_path": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "runtime_labels_embedded": False,
        "source_files": source_files,
    }
    summary["content_sha256"] = sha256_text(canonical_json(summary))
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    dataset_root = V15_ROOT.parent / "datasets"
    output_root = (
        V15_ROOT
        / "results"
        / "external"
        / "beavertails_v"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=dataset_root / "BeaverTails-V" / "data",
    )
    parser.add_argument("--dataset-root", type=Path, default=dataset_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "beavertails_v_external_unlabeled.jsonl"
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=output_root / "beavertails_v_external_labels.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=output_root / "preparation_summary.json",
    )
    return parser


if __name__ == "__main__":
    result = prepare(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
