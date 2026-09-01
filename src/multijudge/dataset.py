from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    CanonicalSample,
    DialogueTurn,
    canonical_json,
    sha256_text,
)


def _as_image_references(turn: dict[str, Any]) -> tuple[str, ...]:
    images = turn.get("image") or []
    if isinstance(images, str):
        images = [images]
    return tuple(str(value) for value in images)


def _canonical_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label not in {"safe", "unsafe"}:
        raise ValueError(f"Expected a binary assistant label, found {value!r}")
    return label


def _record_conversation_sha256(record: dict[str, Any]) -> str:
    return sha256_text(canonical_json(record.get("conversations") or []))


def read_jsonl_lines(path: Path, line_numbers: set[int]) -> dict[int, dict[str, Any]]:
    if not line_numbers:
        return {}
    found: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number not in line_numbers:
                continue
            found[line_number] = json.loads(line)
            if len(found) == len(line_numbers):
                break
    missing = sorted(line_numbers - set(found))
    if missing:
        raise ValueError(f"Manifest references missing JSONL lines: {missing[:10]}")
    return found


def load_mmds_manifest(
    dataset_root: Path,
    manifest_path: Path,
) -> list[CanonicalSample]:
    dataset_root = dataset_root.resolve()
    manifest_path = manifest_path.resolve()
    if manifest_path.suffix.casefold() == ".jsonl":
        entries = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a nonempty entries array")

    metadata_path = dataset_root / "MMDS" / "mmds.jsonl"
    records = read_jsonl_lines(
        metadata_path,
        {int(entry["jsonl_line"]) for entry in entries},
    )
    samples: list[CanonicalSample] = []
    seen_sample_ids: set[str] = set()

    for entry in entries:
        line_number = int(entry["jsonl_line"])
        record = records[line_number]
        split = str(record.get("set", "")).strip().lower()
        raw_id = str(record.get("id"))
        sample_id = f"mmds:{split}:{line_number}:{raw_id}"
        if sample_id != entry.get("sample_id"):
            raise ValueError(
                f"Manifest sample ID mismatch at line {line_number}: "
                f"{entry.get('sample_id')!r} != {sample_id!r}"
            )
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate manifest sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)

        conversation_digest = _record_conversation_sha256(record)
        if conversation_digest != entry.get("conversation_sha256"):
            raise ValueError(f"Conversation hash mismatch for {sample_id}")

        turns: list[DialogueTurn] = []
        for raw_turn in record.get("conversations") or []:
            role = str(raw_turn.get("role", "")).strip().lower()
            references = _as_image_references(raw_turn)
            paths = tuple(dataset_root / "MMDS" / reference for reference in references)
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing images for {sample_id}: {missing[:3]}"
                )
            turns.append(
                DialogueTurn(
                    role=role,  # type: ignore[arg-type]
                    content=str(raw_turn.get("content", "")),
                    image_references=references,
                    image_paths=paths,
                )
            )

        expected_label = None
        expected_dimension = None
        if "assistant_label" in entry:
            expected_label = _canonical_label(entry["assistant_label"])
            source_label = _canonical_label(record.get("assistant_rating"))
            if expected_label != source_label:
                raise ValueError(f"Label mismatch for {sample_id}")
            expected_dimension = (
                str(entry.get("assistant_dimension", "")).strip()
                or str(record.get("assistant_dimension", "")).strip()
                or "unspecified"
            )

        samples.append(
            CanonicalSample(
                sample_id=sample_id,
                dataset="MMDS",
                split=str(entry.get("softfusion_split") or split),
                raw_id=raw_id,
                jsonl_line=line_number,
                expected_label=expected_label,
                expected_dimension=expected_dimension,
                turns=tuple(turns),
                conversation_sha256=conversation_digest,
            )
        )
    return samples


def _manifest_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if manifest_path.suffix.casefold() == ".jsonl":
        entries = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a nonempty entries array")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Manifest entries must be objects")
    return entries


def load_explicit_manifest(
    dataset_root: Path,
    manifest_path: Path,
) -> list[CanonicalSample]:
    dataset_root = dataset_root.resolve()
    entries = _manifest_entries(manifest_path.resolve())
    samples: list[CanonicalSample] = []
    seen_sample_ids: set[str] = set()
    for line_number, entry in enumerate(entries, start=1):
        if entry.get("manifest_format") != "canonical-explicit-v1":
            raise ValueError(f"Unsupported explicit manifest format at line {line_number}")
        sample_id = str(entry["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate manifest sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)
        raw_turns = entry.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError(f"Missing turns for {sample_id}")
        conversation_digest = sha256_text(canonical_json(raw_turns))
        if conversation_digest != entry.get("conversation_sha256"):
            raise ValueError(f"Conversation hash mismatch for {sample_id}")
        turns: list[DialogueTurn] = []
        for raw_turn in raw_turns:
            references = raw_turn.get("image_references") or []
            if isinstance(references, str):
                references = [references]
            image_paths = tuple(dataset_root / str(value) for value in references)
            missing = [str(path) for path in image_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing images for {sample_id}: {missing[:3]}"
                )
            turns.append(
                DialogueTurn(
                    role=str(raw_turn.get("role", "")).strip().lower(),  # type: ignore[arg-type]
                    content=str(raw_turn.get("content", "")),
                    image_references=tuple(str(value) for value in references),
                    image_paths=image_paths,
                )
            )
        expected_label = None
        if "assistant_label" in entry:
            expected_label = _canonical_label(entry["assistant_label"])
        samples.append(
            CanonicalSample(
                sample_id=sample_id,
                dataset=str(entry["dataset"]),
                split=str(entry["split"]),
                raw_id=str(entry.get("raw_id", line_number)),
                jsonl_line=int(entry.get("jsonl_line", line_number)),
                expected_label=expected_label,
                expected_dimension=(
                    None
                    if entry.get("assistant_dimension") is None
                    else str(entry["assistant_dimension"])
                ),
                turns=tuple(turns),
                conversation_sha256=conversation_digest,
            )
        )
    return samples


def load_dataset_manifest(
    dataset_root: Path,
    manifest_path: Path,
) -> list[CanonicalSample]:
    entries = _manifest_entries(manifest_path.resolve())
    if entries[0].get("manifest_format") == "canonical-explicit-v1":
        return load_explicit_manifest(dataset_root, manifest_path)
    return load_mmds_manifest(dataset_root, manifest_path)


def dialogue_transcript(sample: CanonicalSample) -> str:
    lines: list[str] = []
    image_index = 0
    for turn_index, turn in enumerate(sample.turns, start=1):
        role = turn.role.upper()
        markers: list[str] = []
        for _ in turn.image_paths:
            image_index += 1
            markers.append(f"[IMAGE_{image_index:02d}]")
        marker_text = " ".join(markers)
        header = f"[TURN_{turn_index:02d}][{role}]"
        if marker_text:
            header = f"{header} {marker_text}"
        lines.extend((header, turn.content.strip(), ""))
    return "\n".join(lines).strip()


def iter_image_paths(sample: CanonicalSample) -> Iterable[Path]:
    for turn in sample.turns:
        yield from turn.image_paths
