from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


SCHEMA_VERSION = "softfusion-split-v1"
DEFAULT_SEED = 20260729
SPLITS = ("fit", "select", "confirm")
TARGET_RATIOS = {"fit": 0.60, "select": 0.175, "confirm": 0.225}
MINIMUMS = {
    "fit": {"unsafe_bearing_components": 80, "unsafe_records": 200},
    "select": {"unsafe_bearing_components": 25, "unsafe_records": 60},
    "confirm": {
        "unsafe_bearing_components": 50,
        "unsafe_records": 120,
        "safe_only_components": 100,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_label(value: Any) -> str | None:
    label = str(value or "").strip().lower()
    return label if label in {"safe", "unsafe"} else None


def normalized_data_type(value: Any) -> str:
    value = str(value or "").strip().lower()
    if value in {"augmentaion", "augmentation"}:
        return "augmentation"
    return value or "missing"


def image_references(record: dict[str, Any]) -> tuple[str, ...]:
    references: list[str] = []
    for turn in record.get("conversations") or []:
        images = turn.get("image") or []
        if isinstance(images, str):
            images = [images]
        references.extend(str(item) for item in images if str(item).strip())
    return tuple(dict.fromkeys(references))


def image_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4+"


@dataclass(frozen=True)
class Record:
    index: int
    line_number: int
    sample_id: str
    raw_id: str
    source_split: str
    label: str | None
    dimension: str
    data_type: str
    target_model: str
    image_references: tuple[str, ...]
    conversation_sha256: str


@dataclass(frozen=True)
class ImageFingerprint:
    reference: str
    pixel_sha256: str
    phash: int
    width: int
    height: int


@dataclass
class Component:
    component_id: str
    record_indices: list[int]
    record_count: int
    safe_count: int
    unsafe_count: int
    composition: str
    dominant_dimension: str


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, str, dict[int, Any]] | None = None

    def add(self, value: int, key: str) -> None:
        if self.root is None:
            self.root = (value, key, {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, key, {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        found: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node_value, node_key, children = stack.pop()
            distance = (value ^ node_value).bit_count()
            if distance <= radius:
                found.append((distance, node_key))
            lower = max(0, distance - radius)
            upper = distance + radius
            stack.extend(
                child
                for edge, child in children.items()
                if lower <= edge <= upper
            )
        return found


def read_records(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            source_split = str(raw.get("set", "")).strip().lower()
            raw_id = str(raw.get("id"))
            conversations = raw.get("conversations") or []
            records.append(
                Record(
                    index=len(records),
                    line_number=line_number,
                    sample_id=f"mmds:{source_split}:{line_number}:{raw_id}",
                    raw_id=raw_id,
                    source_split=source_split,
                    label=normalized_label(raw.get("assistant_rating")),
                    dimension=(
                        str(raw.get("assistant_dimension", "")).strip()
                        or "unspecified"
                    ),
                    data_type=normalized_data_type(raw.get("data_type")),
                    target_model=(
                        str(raw.get("target_model", "")).strip() or "unspecified"
                    ),
                    image_references=image_references(raw),
                    conversation_sha256=sha256_text(canonical_json(conversations)),
                )
            )
    return records


def _pixel_digest_and_phash(path: Path) -> tuple[str, int, int, int]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        digest = hashlib.sha256()
        digest.update(f"RGB|{width}|{height}|".encode("ascii"))
        digest.update(image.tobytes())

        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        coefficients = dctn(
            np.asarray(gray, dtype=np.float32),
            type=2,
            norm="ortho",
        )[:8, :8]
        values = coefficients.flatten()
        median = float(np.median(values[1:]))
        bits = values > median
        phash = 0
        for bit in bits:
            phash = (phash << 1) | int(bool(bit))
        return digest.hexdigest(), phash, width, height


def fingerprint_images(
    records: Sequence[Record],
    mmds_root: Path,
) -> tuple[dict[str, ImageFingerprint], list[str]]:
    references = sorted(
        {
            reference
            for record in records
            for reference in record.image_references
        }
    )
    fingerprints: dict[str, ImageFingerprint] = {}
    missing: list[str] = []
    for index, reference in enumerate(references, start=1):
        path = mmds_root / Path(reference)
        if not path.is_file():
            missing.append(reference)
            continue
        pixel_sha256, phash, width, height = _pixel_digest_and_phash(path)
        fingerprints[reference] = ImageFingerprint(
            reference=reference,
            pixel_sha256=pixel_sha256,
            phash=phash,
            width=width,
            height=height,
        )
        if index % 500 == 0:
            print(f"fingerprinted {index}/{len(references)} image references")
    return fingerprints, missing


def build_global_components(
    records: Sequence[Record],
    fingerprints: dict[str, ImageFingerprint],
    near_duplicate_decisions: dict[str, str] | None = None,
) -> DisjointSet:
    dsu = DisjointSet(len(records))
    first_by_pixel: dict[str, int] = {}
    first_by_conversation: dict[str, int] = {}
    for record in records:
        prior = first_by_conversation.setdefault(
            record.conversation_sha256, record.index
        )
        dsu.union(record.index, prior)
        for reference in record.image_references:
            fingerprint = fingerprints.get(reference)
            if fingerprint is None:
                continue
            prior = first_by_pixel.setdefault(
                fingerprint.pixel_sha256, record.index
            )
            dsu.union(record.index, prior)
    for pair_key, decision in (near_duplicate_decisions or {}).items():
        if decision != "same":
            continue
        parts = pair_key.split("__")
        if len(parts) != 2 or any(len(part) != 64 for part in parts):
            raise ValueError(f"Invalid near-duplicate pair key: {pair_key}")
        if parts[0] not in first_by_pixel or parts[1] not in first_by_pixel:
            raise ValueError(
                f"Near-duplicate decision references unknown pixel digest: {pair_key}"
            )
        dsu.union(first_by_pixel[parts[0]], first_by_pixel[parts[1]])
    return dsu


def load_near_duplicate_decisions(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Near-duplicate decisions must be a JSON object")
    decisions = {
        str(key): str(value).strip().lower() for key, value in payload.items()
    }
    invalid = sorted(
        key for key, value in decisions.items() if value not in {"same", "different"}
    )
    if invalid:
        raise ValueError(
            f"Invalid near-duplicate decisions for keys: {invalid[:5]}"
        )
    return decisions


def manifest_line_numbers(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(entry["jsonl_line"]) for entry in payload.get("entries") or []}


def exclusion_reasons(
    records: Sequence[Record],
    dsu: DisjointSet,
    fingerprints: dict[str, ImageFingerprint],
    engineering_lines: set[int],
) -> dict[int, list[str]]:
    engineering_roots = {
        dsu.find(record.index)
        for record in records
        if record.line_number in engineering_lines
    }
    legacy_roots = {
        dsu.find(record.index)
        for record in records
        if record.source_split in {"val", "test"}
    }
    reasons: dict[int, list[str]] = {}
    for record in records:
        values: list[str] = []
        if record.source_split != "train":
            values.append(f"source_split_{record.source_split or 'missing'}")
        if record.label is None:
            values.append("nonbinary_label")
        if not record.image_references:
            values.append("no_image")
        if any(ref not in fingerprints for ref in record.image_references):
            values.append("missing_image")
        root = dsu.find(record.index)
        if root in engineering_roots:
            values.append("engineering_component_overlap")
        if root in legacy_roots:
            values.append("legacy_component_overlap")
        if values:
            reasons[record.index] = values
    return reasons


def build_eligible_components(
    records: Sequence[Record],
    dsu: DisjointSet,
    exclusions: dict[int, list[str]],
) -> tuple[list[Component], dict[int, str]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for record in records:
        if record.index not in exclusions:
            grouped[dsu.find(record.index)].append(record.index)

    components: list[Component] = []
    record_to_component: dict[int, str] = {}
    for indices in grouped.values():
        sample_ids = sorted(records[index].sample_id for index in indices)
        component_id = "cmp-" + sha256_text("|".join(sample_ids))[:16]
        labels = Counter(records[index].label for index in indices)
        dimensions = Counter(records[index].dimension for index in indices)
        safe_count = labels["safe"]
        unsafe_count = labels["unsafe"]
        if safe_count and unsafe_count:
            composition = "mixed"
        elif unsafe_count:
            composition = "unsafe-only"
        else:
            composition = "safe-only"
        component = Component(
            component_id=component_id,
            record_indices=sorted(indices),
            record_count=len(indices),
            safe_count=safe_count,
            unsafe_count=unsafe_count,
            composition=composition,
            dominant_dimension=sorted(
                dimensions.items(), key=lambda item: (-item[1], item[0])
            )[0][0],
        )
        components.append(component)
        for index in indices:
            record_to_component[index] = component_id
    components.sort(key=lambda value: value.component_id)
    return components, record_to_component


def _unsafe_component_targets(count: int) -> dict[str, int]:
    targets = {
        split: max(
            MINIMUMS[split]["unsafe_bearing_components"],
            round(TARGET_RATIOS[split] * count),
        )
        for split in SPLITS
    }
    overflow = sum(targets.values()) - count
    while overflow > 0:
        candidates = [
            split
            for split in SPLITS
            if targets[split] > MINIMUMS[split]["unsafe_bearing_components"]
        ]
        if not candidates:
            raise RuntimeError(
                "Unsafe-bearing component minima exceed the available components"
            )
        split = max(
            candidates,
            key=lambda value: targets[value] - TARGET_RATIOS[value] * count,
        )
        targets[split] -= 1
        overflow -= 1
    while sum(targets.values()) < count:
        split = max(
            SPLITS,
            key=lambda value: TARGET_RATIOS[value] * count - targets[value],
        )
        targets[split] += 1
    return targets


def _fixed_count_partition(
    components: Sequence[Component],
    targets: dict[str, int],
    seed: int,
    iterations: int,
) -> tuple[dict[str, list[Component]], float]:
    rng = np.random.default_rng(seed)
    unsafe = np.asarray([component.unsafe_count for component in components])
    totals = np.asarray([component.record_count for component in components])
    total_unsafe = int(unsafe.sum())
    total_records = int(totals.sum())
    boundaries = np.cumsum([targets[split] for split in SPLITS])
    best: tuple[float, np.ndarray] | None = None

    for _ in range(iterations):
        order = rng.permutation(len(components))
        partitions = np.split(order, boundaries[:-1])
        counts = {
            split: {
                "unsafe": int(unsafe[part].sum()),
                "records": int(totals[part].sum()),
            }
            for split, part in zip(SPLITS, partitions, strict=True)
        }
        violations = sum(
            max(0, MINIMUMS[split]["unsafe_records"] - counts[split]["unsafe"])
            for split in SPLITS
        )
        score = violations * 1_000_000.0
        for split in SPLITS:
            target_unsafe = TARGET_RATIOS[split] * total_unsafe
            target_records = TARGET_RATIOS[split] * total_records
            score += ((counts[split]["unsafe"] - target_unsafe) / max(1, total_unsafe)) ** 2
            score += ((counts[split]["records"] - target_records) / max(1, total_records)) ** 2
        if best is None or score < best[0]:
            best = (score, order.copy())

    if best is None:
        raise RuntimeError("No unsafe-bearing component partition was generated")
    order = best[1]
    partitions = np.split(order, boundaries[:-1])
    result = {
        split: [components[int(index)] for index in partition]
        for split, partition in zip(SPLITS, partitions, strict=True)
    }
    for split in SPLITS:
        unsafe_records = sum(component.unsafe_count for component in result[split])
        if unsafe_records < MINIMUMS[split]["unsafe_records"]:
            raise RuntimeError(
                f"{split} has only {unsafe_records} Unsafe records after "
                "component allocation"
            )
    return result, best[0]


def _safe_component_partition(
    components: Sequence[Component],
    existing: dict[str, list[Component]],
    total_record_count: int,
    seed: int,
    iterations: int,
) -> tuple[dict[str, list[Component]], float]:
    count = len(components)
    targets = {
        split: round(TARGET_RATIOS[split] * count) for split in SPLITS
    }
    targets["confirm"] = max(
        MINIMUMS["confirm"]["safe_only_components"], targets["confirm"]
    )
    while sum(targets.values()) > count:
        split = max(
            ("fit", "select"),
            key=lambda value: targets[value] - TARGET_RATIOS[value] * count,
        )
        targets[split] -= 1
    while sum(targets.values()) < count:
        split = max(
            SPLITS,
            key=lambda value: TARGET_RATIOS[value] * count - targets[value],
        )
        targets[split] += 1

    rng = np.random.default_rng(seed)
    totals = np.asarray([component.record_count for component in components])
    existing_records = {
        split: sum(component.record_count for component in existing[split])
        for split in SPLITS
    }
    boundaries = np.cumsum([targets[split] for split in SPLITS])
    best: tuple[float, np.ndarray] | None = None
    for _ in range(iterations):
        order = rng.permutation(count)
        partitions = np.split(order, boundaries[:-1])
        score = 0.0
        for split, partition in zip(SPLITS, partitions, strict=True):
            records = existing_records[split] + int(totals[partition].sum())
            target = TARGET_RATIOS[split] * total_record_count
            score += ((records - target) / max(1, total_record_count)) ** 2
        if best is None or score < best[0]:
            best = (score, order.copy())
    if best is None:
        raise RuntimeError("No safe-only component partition was generated")
    partitions = np.split(best[1], boundaries[:-1])
    result = {
        split: [components[int(index)] for index in partition]
        for split, partition in zip(SPLITS, partitions, strict=True)
    }
    return result, best[0]


def allocate_components(
    components: Sequence[Component],
    seed: int,
    search_iterations: int,
) -> tuple[dict[str, list[Component]], dict[str, Any]]:
    unsafe_bearing = [
        component for component in components if component.unsafe_count > 0
    ]
    safe_only = [
        component for component in components if component.unsafe_count == 0
    ]
    unsafe_targets = _unsafe_component_targets(len(unsafe_bearing))
    unsafe_result, unsafe_score = _fixed_count_partition(
        unsafe_bearing,
        unsafe_targets,
        seed,
        search_iterations,
    )
    safe_result, safe_score = _safe_component_partition(
        safe_only,
        unsafe_result,
        sum(component.record_count for component in components),
        seed + 1,
        max(1000, search_iterations // 2),
    )
    assignment = {
        split: sorted(
            unsafe_result[split] + safe_result[split],
            key=lambda value: value.component_id,
        )
        for split in SPLITS
    }
    diagnostics = {
        "seed": seed,
        "search_iterations": search_iterations,
        "unsafe_component_targets": unsafe_targets,
        "unsafe_partition_score": unsafe_score,
        "safe_partition_score": safe_score,
    }
    return assignment, diagnostics


def split_counts(
    assignment: dict[str, list[Component]],
    records: Sequence[Record],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_records = sum(
        component.record_count
        for components in assignment.values()
        for component in components
    )
    for split, components in assignment.items():
        indices = [
            index for component in components for index in component.record_indices
        ]
        labels = Counter(records[index].label for index in indices)
        dimensions = Counter(records[index].dimension for index in indices)
        data_types = Counter(records[index].data_type for index in indices)
        target_models = Counter(records[index].target_model for index in indices)
        image_buckets = Counter(
            image_count_bucket(len(records[index].image_references))
            for index in indices
        )
        compositions = Counter(component.composition for component in components)
        result[split] = {
            "record_count": len(indices),
            "record_fraction": len(indices) / max(1, total_records),
            "component_count": len(components),
            "label_counts": dict(sorted(labels.items())),
            "unsafe_bearing_component_count": sum(
                component.unsafe_count > 0 for component in components
            ),
            "safe_only_component_count": compositions["safe-only"],
            "component_composition_counts": dict(sorted(compositions.items())),
            "dimension_counts": dict(sorted(dimensions.items())),
            "data_type_counts": dict(sorted(data_types.items())),
            "target_model_counts": dict(sorted(target_models.items())),
            "image_count_bucket_counts": dict(sorted(image_buckets.items())),
        }
    return result


def allocation_gate(counts: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for split in SPLITS:
        values = counts[split]
        checks[f"{split}_record_fraction"] = {
            "fit": 0.55 <= values["record_fraction"] <= 0.65,
            "select": 0.15 <= values["record_fraction"] <= 0.20,
            "confirm": 0.20 <= values["record_fraction"] <= 0.30,
        }[split]
        checks[f"{split}_unsafe_bearing_components"] = (
            values["unsafe_bearing_component_count"]
            >= MINIMUMS[split]["unsafe_bearing_components"]
        )
        checks[f"{split}_unsafe_records"] = (
            values["label_counts"].get("unsafe", 0)
            >= MINIMUMS[split]["unsafe_records"]
        )
    checks["confirm_safe_only_components"] = (
        counts["confirm"]["safe_only_component_count"]
        >= MINIMUMS["confirm"]["safe_only_components"]
    )
    return {"checks": checks, "passed": all(checks.values())}


def _component_sets(
    assignment: dict[str, list[Component]],
) -> dict[str, set[str]]:
    return {
        split: {component.component_id for component in components}
        for split, components in assignment.items()
    }


def exact_leakage_audit(
    assignment: dict[str, list[Component]],
    records: Sequence[Record],
    fingerprints: dict[str, ImageFingerprint],
    engineering_lines: set[int],
) -> dict[str, Any]:
    component_sets = _component_sets(assignment)
    index_to_split: dict[int, str] = {}
    for split, components in assignment.items():
        for component in components:
            for index in component.record_indices:
                index_to_split[index] = split
    audit_sets = (*SPLITS, "engineering", "legacy_val", "legacy_test")
    pixels: dict[str, set[str]] = {split: set() for split in audit_sets}
    conversations: dict[str, set[str]] = {split: set() for split in audit_sets}
    for index, split in index_to_split.items():
        record = records[index]
        conversations[split].add(record.conversation_sha256)
        pixels[split].update(
            fingerprints[reference].pixel_sha256
            for reference in record.image_references
        )
    for record in records:
        tags: list[str] = []
        if record.line_number in engineering_lines:
            tags.append("engineering")
        if record.source_split == "val":
            tags.append("legacy_val")
        elif record.source_split == "test":
            tags.append("legacy_test")
        for tag in tags:
            conversations[tag].add(record.conversation_sha256)
            pixels[tag].update(
                fingerprints[reference].pixel_sha256
                for reference in record.image_references
                if reference in fingerprints
            )
    pairwise: dict[str, Any] = {}
    required_pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(SPLITS):
        required_pairs.extend((left, right) for right in SPLITS[left_index + 1 :])
        required_pairs.extend(
            (left, right)
            for right in ("engineering", "legacy_val", "legacy_test")
        )
    for left, right in required_pairs:
        key = f"{left}__{right}"
        pairwise[key] = {
            "shared_component_count": (
                len(component_sets[left] & component_sets[right])
                if right in component_sets
                else 0
            ),
            "shared_pixel_digest_count": len(pixels[left] & pixels[right]),
            "shared_conversation_digest_count": len(
                conversations[left] & conversations[right]
            ),
        }
    passed = all(
        all(value == 0 for value in audit.values())
        for audit in pairwise.values()
    )
    return {"pairwise": pairwise, "passed": passed}


def _normalized_thumbnail(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((64, 64), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (64, 64), "white")
        offset = ((64 - image.width) // 2, (64 - image.height) // 2)
        canvas.paste(image, offset)
        return np.asarray(canvas, dtype=np.float32)


def perceptual_candidate_audit(
    *,
    assignment: dict[str, list[Component]],
    records: Sequence[Record],
    fingerprints: dict[str, ImageFingerprint],
    mmds_root: Path,
    engineering_lines: set[int],
    dsu: DisjointSet,
    decisions_path: Path | None,
    phash_radius: int = 4,
    mean_absolute_difference_limit: float = 8.0,
) -> dict[str, Any]:
    index_to_split: dict[int, str] = {}
    for split, components in assignment.items():
        for component in components:
            for index in component.record_indices:
                index_to_split[index] = split

    pixel_membership: dict[str, set[str]] = defaultdict(set)
    pixel_component_roots: dict[str, set[int]] = defaultdict(set)
    representative: dict[str, ImageFingerprint] = {}
    for record in records:
        tags: list[str] = []
        if record.index in index_to_split:
            tags.append(index_to_split[record.index])
        if record.line_number in engineering_lines:
            tags.append("engineering")
        if record.source_split == "val":
            tags.append("legacy_val")
        elif record.source_split == "test":
            tags.append("legacy_test")
        if not tags:
            tags.append("excluded")
        for reference in record.image_references:
            fingerprint = fingerprints.get(reference)
            if fingerprint is None:
                continue
            representative.setdefault(fingerprint.pixel_sha256, fingerprint)
            pixel_membership[fingerprint.pixel_sha256].update(tags)
            pixel_component_roots[fingerprint.pixel_sha256].add(
                dsu.find(record.index)
            )

    decisions = load_near_duplicate_decisions(decisions_path)

    tree = BKTree()
    candidates: list[dict[str, Any]] = []
    thumbnails: dict[str, np.ndarray] = {}
    for pixel_sha256 in sorted(representative):
        current = representative[pixel_sha256]
        for phash_distance, other_sha256 in tree.query(
            current.phash, phash_radius
        ):
            memberships = (
                pixel_membership[pixel_sha256],
                pixel_membership[other_sha256],
            )
            other = representative[other_sha256]
            current_ratio = current.width / max(1, current.height)
            other_ratio = other.width / max(1, other.height)
            ratio_delta = abs(math.log(current_ratio / other_ratio))
            if ratio_delta > 0.03:
                continue
            for fingerprint in (current, other):
                if fingerprint.pixel_sha256 not in thumbnails:
                    thumbnails[fingerprint.pixel_sha256] = _normalized_thumbnail(
                        mmds_root / fingerprint.reference
                    )
            difference = float(
                np.mean(
                    np.abs(
                        thumbnails[pixel_sha256] - thumbnails[other_sha256]
                    )
                )
            )
            if difference > mean_absolute_difference_limit:
                continue
            pair = sorted((pixel_sha256, other_sha256))
            pair_key = f"{pair[0]}__{pair[1]}"
            same_component = bool(
                pixel_component_roots[pixel_sha256]
                & pixel_component_roots[other_sha256]
            )
            candidates.append(
                {
                    "pair_key": pair_key,
                    "left": {
                        "pixel_sha256": pixel_sha256,
                        "reference": current.reference,
                        "memberships": sorted(pixel_membership[pixel_sha256]),
                        "size": [current.width, current.height],
                    },
                    "right": {
                        "pixel_sha256": other_sha256,
                        "reference": other.reference,
                        "memberships": sorted(pixel_membership[other_sha256]),
                        "size": [other.width, other.height],
                    },
                    "phash_hamming_distance": phash_distance,
                    "normalized_thumbnail_mean_absolute_difference": difference,
                    "decision": decisions.get(pair_key, "unreviewed"),
                    "already_in_same_component": same_component,
                }
            )
        tree.add(current.phash, pixel_sha256)

    candidates.sort(key=lambda value: value["pair_key"])
    unresolved = [
        value for value in candidates if value["decision"] == "unreviewed"
    ]
    confirmed_same = [
        value for value in candidates if value["decision"] == "same"
    ]
    confirmed_same_unmerged = [
        value
        for value in confirmed_same
        if not value["already_in_same_component"]
    ]
    unused_decisions = sorted(
        set(decisions) - {value["pair_key"] for value in candidates}
    )
    return {
        "schema": "softfusion-perceptual-candidate-audit-v1",
        "method": {
            "hash": "64-bit DCT perceptual hash",
            "phash_hamming_radius": phash_radius,
            "aspect_log_ratio_limit": 0.03,
            "normalized_thumbnail_mean_absolute_difference_limit": (
                mean_absolute_difference_limit
            ),
            "policy": (
                "Candidates are never merged automatically. A human decision "
                "must mark each candidate as same or different."
            ),
        },
        "candidate_count": len(candidates),
        "unresolved_count": len(unresolved),
        "confirmed_same_count": len(confirmed_same),
        "confirmed_same_unmerged_count": len(confirmed_same_unmerged),
        "confirmed_different_count": sum(
            value["decision"] == "different" for value in candidates
        ),
        "unused_decision_keys": unused_decisions,
        "candidates": candidates,
        "passed": not unresolved and not confirmed_same_unmerged,
    }


def _macro_f1_from_counts(
    tp: np.ndarray | float,
    tn: np.ndarray | float,
    fp: np.ndarray | float,
    fn: np.ndarray | float,
) -> np.ndarray:
    tp_array = np.asarray(tp, dtype=float)
    tn_array = np.asarray(tn, dtype=float)
    fp_array = np.asarray(fp, dtype=float)
    fn_array = np.asarray(fn, dtype=float)
    unsafe_f1 = np.divide(
        2 * tp_array,
        2 * tp_array + fp_array + fn_array,
        out=np.zeros_like(tp_array),
        where=(2 * tp_array + fp_array + fn_array) > 0,
    )
    safe_f1 = np.divide(
        2 * tn_array,
        2 * tn_array + fp_array + fn_array,
        out=np.zeros_like(tn_array),
        where=(2 * tn_array + fp_array + fn_array) > 0,
    )
    return (unsafe_f1 + safe_f1) / 2


def legacy_majority_rates(judgment_paths: Sequence[Path]) -> dict[str, Any]:
    by_model: list[dict[str, dict[str, Any]]] = []
    source_hashes: dict[str, str] = {}
    for path in judgment_paths:
        values: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    values[record["sample_id"]] = record
        by_model.append(values)
        source_hashes[str(path)] = sha256_file(path)
    sample_ids = set(by_model[0])
    if any(set(values) != sample_ids for values in by_model[1:]):
        raise ValueError("Legacy judgment files do not contain identical samples")
    tp = tn = fp = fn = 0
    for sample_id in sorted(sample_ids):
        rows = [values[sample_id] for values in by_model]
        expected = rows[0]["expected_label"]
        votes = sum(row["decision"]["label"] == "unsafe" for row in rows)
        predicted = "unsafe" if votes >= math.ceil(len(rows) / 2) else "safe"
        if expected == "unsafe" and predicted == "unsafe":
            tp += 1
        elif expected == "safe" and predicted == "safe":
            tn += 1
        elif expected == "safe":
            fp += 1
        else:
            fn += 1
    return {
        "sample_count": len(sample_ids),
        "source_hashes": source_hashes,
        "tie_rule": "unsafe when at least half of the four judges vote unsafe",
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "unsafe_recall": tp / (tp + fn),
        "safe_specificity": tn / (tn + fp),
        "macro_f1": float(_macro_f1_from_counts(tp, tn, fp, fn)),
    }


def _target_recall_for_delta(
    unsafe_count: int,
    safe_count: int,
    baseline_recall: float,
    specificity: float,
    delta: float,
) -> float:
    baseline = float(
        _macro_f1_from_counts(
            unsafe_count * baseline_recall,
            safe_count * specificity,
            safe_count * (1 - specificity),
            unsafe_count * (1 - baseline_recall),
        )
    )
    target = baseline + delta
    low, high = baseline_recall, 1.0
    achievable = float(
        _macro_f1_from_counts(
            unsafe_count,
            safe_count * specificity,
            safe_count * (1 - specificity),
            0,
        )
    )
    if achievable < target:
        return 1.0
    for _ in range(60):
        middle = (low + high) / 2
        value = float(
            _macro_f1_from_counts(
                unsafe_count * middle,
                safe_count * specificity,
                safe_count * (1 - specificity),
                unsafe_count * (1 - middle),
            )
        )
        if value < target:
            low = middle
        else:
            high = middle
    return high


def power_precision_audit(
    confirm_components: Sequence[Component],
    records: Sequence[Record],
    legacy_rates: dict[str, Any],
    seed: int,
    simulations: int = 250,
    bootstrap_replicates: int = 400,
) -> dict[str, Any]:
    component_indices = [component.record_indices for component in confirm_components]
    labels = np.asarray(
        [
            1 if records[index].label == "unsafe" else 0
            for indices in component_indices
            for index in indices
        ],
        dtype=np.int8,
    )
    component_for_record = np.asarray(
        [
            component_index
            for component_index, indices in enumerate(component_indices)
            for _ in indices
        ],
        dtype=np.int32,
    )
    component_count = len(component_indices)
    unsafe_count = int(labels.sum())
    safe_count = len(labels) - unsafe_count
    recall = float(legacy_rates["unsafe_recall"])
    specificity = float(legacy_rates["safe_specificity"])
    rng = np.random.default_rng(seed)
    bootstrap_weights = rng.multinomial(
        component_count,
        np.full(component_count, 1 / component_count),
        size=bootstrap_replicates,
    )

    def scenario(delta: float, shared_uniform_probability: float) -> dict[str, Any]:
        target_recall = _target_recall_for_delta(
            unsafe_count,
            safe_count,
            recall,
            specificity,
            delta,
        )
        positive_results = 0
        widths: list[float] = []
        observed_deltas: list[float] = []
        for _ in range(simulations):
            baseline_u = rng.random(len(labels))
            independent_u = rng.random(len(labels))
            use_shared = rng.random(len(labels)) < shared_uniform_probability
            candidate_u = np.where(use_shared, baseline_u, independent_u)
            baseline = np.where(
                labels == 1,
                baseline_u < recall,
                baseline_u >= specificity,
            )
            candidate = np.where(
                labels == 1,
                candidate_u < target_recall,
                candidate_u >= specificity,
            )

            component_counts = np.zeros((8, component_count), dtype=np.int32)
            for prediction, offset in ((baseline, 0), (candidate, 4)):
                masks = (
                    (labels == 1) & prediction,
                    (labels == 0) & ~prediction,
                    (labels == 0) & prediction,
                    (labels == 1) & ~prediction,
                )
                for position, mask in enumerate(masks):
                    component_counts[offset + position] = np.bincount(
                        component_for_record[mask],
                        minlength=component_count,
                    )
            boot = bootstrap_weights @ component_counts.T
            baseline_f1 = _macro_f1_from_counts(
                boot[:, 0], boot[:, 1], boot[:, 2], boot[:, 3]
            )
            candidate_f1 = _macro_f1_from_counts(
                boot[:, 4], boot[:, 5], boot[:, 6], boot[:, 7]
            )
            differences = candidate_f1 - baseline_f1
            lower, upper = np.quantile(differences, [0.025, 0.975])
            positive_results += int(lower > 0)
            widths.append(float(upper - lower))
            full = component_counts.sum(axis=1)
            observed_deltas.append(
                float(
                    _macro_f1_from_counts(*full[4:8])
                    - _macro_f1_from_counts(*full[0:4])
                )
            )
        return {
            "assumed_macro_f1_delta": delta,
            "shared_error_uniform_probability": shared_uniform_probability,
            "target_unsafe_recall": target_recall,
            "estimated_power": positive_results / simulations,
            "median_observed_delta": float(np.median(observed_deltas)),
            "median_cluster_bootstrap_ci_width": float(np.median(widths)),
            "simulation_count": simulations,
            "cluster_bootstrap_replicates": bootstrap_replicates,
        }

    target_scenarios = [
        scenario(0.05, probability) for probability in (0.25, 0.50, 0.75)
    ]
    mde_grid: list[dict[str, Any]] = []
    for delta in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08):
        result = scenario(delta, 0.50)
        mde_grid.append(
            {
                "delta": delta,
                "estimated_power": result["estimated_power"],
                "median_ci_width": result["median_cluster_bootstrap_ci_width"],
            }
        )
    detectable = next(
        (
            value["delta"]
            for value in mde_grid
            if value["estimated_power"] >= 0.80
        ),
        None,
    )
    minimum_power = min(
        value["estimated_power"] for value in target_scenarios
    )
    return {
        "schema": "softfusion-power-precision-audit-v1",
        "analysis_scope": (
            "design-stage Monte Carlo using confirm labels, image components, "
            "and the frozen legacy majority error profile; no new-method outputs"
        ),
        "confirm_record_count": len(labels),
        "confirm_component_count": component_count,
        "confirm_label_counts": {"safe": safe_count, "unsafe": unsafe_count},
        "legacy_majority": legacy_rates,
        "target_delta_scenarios": target_scenarios,
        "minimum_power_at_delta_0_05": minimum_power,
        "power_target_met": minimum_power >= 0.80,
        "mde_grid_at_shared_probability_0_50": mde_grid,
        "minimum_detectable_delta_at_80_percent_power": detectable,
        "design_gate_passed": minimum_power >= 0.80 or detectable is not None,
        "seed": seed,
        "limitations": [
            "This is a design-stage precision calculation, not an observed method result.",
            "The simulation transfers legacy class-conditional error rates but does not assume that the new method has already achieved the target effect.",
            "The confirmatory analysis will use image-component bootstrap intervals on frozen predictions.",
        ],
    }


def coarse_stratum(component: Component) -> str:
    return f"{component.composition}|{component.dominant_dimension}"


def inclusion_probabilities(
    components: Sequence[Component],
    assignment: dict[str, list[Component]],
) -> dict[str, dict[str, float]]:
    total = Counter(coarse_stratum(component) for component in components)
    result: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        selected = Counter(
            coarse_stratum(component) for component in assignment[split]
        )
        result[split] = {
            stratum: selected[stratum] / count
            for stratum, count in sorted(total.items())
        }
    return result


def manifest_entry(
    record: Record,
    component: Component,
    split: str,
    probability: float,
    include_label: bool,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "sample_id": record.sample_id,
        "dataset": "MMDS",
        "source_split": record.source_split,
        "softfusion_split": split,
        "jsonl_line": record.line_number,
        "raw_id": record.raw_id,
        "component_id": component.component_id,
        "component_composition": component.composition,
        "image_references": list(record.image_references),
        "conversation_sha256": record.conversation_sha256,
        "data_type": record.data_type,
        "target_model": record.target_model,
        "image_count_bucket": image_count_bucket(len(record.image_references)),
        "coarse_stratum": coarse_stratum(component),
        "component_inclusion_probability": probability,
    }
    if include_label:
        value["assistant_label"] = record.label
        value["assistant_dimension"] = record.dimension
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.mmds_jsonl.resolve()
    mmds_root = metadata_path.parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(metadata_path)
    fingerprints, missing_images = fingerprint_images(records, mmds_root)
    decisions_path = (
        args.near_duplicate_decisions.resolve()
        if args.near_duplicate_decisions is not None
        else None
    )
    near_duplicate_decisions = load_near_duplicate_decisions(decisions_path)
    dsu = build_global_components(
        records,
        fingerprints,
        near_duplicate_decisions=near_duplicate_decisions,
    )
    engineering_lines = manifest_line_numbers(args.engineering_manifest.resolve())
    exclusions = exclusion_reasons(
        records,
        dsu,
        fingerprints,
        engineering_lines,
    )
    components, _record_to_component = build_eligible_components(
        records, dsu, exclusions
    )
    assignment, allocation_diagnostics = allocate_components(
        components,
        args.seed,
        args.search_iterations,
    )
    counts = split_counts(assignment, records)
    allocation = allocation_gate(counts)
    exact_leakage = exact_leakage_audit(
        assignment, records, fingerprints, engineering_lines
    )
    perceptual = perceptual_candidate_audit(
        assignment=assignment,
        records=records,
        fingerprints=fingerprints,
        mmds_root=mmds_root,
        engineering_lines=engineering_lines,
        dsu=dsu,
        decisions_path=decisions_path,
    )
    probabilities = inclusion_probabilities(components, assignment)

    record_by_sample_id = {record.sample_id: record for record in records}
    entries: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        values: list[dict[str, Any]] = []
        for component in assignment[split]:
            probability = probabilities[split][coarse_stratum(component)]
            for index in component.record_indices:
                values.append(
                    manifest_entry(
                        records[index],
                        component,
                        split,
                        probability,
                        include_label=split != "confirm",
                    )
                )
        entries[split] = sorted(values, key=lambda value: value["jsonl_line"])

    fit_path = output_dir / "fit.jsonl"
    select_path = output_dir / "select.jsonl"
    confirm_unlabeled_path = output_dir / "confirm-unlabeled.jsonl"
    confirm_labels_path = output_dir / "confirm-labels.jsonl"
    write_jsonl(fit_path, entries["fit"])
    write_jsonl(select_path, entries["select"])
    write_jsonl(confirm_unlabeled_path, entries["confirm"])
    confirm_labels = [
        {
            "sample_id": value["sample_id"],
            "component_id": value["component_id"],
            "assistant_label": record_by_sample_id[value["sample_id"]].label,
            "assistant_dimension": record_by_sample_id[
                value["sample_id"]
            ].dimension,
        }
        for value in entries["confirm"]
    ]
    write_jsonl(confirm_labels_path, confirm_labels)

    exclusion_rows = [
        {
            "sample_id": record.sample_id,
            "jsonl_line": record.line_number,
            "source_split": record.source_split,
            "reasons": exclusions[record.index],
        }
        for record in records
        if record.index in exclusions
    ]
    eligibility = {
        "schema": "softfusion-eligibility-audit-v1",
        "source_record_count": len(records),
        "eligible_record_count": sum(
            component.record_count for component in components
        ),
        "eligible_component_count": len(components),
        "eligible_label_counts": dict(
            Counter(
                record.label
                for record in records
                if record.index not in exclusions
            )
        ),
        "eligible_component_composition_counts": dict(
            Counter(component.composition for component in components)
        ),
        "exclusion_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for reasons in exclusions.values()
                    for reason in reasons
                ).items()
            )
        ),
        "excluded_records": exclusion_rows,
        "missing_source_images": missing_images,
    }
    write_json(output_dir / "eligibility_audit.json", eligibility)

    component_audit = {
        "schema": "softfusion-component-audit-v1",
        "eligible_component_count": len(components),
        "eligible_record_count": sum(
            component.record_count for component in components
        ),
        "component_composition_counts": dict(
            sorted(Counter(c.composition for c in components).items())
        ),
        "unsafe_bearing_component_count": sum(
            component.unsafe_count > 0 for component in components
        ),
        "split_counts": counts,
        "allocation": allocation,
        "allocation_diagnostics": allocation_diagnostics,
        "components": [
            {
                "component_id": component.component_id,
                "record_count": component.record_count,
                "safe_count": component.safe_count,
                "unsafe_count": component.unsafe_count,
                "composition": component.composition,
                "dominant_dimension": component.dominant_dimension,
                "sample_ids": [
                    records[index].sample_id
                    for index in component.record_indices
                ],
            }
            for component in components
        ],
    }
    write_json(output_dir / "component_audit.json", component_audit)

    write_json(output_dir / "perceptual_candidate_audit.json", perceptual)
    leakage_passed = exact_leakage["passed"] and perceptual["passed"]
    leakage = {
        "schema": "softfusion-leakage-audit-v1",
        "exact": exact_leakage,
        "perceptual_near_duplicate_status": (
            "passed" if perceptual["passed"] else "review-required"
        ),
        "perceptual_near_duplicate_candidate_count": perceptual[
            "candidate_count"
        ],
        "perceptual_unresolved_count": perceptual["unresolved_count"],
        "perceptual_confirmed_same_count": perceptual[
            "confirmed_same_count"
        ],
        "perceptual_confirmed_same_unmerged_count": perceptual[
            "confirmed_same_unmerged_count"
        ],
        "passed": leakage_passed,
        "note": (
            "A confirmed-same pair requires component merging and a fresh split; "
            "a confirmed-different pair remains separate."
        ),
    }
    write_json(output_dir / "leakage_audit.json", leakage)

    judgment_paths = [
        args.legacy_judgment_root.resolve() / model / "judgments.jsonl"
        for model in ("qwen", "safework", "internvl", "minicpm")
    ]
    legacy_rates = legacy_majority_rates(judgment_paths)
    power = power_precision_audit(
        assignment["confirm"],
        records,
        legacy_rates,
        args.seed + 2,
        simulations=args.power_simulations,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    write_json(output_dir / "power_precision_audit.json", power)

    manifest = {
        "schema": SCHEMA_VERSION,
        "status": (
            "frozen"
            if (
                allocation["passed"]
                and leakage_passed
                and power["design_gate_passed"]
            )
            else "candidate"
        ),
        "source": {
            "mmds_jsonl": str(metadata_path),
            "mmds_jsonl_sha256": sha256_file(metadata_path),
            "engineering_manifest": str(args.engineering_manifest.resolve()),
            "engineering_manifest_sha256": sha256_file(
                args.engineering_manifest.resolve()
            ),
            "near_duplicate_decisions": (
                str(decisions_path) if decisions_path is not None else None
            ),
            "near_duplicate_decisions_sha256": (
                sha256_file(decisions_path)
                if decisions_path is not None and decisions_path.is_file()
                else None
            ),
        },
        "seed": args.seed,
        "component_identity": (
            "connected components over decoded RGB pixel SHA-256 and exact "
            "canonical conversation SHA-256"
        ),
        "split_ratios": TARGET_RATIOS,
        "minimums": MINIMUMS,
        "split_counts": counts,
        "allocation_gate": allocation,
        "exact_leakage_gate": exact_leakage,
        "perceptual_leakage_gate": {
            "passed": perceptual["passed"],
            "candidate_count": perceptual["candidate_count"],
            "unresolved_count": perceptual["unresolved_count"],
            "confirmed_same_count": perceptual["confirmed_same_count"],
            "confirmed_same_unmerged_count": perceptual[
                "confirmed_same_unmerged_count"
            ],
        },
        "power_precision_gate": {
            "passed": power["design_gate_passed"],
            "power_at_delta_0_05": power["minimum_power_at_delta_0_05"],
            "minimum_detectable_delta": power[
                "minimum_detectable_delta_at_80_percent_power"
            ],
        },
        "output_files": {},
    }
    tracked_paths = [
        fit_path,
        select_path,
        confirm_unlabeled_path,
        confirm_labels_path,
        output_dir / "eligibility_audit.json",
        output_dir / "component_audit.json",
        output_dir / "perceptual_candidate_audit.json",
        output_dir / "leakage_audit.json",
        output_dir / "power_precision_audit.json",
    ]
    manifest["output_files"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in tracked_paths
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build label-stratified, image-component-isolated development and "
            "confirmatory splits for the v15 soft-fusion experiment."
        )
    )
    parser.add_argument("--mmds-jsonl", type=Path, required=True)
    parser.add_argument("--engineering-manifest", type=Path, required=True)
    parser.add_argument("--legacy-judgment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-duplicate-decisions", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--search-iterations", type=int, default=20_000)
    parser.add_argument("--power-simulations", type=int, default=250)
    parser.add_argument("--bootstrap-replicates", type=int, default=400)
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
