from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from prepare_softfusion_splits import (
    Component,
    Record,
    allocate_components,
    allocation_gate,
    build_eligible_components,
    build_global_components,
    exact_leakage_audit,
    exclusion_reasons,
    fingerprint_images,
    manifest_entry,
    perceptual_candidate_audit,
    split_counts,
)


def _record(
    index: int,
    *,
    split: str = "train",
    label: str = "safe",
    images: tuple[str, ...] = (),
    conversation: str | None = None,
) -> Record:
    return Record(
        index=index,
        line_number=index + 1,
        sample_id=f"mmds:{split}:{index + 1}:{index}",
        raw_id=str(index),
        source_split=split,
        label=label,
        dimension="unsafe" if label == "unsafe" else "safe",
        data_type="default",
        target_model="fixture",
        image_references=images,
        conversation_sha256=conversation or f"conversation-{index}",
    )


def test_decoded_pixel_identity_connects_different_files(tmp_path: Path) -> None:
    root = tmp_path / "MMDS"
    image_root = root / "images"
    image_root.mkdir(parents=True)
    image = Image.new("RGB", (32, 24), (10, 20, 30))
    image.save(image_root / "a.png")
    image.save(image_root / "b.bmp")
    records = [
        _record(0, images=("images/a.png",)),
        _record(1, images=("images/b.bmp",)),
    ]
    fingerprints, missing = fingerprint_images(records, root)
    assert missing == []
    assert (
        fingerprints["images/a.png"].pixel_sha256
        == fingerprints["images/b.bmp"].pixel_sha256
    )
    dsu = build_global_components(records, fingerprints)
    assert dsu.find(0) == dsu.find(1)


def test_engineering_overlap_excludes_whole_component(tmp_path: Path) -> None:
    root = tmp_path / "MMDS"
    image_root = root / "images"
    image_root.mkdir(parents=True)
    Image.new("RGB", (20, 20), "red").save(image_root / "shared.png")
    records = [
        _record(0, images=("images/shared.png",)),
        _record(1, images=("images/shared.png",)),
    ]
    fingerprints, _ = fingerprint_images(records, root)
    dsu = build_global_components(records, fingerprints)
    exclusions = exclusion_reasons(records, dsu, fingerprints, {1})
    assert "engineering_component_overlap" in exclusions[0]
    assert "engineering_component_overlap" in exclusions[1]


def test_component_allocation_meets_all_minima() -> None:
    components: list[Component] = []
    records: list[Record] = []
    index = 0
    for component_index in range(170):
        component_records: list[int] = []
        for _ in range(3):
            records.append(_record(index, label="unsafe"))
            component_records.append(index)
            index += 1
        components.append(
            Component(
                component_id=f"unsafe-{component_index:03d}",
                record_indices=component_records,
                record_count=3,
                safe_count=0,
                unsafe_count=3,
                composition="unsafe-only",
                dominant_dimension="unsafe",
            )
        )
    for component_index in range(1017):
        records.append(_record(index, label="safe"))
        components.append(
            Component(
                component_id=f"safe-{component_index:04d}",
                record_indices=[index],
                record_count=1,
                safe_count=1,
                unsafe_count=0,
                composition="safe-only",
                dominant_dimension="safe",
            )
        )
        index += 1
    assignment, _ = allocate_components(
        components, seed=20260729, search_iterations=1000
    )
    counts = split_counts(assignment, records)
    assert allocation_gate(counts)["passed"] is True


def test_confirm_manifest_entry_has_no_ground_truth() -> None:
    record = _record(0, label="unsafe", images=("images/a.png",))
    component = Component(
        component_id="cmp-test",
        record_indices=[0],
        record_count=1,
        safe_count=0,
        unsafe_count=1,
        composition="unsafe-only",
        dominant_dimension="unsafe",
    )
    entry = manifest_entry(
        record,
        component,
        "confirm",
        probability=0.25,
        include_label=False,
    )
    assert "assistant_label" not in entry
    assert "assistant_dimension" not in entry
    assert entry["component_id"] == "cmp-test"


def test_exact_leakage_audit_detects_no_cross_split_after_components(
    tmp_path: Path,
) -> None:
    root = tmp_path / "MMDS"
    image_root = root / "images"
    image_root.mkdir(parents=True)
    for name, color in (("a.png", "red"), ("b.png", "blue"), ("c.png", "green")):
        Image.new("RGB", (20, 20), color).save(image_root / name)
    records = [
        _record(0, images=("images/a.png",)),
        _record(1, images=("images/b.png",)),
        _record(2, split="test", images=("images/c.png",)),
    ]
    fingerprints, _ = fingerprint_images(records, root)
    assignment = {
        "fit": [
            Component("a", [0], 1, 1, 0, "safe-only", "safe")
        ],
        "select": [
            Component("b", [1], 1, 1, 0, "safe-only", "safe")
        ],
        "confirm": [],
    }
    audit = exact_leakage_audit(
        assignment, records, fingerprints, engineering_lines=set()
    )
    assert audit["passed"] is True


def test_perceptual_candidates_are_not_auto_merged(tmp_path: Path) -> None:
    root = tmp_path / "MMDS"
    image_root = root / "images"
    image_root.mkdir(parents=True)
    image = Image.new("RGB", (64, 64), (100, 120, 140))
    image.save(image_root / "a.png")
    changed = image.copy()
    changed.putpixel((32, 32), (102, 121, 139))
    changed.save(image_root / "b.png")
    records = [
        _record(0, images=("images/a.png",)),
        _record(1, images=("images/b.png",)),
    ]
    fingerprints, _ = fingerprint_images(records, root)
    assignment = {
        "fit": [
            Component("a", [0], 1, 1, 0, "safe-only", "safe")
        ],
        "select": [
            Component("b", [1], 1, 1, 0, "safe-only", "safe")
        ],
        "confirm": [],
    }
    audit = perceptual_candidate_audit(
        assignment=assignment,
        records=records,
        fingerprints=fingerprints,
        mmds_root=root,
        engineering_lines=set(),
        dsu=build_global_components(records, fingerprints),
        decisions_path=None,
    )
    assert audit["candidate_count"] == 1
    assert audit["unresolved_count"] == 1
    assert audit["passed"] is False


def test_build_eligible_components_keeps_mixed_component_whole() -> None:
    records = [
        _record(0, label="safe", conversation="same"),
        _record(1, label="unsafe", conversation="same"),
    ]
    dsu = build_global_components(records, {})
    components, mapping = build_eligible_components(records, dsu, {})
    assert len(components) == 1
    assert components[0].composition == "mixed"
    assert mapping[0] == mapping[1]
