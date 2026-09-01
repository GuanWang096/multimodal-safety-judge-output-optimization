from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "freeze_sf0.py"
SPEC = importlib.util.spec_from_file_location("freeze_sf0", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_confirm_unlabeled_rejects_target_fields(tmp_path: Path) -> None:
    path = tmp_path / "confirm.jsonl"
    path.write_text(
        json.dumps({"sample_id": "a"})
        + "\n"
        + json.dumps({"sample_id": "b", "assistant_label": "unsafe"})
        + "\n",
        encoding="utf-8",
    )
    count, violations = MODULE.check_confirm_unlabeled(path)
    assert count == 2
    assert violations == ["line_2:assistant_label"]


def test_blind_sheet_rejects_official_answer_header(tmp_path: Path) -> None:
    path = tmp_path / "blind.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["blind_item_id", "correct_option", "binary_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "blind_item_id": "a",
                "correct_option": "",
                "binary_label": "",
            }
        )
    count, forbidden = MODULE.check_blind_sheet(path)
    assert count == 1
    assert forbidden == ["correct_option"]


def test_real_split_manifest_uses_record_count() -> None:
    split_path = (
        MODULE.V15_ROOT
        / "results"
        / "mmds"
        / "data"
        / "split_manifest.json"
    )
    split = MODULE.load_json(split_path)
    assert split["split_counts"]["confirm"]["record_count"] > 0
