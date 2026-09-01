from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "audit_model_data_overlap.py"
SPEC = importlib.util.spec_from_file_location("audit_model_data_overlap", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_term_matching_is_case_insensitive_and_token_bounded() -> None:
    matches = MODULE.term_line_matches(
        "SIUO score\nsiuo benchmark\nNOTSIUOX\nMMDS",
        ["SIUO", "MMDS"],
    )
    assert matches["SIUO"] == [1, 2]
    assert matches["MMDS"] == [4]


def test_dataset_status_distinguishes_absence_from_proven_independence() -> None:
    assert (
        MODULE._dataset_status("mmds", False, set(), set())
        == "no_direct_overlap_disclosed"
    )
    assert (
        MODULE._dataset_status("siuo", True, set(), {"siuo"})
        == "declared_evaluation_exposure"
    )
    assert (
        MODULE._dataset_status("siuo", True, set(), set())
        == "named_in_model_card_but_unclassified"
    )
