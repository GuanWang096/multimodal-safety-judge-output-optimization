from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load("run_soft_scores.py")
AUDIT = _load("audit_soft_qualification.py")


def _record(sample_id: str, margin_a: float, margin_b: float) -> dict:
    margin = (margin_a + margin_b) / 2
    p_unsafe = 1.0 / (1.0 + math.exp(-margin))
    return {
        "schema": "soft-evidence-v1",
        "sample_id": sample_id,
        "expected_label": "unsafe",
        "soft_evidence": {
            "p_unsafe": p_unsafe,
            "margin_unsafe_minus_safe": margin,
            "predicted_label": "unsafe" if margin >= 0 else "safe",
            "codebooks": [
                {
                    "margin_unsafe_minus_safe": margin_a,
                    "p_unsafe": 1.0 / (1.0 + math.exp(-margin_a)),
                },
                {
                    "margin_unsafe_minus_safe": margin_b,
                    "p_unsafe": 1.0 / (1.0 + math.exp(-margin_b)),
                },
            ],
        },
        "hard_regression": {
            "status": "ok",
            "label": "unsafe" if margin >= 0 else "safe",
            "agrees_with_soft_argmax": True,
        },
    }


def _native_record(sample_id: str, margin: float) -> dict:
    return {
        "schema": "soft-evidence-v1",
        "sample_id": sample_id,
        "soft_evidence": {
            "score_method": "native_label_sequence_likelihood",
            "p_unsafe": 1.0 / (1.0 + math.exp(-margin)),
            "margin_unsafe_minus_safe": margin,
            "predicted_label": "unsafe" if margin >= 0 else "safe",
            "codebooks": [],
        },
        "hard_regression": {
            "status": "ok",
            "label": "unsafe" if margin >= 0 else "safe",
            "agrees_with_soft_argmax": True,
        },
    }


def test_single_run_summary_enforces_hard_agreement() -> None:
    records = [_record(f"s{index}", index + 0.1, index + 0.2) for index in range(20)]
    summary = RUNNER.summarize_run(records, require_hard_regression=True)
    assert summary["passed"] is True
    records[0]["hard_regression"]["agrees_with_soft_argmax"] = False
    summary = RUNNER.summarize_run(records, require_hard_regression=True)
    assert summary["hard_argmax_agreement_rate"] == 0.95
    assert summary["passed"] is True
    records[1]["hard_regression"]["agrees_with_soft_argmax"] = False
    assert (
        RUNNER.summarize_run(records, require_hard_regression=True)["passed"]
        is False
    )


def test_small_smoke_can_require_valid_hard_outputs_without_agreement() -> None:
    records = [_record(f"s{index}", index + 0.1, index + 0.2) for index in range(3)]
    records[0]["hard_regression"]["agrees_with_soft_argmax"] = False
    summary = RUNNER.summarize_run(
        records,
        require_hard_regression=True,
        minimum_hard_agreement=0.0,
    )
    assert summary["hard_regression_valid_count"] == 3
    assert summary["hard_argmax_agreement_rate"] == 2 / 3
    assert summary["passed"] is True


def test_repeat_audit_detects_margin_change(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    records = [_record(f"s{index}", index + 0.1, index + 0.2) for index in range(5)]
    first.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    assert AUDIT.audit(first, second)["passed"] is True
    changed = [dict(record) for record in records]
    changed[0] = json.loads(json.dumps(changed[0]))
    changed[0]["soft_evidence"]["margin_unsafe_minus_safe"] += 0.01
    second.write_text(
        "\n".join(json.dumps(record) for record in changed) + "\n",
        encoding="utf-8",
    )
    assert AUDIT.audit(first, second)["passed"] is False


def test_codebook_gate_uses_probability_order(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    records = [_record(f"s{index}", index + 0.1, index + 0.2) for index in range(5)]
    for index, record in enumerate(records):
        record["soft_evidence"]["codebooks"][1]["p_unsafe"] = (
            1.0 - index / 10.0
        )
    payload = "\n".join(json.dumps(record) for record in records) + "\n"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")
    report = AUDIT.audit(first, second, enforce_codebook_gate=True)
    assert report["passed"] is False
    assert (
        report["codebook_sensitivity"][
            "spearman_probability_correlation"
        ]
        < 0
    )


def test_native_sequence_fallback_satisfies_codebook_gate(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    records = [_native_record(f"s{index}", index - 2.5) for index in range(6)]
    payload = "\n".join(json.dumps(record) for record in records) + "\n"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")
    report = AUDIT.audit(first, second, enforce_codebook_gate=True)
    assert report["passed"] is True
    assert report["codebook_sensitivity"]["applicable"] is False
    assert (
        report["codebook_sensitivity"][
            "native_sequence_fallback_accepted"
        ]
        is True
    )
