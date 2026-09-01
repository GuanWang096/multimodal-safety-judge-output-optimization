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


AUDIT = _load("audit_soft_confirmatory_outputs.py")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(value) for value in values) + "\n",
        encoding="utf-8",
    )


def _record(sample_id: str, index: int) -> dict:
    margin = float(index - 1)
    return {
        "schema": "soft-evidence-v1",
        "sample_id": sample_id,
        "judge_service": {"adapter_version": "judge-v1"},
        "soft_evidence": {
            "p_unsafe": 1.0 / (1.0 + math.exp(-margin)),
            "margin_unsafe_minus_safe": margin,
            "predicted_label": "unsafe" if margin >= 0 else "safe",
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "confirm.jsonl"
    entries = [{"sample_id": f"s{index}"} for index in range(3)]
    _write_jsonl(manifest, entries)
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    for run in (run_a, run_b):
        run.mkdir()
        _write_jsonl(
            run / "judgments.jsonl",
            [_record(f"s{index}", index) for index in range(3)],
        )
        _write_json(
            run / "config.json",
            {
                "manifest_sha256": AUDIT.sha256_file(manifest),
                "selected_sample_count": 3,
                "consistency_view_id": "canonical",
                "image_transform": "original",
                "adapter": "judge",
                "adapter_version": "judge-v1",
                "hard_regression": False,
            },
        )
        _write_json(run / "status.json", {"state": "completed"})
    return manifest, run_a, run_b


def test_label_blind_confirmatory_audit_passes(tmp_path: Path) -> None:
    manifest, run_a, run_b = _fixture(tmp_path)
    result = AUDIT.audit_confirmatory_run(
        manifest_path=manifest,
        run_dir=run_a,
        judge="judge",
        adapter_version="judge-v1",
        limit=3,
        repeat_run_dir=run_b,
    )
    assert result["passed"] is True
    assert result["labels_accessed"] is False


def test_confirmatory_audit_rejects_label_leakage(tmp_path: Path) -> None:
    manifest, run_a, _ = _fixture(tmp_path)
    records = AUDIT.load_jsonl(run_a / "judgments.jsonl")
    records[0]["expected_label"] = "unsafe"
    _write_jsonl(run_a / "judgments.jsonl", records)
    result = AUDIT.audit_confirmatory_run(
        manifest_path=manifest,
        run_dir=run_a,
        judge="judge",
        adapter_version="judge-v1",
        limit=3,
    )
    assert result["passed"] is False
    assert result["output_forbidden_field_count"] == 1


def test_confirmatory_audit_rejects_manifest_labels(tmp_path: Path) -> None:
    manifest, run_a, _ = _fixture(tmp_path)
    entries = AUDIT.load_jsonl(manifest)
    entries[0]["assistant_label"] = "safe"
    _write_jsonl(manifest, entries)
    config = AUDIT.load_json(run_a / "config.json")
    config["manifest_sha256"] = AUDIT.sha256_file(manifest)
    _write_json(run_a / "config.json", config)
    result = AUDIT.audit_confirmatory_run(
        manifest_path=manifest,
        run_dir=run_a,
        judge="judge",
        adapter_version="judge-v1",
        limit=3,
    )
    assert result["passed"] is False
    assert result["manifest_forbidden_fields"] == ["assistant_label"]
