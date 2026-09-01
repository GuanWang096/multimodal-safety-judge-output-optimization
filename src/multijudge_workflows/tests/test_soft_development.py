from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load("audit_soft_development.py")
FREEZE = _load("freeze_soft_committee.py")


def _record(sample_id: str, adapter_version: str, p_unsafe: float) -> dict:
    return {
        "schema": "soft-evidence-v1",
        "sample_id": sample_id,
        "judge_service": {"adapter_version": adapter_version},
        "soft_evidence": {
            "p_unsafe": p_unsafe,
            "margin_unsafe_minus_safe": p_unsafe - 0.5,
            "predicted_label": "unsafe" if p_unsafe >= 0.5 else "safe",
        },
    }


def test_development_audit_requires_exact_coverage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    entries = [{"sample_id": f"s{index}"} for index in range(5)]
    manifest.write_text(
        "\n".join(json.dumps(value) for value in entries) + "\n",
        encoding="utf-8",
    )
    records = [
        _record(f"s{index}", "adapter-v1", 0.1 + index * 0.15)
        for index in range(5)
    ]
    (run_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(value) for value in records) + "\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed"}),
        encoding="utf-8",
    )
    report = AUDIT.audit(
        manifest_path=manifest,
        run_dir=run_dir,
        judge="judge",
        adapter_version="adapter-v1",
    )
    assert report["passed"] is True
    (run_dir / "judgments.jsonl").write_text(
        "\n".join(json.dumps(value) for value in records[:-1]) + "\n",
        encoding="utf-8",
    )
    report = AUDIT.audit(
        manifest_path=manifest,
        run_dir=run_dir,
        judge="judge",
        adapter_version="adapter-v1",
    )
    assert report["passed"] is False
    assert report["missing_count"] == 1


def test_committee_freeze_rejects_unqualified_judge(tmp_path: Path) -> None:
    for judge, version in FREEZE.QUALIFIED_SOFT_JUDGES.items():
        judge_dir = tmp_path / judge
        (judge_dir / "qual100-a").mkdir(parents=True)
        (judge_dir / "qual100-b").mkdir(parents=True)
        (judge_dir / "qualification_audit.json").write_text(
            json.dumps({"passed": True, "sample_count": 100}),
            encoding="utf-8",
        )
        for run_name in ("qual100-a", "qual100-b"):
            run_dir = judge_dir / run_name
            (run_dir / "status.json").write_text(
                json.dumps({"state": "completed"}),
                encoding="utf-8",
            )
            (run_dir / "qualification_summary.json").write_text(
                json.dumps({"passed": True, "sample_count": 100}),
                encoding="utf-8",
            )
            (run_dir / "judgments.jsonl").write_text(
                json.dumps(
                    {
                        "judge_service": {
                            "adapter_version": version,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    frozen = FREEZE.freeze(tmp_path)
    assert [
        value["judge"] for value in frozen["qualified_soft_judges"]
    ] == ["qwen", "safework", "minicpm"]
    (tmp_path / "minicpm" / "qualification_audit.json").write_text(
        json.dumps({"passed": False, "sample_count": 100}),
        encoding="utf-8",
    )
    try:
        FREEZE.freeze(tmp_path)
    except RuntimeError as exc:
        assert "minicpm" in str(exc)
    else:
        raise AssertionError("Unqualified judge was accepted")
