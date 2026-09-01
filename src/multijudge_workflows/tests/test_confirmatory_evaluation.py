from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parents[1]
RESULT_BASE = REPOSITORY_ROOT / "results" / "mmds"


def _load(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATE = _load("evaluate_softfusion_confirmatory.py")
AUDIT = _load("audit_softfusion_results.py")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(value) for value in values) + "\n",
        encoding="utf-8",
    )


def test_confirmatory_evaluation_and_independent_audit(tmp_path: Path) -> None:
    manifest = RESULT_BASE / "data" / "confirm-unlabeled.jsonl"
    labels = RESULT_BASE / "data" / "confirm-labels.jsonl"
    fusion_config = RESULT_BASE / "fit" / "fusion_config.json"
    analysis_source = RESULT_BASE / "fit" / "confirmatory_analysis_plan.json"
    entries = EVALUATE.load_jsonl(manifest)
    label_values = EVALUATE.load_jsonl(labels)
    label_by_id = {str(value["sample_id"]): value for value in label_values}
    analysis = EVALUATE.load_json(analysis_source)
    analysis["statistics"]["bootstrap_replicates"] = 100
    analysis["statistics"]["randomization_replicates"] = 100
    analysis.pop("analysis_plan_sha256")
    analysis["analysis_plan_sha256"] = EVALUATE.stable_json_sha256(analysis)
    analysis_path = tmp_path / "analysis.json"
    _write_json(analysis_path, analysis)

    result_root = tmp_path / "raw"
    versions = analysis["adapter_versions"]
    offsets = {"qwen": 7, "safework": 11, "minicpm": 5}
    raw_freeze = {
        "schema": "softfusion-confirmatory-raw-output-freeze-v1",
        "status": "frozen",
        "labels_accessed": False,
        "judges": {},
    }
    for judge, period in offsets.items():
        records = []
        for index, entry in enumerate(entries):
            unsafe = label_by_id[str(entry["sample_id"])]["assistant_label"] == "unsafe"
            predicted_unsafe = unsafe if index % period else not unsafe
            probability = 0.8 if predicted_unsafe else 0.2
            records.append(
                {
                    "schema": "soft-evidence-v1",
                    "sample_id": entry["sample_id"],
                    "judge_service": {"adapter_version": versions[judge]},
                    "soft_evidence": {
                        "p_unsafe": probability,
                        "margin_unsafe_minus_safe": 1.0 if predicted_unsafe else -1.0,
                        "predicted_label": "unsafe" if predicted_unsafe else "safe",
                    },
                }
            )
        path = result_root / judge / "judgments.jsonl"
        _write_jsonl(path, records)
        raw_freeze["judges"][judge] = {
            "files": {"judgments": {"sha256": EVALUATE.sha256_file(path)}}
        }
    raw_freeze["content_sha256"] = EVALUATE.stable_json_sha256(raw_freeze)
    raw_freeze_path = tmp_path / "raw_freeze.json"
    _write_json(raw_freeze_path, raw_freeze)

    output_root = tmp_path / "evaluation"
    result = EVALUATE.evaluate(
        SimpleNamespace(
            manifest=manifest,
            labels=labels,
            result_root=result_root,
            raw_freeze=raw_freeze_path,
            fusion_config=fusion_config,
            analysis_plan=analysis_path,
            output_root=output_root,
        )
    )
    assert result["status"] == "completed"
    assert result["confirmatory_record_count"] == 513
    assert result["confirmatory_component_count"] == 257

    audit = AUDIT.audit(
        SimpleNamespace(
            results=output_root / "confirmatory_results.json",
            predictions=output_root / "confirmatory_predictions.jsonl",
            manifest=manifest,
            labels=labels,
            analysis_plan=analysis_path,
            output=output_root / "result_audit.json",
            report=output_root / "RESULT_AUDIT.md",
        )
    )
    assert audit["passed"] is True
