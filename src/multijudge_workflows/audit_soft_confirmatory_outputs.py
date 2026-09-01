from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any, Mapping


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src" if (V15_ROOT / "src" / "multijudge").is_dir() else V15_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.soft_scores import validate_soft_record  # noqa: E402


FORBIDDEN_LABEL_FIELDS = {
    "assistant_label",
    "assistant_dimension",
    "expected_label",
    "expected_dimension",
    "label",
    "ground_truth",
    "target",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            values.append(value)
    return values


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_confirmatory_run(
    *,
    manifest_path: Path,
    run_dir: Path,
    judge: str,
    adapter_version: str,
    limit: int | None = None,
    repeat_run_dir: Path | None = None,
    deterministic_tolerance: float = 1e-4,
) -> dict[str, Any]:
    manifest = load_jsonl(manifest_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        manifest = manifest[:limit]
    expected_ids = [str(value["sample_id"]) for value in manifest]
    manifest_forbidden = {
        key
        for value in manifest
        for key in value
        if key.casefold() in FORBIDDEN_LABEL_FIELDS
    }

    result_path = run_dir / "judgments.jsonl"
    config_path = run_dir / "config.json"
    status_path = run_dir / "status.json"
    missing_artifacts = [
        str(path)
        for path in (result_path, config_path, status_path)
        if not path.is_file()
    ]
    records = [] if missing_artifacts else load_jsonl(result_path)
    config = {} if missing_artifacts else load_json(config_path)
    status = {} if missing_artifacts else load_json(status_path)

    observed_ids = [str(value.get("sample_id")) for value in records]
    duplicate_count = len(observed_ids) - len(set(observed_ids))
    missing_ids = sorted(set(expected_ids) - set(observed_ids))
    unexpected_ids = sorted(set(observed_ids) - set(expected_ids))
    invalid_records = []
    wrong_adapter = []
    forbidden_output_fields: dict[str, list[str]] = {}
    probabilities = []
    margins = []
    for record in records:
        sample_id = str(record.get("sample_id"))
        try:
            validate_soft_record(record)
            probabilities.append(float(record["soft_evidence"]["p_unsafe"]))
            margins.append(float(record["soft_evidence"]["margin_unsafe_minus_safe"]))
        except (KeyError, TypeError, ValueError) as exc:
            invalid_records.append({"sample_id": sample_id, "error": str(exc)})
        found_forbidden = sorted(
            key for key in record if key.casefold() in FORBIDDEN_LABEL_FIELDS
        )
        if found_forbidden:
            forbidden_output_fields[sample_id] = found_forbidden
        observed_adapter = record.get("judge_service", {}).get("adapter_version")
        if observed_adapter != adapter_version:
            wrong_adapter.append(sample_id)

    finite = all(math.isfinite(value) for value in probabilities + margins)
    in_range = all(0.0 <= value <= 1.0 for value in probabilities)
    nondegenerate = (
        len(records) < 20
        or (
            len({round(value, 12) for value in probabilities}) >= 5
            and pstdev(probabilities) > 1e-8
        )
    )
    config_checks = {
        "manifest_hash_matches": config.get("manifest_sha256") == sha256_file(manifest_path),
        "selected_count_matches": config.get("selected_sample_count") == len(expected_ids),
        "canonical_view": config.get("consistency_view_id") == "canonical",
        "original_images": config.get("image_transform") == "original",
        "adapter_matches": config.get("adapter") == judge,
        "adapter_version_matches": config.get("adapter_version") == adapter_version,
        "hard_regression_disabled": config.get("hard_regression") is False,
    }

    repeatability: dict[str, Any] = {"required": repeat_run_dir is not None}
    if repeat_run_dir is not None:
        repeat_records = load_jsonl(repeat_run_dir / "judgments.jsonl")
        repeat_by_id = {str(value["sample_id"]): value for value in repeat_records}
        differences = []
        for record in records:
            sample_id = str(record["sample_id"])
            other = repeat_by_id.get(sample_id)
            if other is None:
                differences.append({"sample_id": sample_id, "reason": "missing_in_repeat"})
                continue
            left = float(record["soft_evidence"]["margin_unsafe_minus_safe"])
            right = float(other["soft_evidence"]["margin_unsafe_minus_safe"])
            difference = abs(left - right)
            if difference > deterministic_tolerance:
                differences.append(
                    {"sample_id": sample_id, "absolute_margin_difference": difference}
                )
        repeatability.update(
            {
                "repeat_record_count": len(repeat_records),
                "sample_ids_exact": [str(value["sample_id"]) for value in repeat_records]
                == observed_ids,
                "deterministic_tolerance": deterministic_tolerance,
                "difference_count": len(differences),
                "differences": differences[:20],
                "passed": (
                    len(repeat_records) == len(records)
                    and [str(value["sample_id"]) for value in repeat_records] == observed_ids
                    and not differences
                ),
            }
        )
    else:
        repeatability["passed"] = True

    passed = (
        bool(expected_ids)
        and not missing_artifacts
        and not manifest_forbidden
        and len(records) == len(expected_ids)
        and observed_ids == expected_ids
        and duplicate_count == 0
        and not missing_ids
        and not unexpected_ids
        and not invalid_records
        and not forbidden_output_fields
        and not wrong_adapter
        and finite
        and in_range
        and nondegenerate
        and status.get("state") == "completed"
        and all(config_checks.values())
        and bool(repeatability["passed"])
    )
    return {
        "schema": "softfusion-confirmatory-output-audit-v1",
        "passed": passed,
        "labels_accessed": False,
        "judge": judge,
        "adapter_version": adapter_version,
        "manifest_sha256": sha256_file(manifest_path),
        "judgments_sha256": sha256_file(result_path) if result_path.is_file() else None,
        "expected_count": len(expected_ids),
        "observed_count": len(records),
        "sample_order_exact": observed_ids == expected_ids,
        "duplicate_count": duplicate_count,
        "missing_count": len(missing_ids),
        "missing_sample_ids": missing_ids[:20],
        "unexpected_count": len(unexpected_ids),
        "unexpected_sample_ids": unexpected_ids[:20],
        "manifest_forbidden_fields": sorted(manifest_forbidden),
        "output_forbidden_field_count": len(forbidden_output_fields),
        "output_forbidden_fields": forbidden_output_fields,
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records[:20],
        "wrong_adapter_count": len(wrong_adapter),
        "finite_scores": finite,
        "probabilities_in_range": in_range,
        "unique_probability_count": len({round(value, 12) for value in probabilities}),
        "probability_population_std": pstdev(probabilities) if probabilities else None,
        "nondegenerate": nondegenerate,
        "status_state": status.get("state"),
        "config_checks": config_checks,
        "repeatability": repeatability,
        "missing_artifacts": missing_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--adapter-version", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat-run-dir", type=Path)
    parser.add_argument("--deterministic-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_confirmatory_run(
        manifest_path=args.manifest.resolve(),
        run_dir=args.run_dir.resolve(),
        judge=args.judge,
        adapter_version=args.adapter_version,
        limit=args.limit,
        repeat_run_dir=(
            None if args.repeat_run_dir is None else args.repeat_run_dir.resolve()
        ),
        deterministic_tolerance=args.deterministic_tolerance,
    )
    write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "judge": result["judge"],
                "observed_count": result["observed_count"],
                "labels_accessed": result["labels_accessed"],
            }
        )
    )
    if not result["passed"]:
        raise RuntimeError("Confirmatory output audit failed")


if __name__ == "__main__":
    main()
