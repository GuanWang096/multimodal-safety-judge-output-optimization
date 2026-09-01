from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import pstdev
from typing import Any


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = (
    V15_ROOT / "src"
    if (V15_ROOT / "src" / "multijudge").is_dir()
    else V15_ROOT
)
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from multijudge.soft_scores import validate_soft_record  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(
    *,
    manifest_path: Path,
    run_dir: Path,
    judge: str,
    adapter_version: str,
) -> dict[str, Any]:
    manifest = load_jsonl(manifest_path)
    records = load_jsonl(run_dir / "judgments.jsonl")
    expected_ids = [str(value["sample_id"]) for value in manifest]
    observed_ids = [str(value["sample_id"]) for value in records]
    duplicate_count = len(observed_ids) - len(set(observed_ids))
    missing = sorted(set(expected_ids) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(expected_ids))
    invalid_records = []
    wrong_adapter = []
    probabilities = []
    for record in records:
        try:
            validate_soft_record(record)
            probabilities.append(float(record["soft_evidence"]["p_unsafe"]))
        except (KeyError, TypeError, ValueError) as exc:
            invalid_records.append(
                {"sample_id": record.get("sample_id"), "error": str(exc)}
            )
        observed_adapter = (
            record.get("judge_service", {}).get("adapter_version")
        )
        if observed_adapter != adapter_version:
            wrong_adapter.append(str(record.get("sample_id")))
    status = json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )
    finite = all(math.isfinite(value) for value in probabilities)
    nondegenerate = (
        len({round(value, 12) for value in probabilities}) >= 5
        and pstdev(probabilities) > 1e-8
    )
    passed = (
        bool(expected_ids)
        and len(records) == len(expected_ids)
        and duplicate_count == 0
        and not missing
        and not unexpected
        and not invalid_records
        and not wrong_adapter
        and finite
        and nondegenerate
        and status.get("state") == "completed"
    )
    return {
        "schema": "softfusion-development-coverage-audit-v1",
        "passed": passed,
        "judge": judge,
        "adapter_version": adapter_version,
        "manifest": str(manifest_path),
        "expected_count": len(expected_ids),
        "observed_count": len(records),
        "duplicate_count": duplicate_count,
        "missing_count": len(missing),
        "missing_sample_ids": missing[:20],
        "unexpected_count": len(unexpected),
        "unexpected_sample_ids": unexpected[:20],
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records[:20],
        "wrong_adapter_count": len(wrong_adapter),
        "wrong_adapter_sample_ids": wrong_adapter[:20],
        "finite_scores": finite,
        "unique_probability_count": len(
            {round(value, 12) for value in probabilities}
        ),
        "probability_population_std": (
            pstdev(probabilities) if probabilities else None
        ),
        "nondegenerate": nondegenerate,
        "status_state": status.get("state"),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--adapter-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        manifest_path=args.manifest.resolve(),
        run_dir=args.run_dir.resolve(),
        judge=args.judge,
        adapter_version=args.adapter_version,
    )
    write_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "judge": result["judge"],
                "observed_count": result["observed_count"],
            }
        )
    )
    if not result["passed"]:
        raise RuntimeError("Development soft-score coverage audit failed")


if __name__ == "__main__":
    main()
