from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


QUALIFIED_SOFT_JUDGES = {
    "qwen": "qwen3-vl-soft-score-v1",
    "safework": "safework-rm-safety-native-sequence-sum-v3",
    "minicpm": "minicpm-v4.5-soft-score-v1",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No records found in {path}")


def freeze(result_root: Path) -> dict[str, Any]:
    qualified: list[dict[str, Any]] = []
    for judge, adapter_version in QUALIFIED_SOFT_JUDGES.items():
        judge_dir = result_root / judge
        audit_path = judge_dir / "qualification_audit.json"
        audit = read_json(audit_path)
        if audit.get("passed") is not True or audit.get("sample_count") != 100:
            raise RuntimeError(f"{judge} qualification audit did not pass")
        repetitions = []
        for run_name in ("qual100-a", "qual100-b"):
            run_dir = judge_dir / run_name
            status = read_json(run_dir / "status.json")
            summary = read_json(run_dir / "qualification_summary.json")
            if status.get("state") != "completed" or summary.get("passed") is not True:
                raise RuntimeError(f"{judge}/{run_name} is not completed")
            if summary.get("sample_count") != 100:
                raise RuntimeError(f"{judge}/{run_name} has the wrong sample count")
            record = first_record(run_dir / "judgments.jsonl")
            observed = record["judge_service"]["adapter_version"]
            if observed != adapter_version:
                raise RuntimeError(
                    f"{judge} adapter mismatch: {observed} != {adapter_version}"
                )
            repetitions.append(
                {
                    "run": run_name,
                    "judgments_sha256": sha256_file(
                        run_dir / "judgments.jsonl"
                    ),
                    "summary_sha256": sha256_file(
                        run_dir / "qualification_summary.json"
                    ),
                }
            )
        qualified.append(
            {
                "judge": judge,
                "adapter_version": adapter_version,
                "qualification_audit_sha256": sha256_file(audit_path),
                "qualification_runs": repetitions,
            }
        )
    return {
        "schema": "softfusion-committee-freeze-v2",
        "qualified_soft_judges": qualified,
        "hard_only_judges": [
            {
                "judge": "internvl",
                "reason": "frozen_soft_hard_agreement_gate_failed",
                "numeric_codebook_agreement": 0.93,
                "direct_label_agreement": 0.92,
                "minimum_required_agreement": 0.95,
                "allowed_role": "hard_vote_baseline_only",
            }
        ],
        "primary_simple_baseline": "two_of_three_hard_majority_same_pool",
        "additional_baseline": "four_judge_hard_majority",
        "missing_score_policy": "fail_closed_no_imputation",
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = freeze(args.result_root.resolve())
    write_json(args.output.resolve(), frozen)
    print(
        json.dumps(
            {
                "qualified_soft_judges": [
                    value["judge"]
                    for value in frozen["qualified_soft_judges"]
                ],
                "hard_only_judges": [
                    value["judge"] for value in frozen["hard_only_judges"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
