from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402
from multijudge.soft_scores import validate_soft_record  # noqa: E402


FORBIDDEN_FIELDS = {
    "assistant_label",
    "expected_label",
    "ground_truth",
    "is_response_safe",
    "label",
    "target",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    result_root = args.result_root.resolve()
    plan_path = args.analysis_plan.resolve()
    output_path = args.output.resolve()
    manifest = _load_jsonl(manifest_path)
    plan = _load_json(plan_path)
    plan_hash = str(plan.get("analysis_plan_sha256"))
    unhashed_plan = dict(plan)
    unhashed_plan.pop("analysis_plan_sha256", None)
    if plan_hash != sha256_text(canonical_json(unhashed_plan)):
        raise RuntimeError("External analysis plan hash is invalid")
    if plan.get("status") != "frozen-before-external-inference":
        raise RuntimeError("External analysis plan is not frozen")
    if plan.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("External manifest differs from analysis plan")
    expected_ids = [str(value["sample_id"]) for value in manifest]

    judges: dict[str, Any] = {}
    for judge in plan["judge_order"]:
        formal = result_root / str(judge) / "formal"
        paths = {
            "judgments": formal / "judgments.jsonl",
            "config": formal / "config.json",
            "status": formal / "status.json",
            "coverage_audit": formal / "coverage_audit.json",
            "model_fingerprint": formal / "model_fingerprint.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing external outputs for {judge}: {missing}")
        records = _load_jsonl(paths["judgments"])
        observed_ids = [str(value["sample_id"]) for value in records]
        if observed_ids != expected_ids:
            raise RuntimeError(f"External output order mismatch for {judge}")
        forbidden = {
            key
            for value in records
            for key in value
            if key.casefold() in FORBIDDEN_FIELDS
        }
        if forbidden:
            raise RuntimeError(f"External output contains labels for {judge}: {forbidden}")
        for record in records:
            validate_soft_record(record)
        versions = {
            str(value["judge_service"]["adapter_version"]) for value in records
        }
        expected_version = str(plan["adapter_versions"][judge])
        if versions != {expected_version}:
            raise RuntimeError(f"Unexpected adapter version for {judge}: {versions}")
        status = _load_json(paths["status"])
        coverage = _load_json(paths["coverage_audit"])
        if status.get("state") != "completed" or coverage.get("passed") is not True:
            raise RuntimeError(f"External output audit did not pass for {judge}")
        judges[str(judge)] = {
            "adapter_version": expected_version,
            "record_count": len(records),
            "files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        }

    frozen = {
        "schema": "beavertails-v-external-raw-output-freeze-v1",
        "status": "frozen",
        "dataset": "PKU-Alignment/BeaverTails-V",
        "sample_count": len(expected_ids),
        "manifest_sha256": sha256_file(manifest_path),
        "analysis_plan_sha256": plan_hash,
        "judges": judges,
        "labels_accessed": False,
        "external_model_fitting_performed": False,
    }
    frozen["content_sha256"] = sha256_text(canonical_json(frozen))
    if output_path.is_file():
        prior = _load_json(output_path)
        if prior != frozen:
            raise RuntimeError("Existing raw-output freeze differs from current outputs")
    else:
        _write_json(output_path, frozen)
    return frozen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    base = (
        V15_ROOT
        / "results"
        / "external"
        / "beavertails_v"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "beavertails_v_external_unlabeled.jsonl"
        ),
    )
    parser.add_argument("--result-root", type=Path, default=base / "raw")
    parser.add_argument("--analysis-plan", type=Path, default=base / "analysis_plan.json")
    parser.add_argument("--output", type=Path, default=base / "raw_output_freeze.json")
    return parser


if __name__ == "__main__":
    result = freeze(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "sample_count": result["sample_count"],
                "labels_accessed": result["labels_accessed"],
            }
        )
    )
