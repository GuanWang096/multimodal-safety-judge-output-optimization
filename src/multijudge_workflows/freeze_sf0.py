from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
V15_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SF0_DIR = (
    V15_ROOT
    / "results"
    / "mmds"
    / "data"
)
DEFAULT_PUBLICATION_AUDIT = (
    V15_ROOT / "revision-audit" / "PUBLICATION_OVERLAP_AUDIT-v15.md"
)

AUTHORITATIVE_RELATIVE_FILES = [
    "split_manifest.json",
    "fit.jsonl",
    "select.jsonl",
    "confirm-unlabeled.jsonl",
    "confirm-labels.jsonl",
    "eligibility_audit.json",
    "component_audit.json",
    "perceptual_candidate_audit.json",
    "near_duplicate_decisions.json",
    "leakage_audit.json",
    "power_precision_audit.json",
    "model_data_overlap_audit.json",
    "MODEL_DATA_OVERLAP_AUDIT.md",
    "sf0_test_report.json",
    "siuo-annotation/annotation_manifest.json",
    "siuo-annotation/SIUO_BINARY_ANNOTATION_PROTOCOL.md",
    "siuo-annotation/siuo-blind-annotator-a.csv",
    "siuo-annotation/siuo-blind-annotator-b.csv",
    "siuo-annotation/siuo-blind-id-map.json",
]

SOURCE_FILES = [
    SCRIPT_DIR / "prepare_softfusion_splits.py",
    SCRIPT_DIR / "prepare_siuo_blind_annotation.py",
    SCRIPT_DIR / "audit_model_data_overlap.py",
    SCRIPT_DIR / "freeze_sf0.py",
    SCRIPT_DIR / "configs" / "model_data_overlap_evidence.json",
    SCRIPT_DIR / "tests" / "test_softfusion_splits.py",
    SCRIPT_DIR / "tests" / "test_siuo_blind_annotation.py",
    SCRIPT_DIR / "tests" / "test_model_data_overlap.py",
]

CONFIRM_FORBIDDEN_FIELDS = {
    "assistant_label",
    "label",
    "ground_truth",
    "target",
    "unsafe",
    "is_unsafe",
}
SIUO_FORBIDDEN_HEADERS = {
    "correct_option",
    "reference_answer",
    "safety_warning",
    "model_output",
    "model_score",
    "fused_prediction",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def check_confirm_unlabeled(path: Path) -> tuple[int, list[str]]:
    count = 0
    violations: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            count += 1
            record = json.loads(line)
            overlap = sorted(CONFIRM_FORBIDDEN_FIELDS & set(record))
            if overlap:
                violations.append(f"line_{line_number}:{','.join(overlap)}")
    return count, violations


def check_blind_sheet(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        forbidden = sorted(headers & SIUO_FORBIDDEN_HEADERS)
        count = sum(1 for _ in reader)
    return count, forbidden


def build_manifest(sf0_dir: Path, publication_audit: Path) -> dict[str, Any]:
    errors: list[str] = []
    authoritative_paths = [sf0_dir / relative for relative in AUTHORITATIVE_RELATIVE_FILES]
    all_paths = authoritative_paths + SOURCE_FILES + [publication_audit]
    missing = [str(path) for path in all_paths if not path.is_file()]
    errors.extend(f"missing:{path}" for path in missing)
    if missing:
        return {
            "schema": "softfusion-sf0-freeze-v1",
            "status": "failed",
            "passed": False,
            "errors": errors,
        }

    split = load_json(sf0_dir / "split_manifest.json")
    leakage = load_json(sf0_dir / "leakage_audit.json")
    perceptual = load_json(sf0_dir / "perceptual_candidate_audit.json")
    power = load_json(sf0_dir / "power_precision_audit.json")
    model_overlap = load_json(sf0_dir / "model_data_overlap_audit.json")
    annotation = load_json(sf0_dir / "siuo-annotation" / "annotation_manifest.json")

    gates = {
        "split_frozen": split.get("status") == "frozen",
        "allocation_passed": split.get("allocation_gate", {}).get("passed") is True,
        "exact_leakage_passed": leakage.get("passed") is True,
        "perceptual_leakage_passed": perceptual.get("passed") is True,
        "power_design_passed": power.get("design_gate_passed") is True,
        "model_overlap_passed": model_overlap.get("passed") is True,
        "siuo_policy_frozen": (
            annotation.get("status") == "frozen-for-independent-annotation"
        ),
    }
    errors.extend(f"gate_failed:{key}" for key, value in gates.items() if not value)

    confirm_count, confirm_violations = check_confirm_unlabeled(
        sf0_dir / "confirm-unlabeled.jsonl"
    )
    if confirm_violations:
        errors.extend(f"confirm_label_leak:{item}" for item in confirm_violations)

    blind_sheet_checks: dict[str, Any] = {}
    for name in ("siuo-blind-annotator-a.csv", "siuo-blind-annotator-b.csv"):
        count, forbidden = check_blind_sheet(sf0_dir / "siuo-annotation" / name)
        blind_sheet_checks[name] = {
            "row_count": count,
            "forbidden_headers": forbidden,
        }
        if forbidden:
            errors.append(f"blind_sheet_forbidden_headers:{name}:{','.join(forbidden)}")

    expected_confirm_count = split["split_counts"]["confirm"]["record_count"]
    if confirm_count != expected_confirm_count:
        errors.append(
            f"confirm_count_mismatch:{confirm_count}!={expected_confirm_count}"
        )

    hashes: dict[str, Any] = {}
    for path in all_paths:
        if path.is_relative_to(V15_ROOT):
            key = path.relative_to(V15_ROOT).as_posix()
        else:
            key = str(path)
        hashes[key] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    return {
        "schema": "softfusion-sf0-freeze-v1",
        "status": "frozen" if not errors else "failed",
        "passed": not errors,
        "freeze_policy": (
            "After this manifest passes, confirm membership and labels, near-duplicate "
            "decisions, model identities, and SIUO annotation policy cannot be changed "
            "in response to model outputs."
        ),
        "gates": gates,
        "confirm_unlabeled_check": {
            "record_count": confirm_count,
            "forbidden_field_violations": confirm_violations,
        },
        "siuo_blind_sheet_checks": blind_sheet_checks,
        "design_precision_boundary": {
            "minimum_power_at_delta_0_05": power["minimum_power_at_delta_0_05"],
            "minimum_detectable_delta_at_80_percent_power": power[
                "minimum_detectable_delta_at_80_percent_power"
            ],
        },
        "model_data_boundary": model_overlap["frozen_interpretation"],
        "hashes": hashes,
        "errors": errors,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    sf0_dir = args.sf0_dir.resolve()
    publication_audit = args.publication_audit.resolve()
    manifest = build_manifest(sf0_dir, publication_audit)
    output = sf0_dir / "sf0_freeze_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": manifest["passed"], "status": manifest["status"]}))
    if not manifest["passed"]:
        raise RuntimeError("SF0 freeze gate failed")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf0-dir", type=Path, default=DEFAULT_SF0_DIR)
    parser.add_argument(
        "--publication-audit",
        type=Path,
        default=DEFAULT_PUBLICATION_AUDIT,
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
