from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from cluster_inference import (  # noqa: E402
    paired_component_bootstrap,
    weighted_classification_metrics,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def metrics_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = {
        "sample_count",
        "accuracy",
        "macro_f1",
        "unsafe_recall",
        "safe_specificity",
        "false_safe_rate",
        "mcc",
        "unsafe_true_positive",
        "safe_true_positive",
        "unsafe_false_positive",
        "unsafe_false_negative",
    }
    for key in keys:
        if key not in left or key not in right:
            return False
        if isinstance(left[key], (float, int)) and isinstance(right[key], (float, int)):
            if not math.isclose(float(left[key]), float(right[key]), rel_tol=0, abs_tol=1e-12):
                return False
        elif left[key] != right[key]:
            return False
    return True


def audit(args: argparse.Namespace) -> dict[str, Any]:
    results = load_json(args.results.resolve())
    predictions = load_jsonl(args.predictions.resolve())
    manifest = load_jsonl(args.manifest.resolve())
    labels = load_jsonl(args.labels.resolve())
    analysis = load_json(args.analysis_plan.resolve())
    result_hash = str(results.get("result_content_sha256"))
    result_unhashed = dict(results)
    result_unhashed.pop("result_content_sha256", None)
    analysis_hash = str(analysis.get("analysis_plan_sha256"))
    analysis_unhashed = dict(analysis)
    analysis_unhashed.pop("analysis_plan_sha256", None)

    expected_ids = [str(value["sample_id"]) for value in manifest]
    observed_ids = [str(value["sample_id"]) for value in predictions]
    labels_by_id = {str(value["sample_id"]): value for value in labels}
    targets = np.asarray(
        [1 if labels_by_id[sample_id]["assistant_label"] == "unsafe" else 0 for sample_id in expected_ids],
        dtype=np.int64,
    )
    groups = [str(value["component_id"]) for value in predictions]
    methods = ["fusion", *sorted(predictions[0]["baseline_predictions"])] if predictions else []
    vectors: dict[str, np.ndarray] = {}
    if predictions:
        vectors["fusion"] = np.asarray(
            [1 if value["fusion_prediction"] == "unsafe" else 0 for value in predictions],
            dtype=np.int64,
        )
        for method in predictions[0]["baseline_predictions"]:
            vectors[method] = np.asarray(
                [
                    1 if value["baseline_predictions"][method] == "unsafe" else 0
                    for value in predictions
                ],
                dtype=np.int64,
            )
    recomputed_metrics = {
        name: weighted_classification_metrics(targets, vector)
        for name, vector in vectors.items()
    }
    metric_mismatches = [
        name
        for name in methods
        if name not in results.get("metrics", {})
        or name not in recomputed_metrics
        or not metrics_match(results["metrics"][name], recomputed_metrics[name])
    ]

    majority_bootstrap = None
    strongest_bootstrap = None
    strongest = str(analysis.get("strongest_simple_baseline"))
    if not metric_mismatches and vectors:
        replicates = int(analysis["statistics"]["bootstrap_replicates"])
        majority_bootstrap = paired_component_bootstrap(
            targets=targets,
            groups=groups,
            left_predictions=vectors["fusion"],
            right_predictions=vectors["majority"],
            replicates=replicates,
            seed=int(analysis["seed_base"]) + 410000,
        )
        strongest_bootstrap = paired_component_bootstrap(
            targets=targets,
            groups=groups,
            left_predictions=vectors["fusion"],
            right_predictions=vectors[strongest],
            replicates=replicates,
            seed=int(analysis["seed_base"]) + 410001,
        )
    stored_majority = results.get("comparisons", {}).get("majority", {}).get("bootstrap")
    stored_strongest = results.get("comparisons", {}).get(strongest, {}).get("bootstrap")
    checks = {
        "result_content_hash": stable_json_sha256(result_unhashed) == result_hash,
        "analysis_plan_hash": stable_json_sha256(analysis_unhashed) == analysis_hash,
        "analysis_plan_link": results.get("provenance", {}).get("analysis_plan_sha256") == analysis_hash,
        "manifest_hash": results.get("provenance", {}).get("manifest_sha256") == sha256_file(args.manifest.resolve()),
        "labels_hash": results.get("provenance", {}).get("labels_sha256") == sha256_file(args.labels.resolve()),
        "sample_ids_exact_and_ordered": observed_ids == expected_ids,
        "sample_ids_unique": len(observed_ids) == len(set(observed_ids)),
        "labels_exact": set(labels_by_id) == set(expected_ids) and len(labels_by_id) == len(labels),
        "stored_metrics_recompute": not metric_mismatches,
        "majority_bootstrap_recompute": majority_bootstrap == stored_majority,
        "strongest_bootstrap_recompute": strongest_bootstrap == stored_strongest,
        "analysis_locked": results.get("analysis_locked") is True,
        "record_count": len(predictions) == 513,
        "component_count": len(set(groups)) == 257,
    }
    passed = all(checks.values())
    audit_result = {
        "schema": "softfusion-confirmatory-result-audit-v1",
        "passed": passed,
        "checks": checks,
        "metric_mismatches": metric_mismatches,
        "recomputed_metrics": recomputed_metrics,
        "claim_status": results.get("claim_status"),
        "result_content_sha256": result_hash,
    }
    write_json(args.output.resolve(), audit_result)
    report = [
        "# SF4 Confirmatory Result Audit",
        "",
        f"**Verdict:** `{'PASS' if passed else 'FAIL'}`",
        "",
        *[f"- `{name}`: `{value}`" for name, value in checks.items()],
        "",
    ]
    args.report.resolve().write_text("\n".join(report), encoding="utf-8")
    return audit_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({"passed": result["passed"], "claim_status": result["claim_status"]}))
    if not result["passed"]:
        raise RuntimeError("Confirmatory result audit failed")


if __name__ == "__main__":
    main()
