from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(path))

from fit_calibrators import load_split  # noqa: E402
from fit_softfusion import component_bootstrap_delta, sha256_file  # noqa: E402
from multijudge.baselines import (  # noqa: E402
    baseline_predictions,
    classification_metrics,
    evaluate_baselines,
    rule_equivalence,
)
from multijudge.calibration import (  # noqa: E402
    apply_calibrator,
    cluster_averaged_log_loss,
    stable_json_sha256,
)
from multijudge.fusion import (  # noqa: E402
    calibrated_feature_matrix,
    fusion_model_from_dict,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def component_permutation_importance(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    model: Any,
    threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    group_values = np.asarray([str(value) for value in groups])
    component_ids = np.asarray(sorted(set(group_values)))
    component_means = {
        component: np.mean(x[group_values == component], axis=0)
        for component in component_ids
    }
    baseline_probability = model.predict_features(x)
    baseline_loss = cluster_averaged_log_loss(
        targets, baseline_probability, groups
    )
    baseline_metrics = classification_metrics(
        targets, baseline_probability >= threshold
    )
    rng = np.random.default_rng(seed)
    output = {}
    for column, name in enumerate(model.feature_names):
        loss_deltas = []
        macro_deltas = []
        for _ in range(replicates):
            source = rng.permutation(component_ids)
            source_by_target = dict(zip(component_ids, source))
            permuted = x.copy()
            for component in component_ids:
                members = group_values == component
                shift = (
                    component_means[source_by_target[component]][column]
                    - component_means[component][column]
                )
                permuted[members, column] += shift
            probability = model.predict_features(permuted)
            metrics = classification_metrics(targets, probability >= threshold)
            loss_deltas.append(
                cluster_averaged_log_loss(targets, probability, groups)
                - baseline_loss
            )
            macro_deltas.append(
                float(baseline_metrics["macro_f1"])
                - float(metrics["macro_f1"])
            )
        output[name] = {
            "mean_log_loss_increase": float(np.mean(loss_deltas)),
            "log_loss_increase_ci95": np.quantile(
                loss_deltas, [0.025, 0.975]
            ).tolist(),
            "mean_macro_f1_decrease": float(np.mean(macro_deltas)),
            "macro_f1_decrease_ci95": np.quantile(
                macro_deltas, [0.025, 0.975]
            ).tolist(),
        }
    return {
        "schema": "softfusion-component-permutation-importance-v1",
        "role": "descriptive_final_development_fit_only",
        "replicates": replicates,
        "features": output,
    }


def audit_markdown(report: dict[str, Any]) -> str:
    delta = report["development_comparison"]
    lines = [
        "# SF3 Freeze Audit",
        "",
        f"**Verdict:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "## Recomputed development comparison",
        "",
        "| Method | Macro-F1 | Unsafe recall | Specificity | MCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("fusion", "majority", "strongest_simple"):
        value = delta[name]
        lines.append(
            "| {name} | {macro:.4f} | {recall:.4f} | {specificity:.4f} | {mcc:.4f} |".format(
                name=name,
                macro=float(value["macro_f1"]),
                recall=float(value["unsafe_recall"]),
                specificity=float(value["safe_specificity"]),
                mcc=float(value["mcc"]),
            )
        )
    lines.extend(
        [
            "",
            "Development results are selection evidence, not confirmatory claims.",
            "",
            "## Gate",
            "",
        ]
    )
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    sf3 = args.sf3_dir.resolve()
    fusion_config = load_json(sf3 / "fusion_config.json")
    freeze_record = load_json(sf3 / "freeze_record.json")
    baseline_selection = load_json(sf3 / "baseline_selection.json")
    feature_schema = load_json(sf3 / "feature_schema.json")
    screening = load_json(args.screening_decision.resolve())
    predictions = load_jsonl(sf3 / "development_oof_predictions.jsonl")
    fit = load_split(
        args.fit_manifest.resolve(), args.soft_root.resolve(), "fit"
    )
    select = load_split(
        args.select_manifest.resolve(), args.soft_root.resolve(), "select"
    )
    targets = np.concatenate([fit["targets"], select["targets"]])
    groups = [f"fit:{value}" for value in fit["components"]] + [
        f"select:{value}" for value in select["components"]
    ]
    raw = {
        judge: np.concatenate(
            [fit["raw_probabilities"][judge], select["raw_probabilities"][judge]]
        )
        for judge in fusion_config["judge_order"]
    }
    expected_ids = fit["sample_ids"] + select["sample_ids"]
    prediction_ids = [str(value["sample_id"]) for value in predictions]
    probabilities = np.asarray(
        [float(value["p_unsafe"]) for value in predictions], dtype=np.float64
    )
    threshold = float(fusion_config["threshold"])
    fusion_metrics = classification_metrics(targets, probabilities >= threshold)
    baselines = evaluate_baselines(targets, raw, fusion_config["judge_order"])
    strongest = str(baseline_selection["strongest_simple_baseline"])
    baseline_binary = baseline_predictions(raw, fusion_config["judge_order"])
    equivalence = rule_equivalence(
        probabilities >= threshold, baseline_binary
    )

    stored_hash = str(fusion_config["config_sha256"])
    hash_input = dict(fusion_config)
    del hash_input["config_sha256"]
    recomputed_hash = stable_json_sha256(hash_input)
    source_hashes = load_json(sf3 / "provenance.json")[
        "source_files_sha256"
    ]
    source_hashes_match = all(
        Path(path).is_file() and sha256_file(Path(path)) == expected
        for path, expected in source_hashes.items()
    )
    reference = baselines[strongest]["metrics"]
    constraints_pass = (
        float(fusion_metrics["unsafe_recall"]) + 1e-12
        >= float(reference["unsafe_recall"]) - 0.01
        and float(fusion_metrics["safe_specificity"]) + 1e-12
        >= float(reference["safe_specificity"]) - 0.03
    )
    checks = {
        "source_hashes_match": source_hashes_match,
        "fusion_config_hash_matches": stored_hash == recomputed_hash,
        "freeze_record_links_config": (
            freeze_record["fusion_config_sha256"] == stored_hash
        ),
        "prediction_ids_exact_and_ordered": prediction_ids == expected_ids,
        "prediction_count_complete": len(predictions) == len(targets) == 1760,
        "stored_metrics_recompute": all(
            abs(
                float(fusion_metrics[name])
                - float(fusion_config["threshold_metrics"][name])
            )
            < 1e-12
            for name in ("macro_f1", "unsafe_recall", "safe_specificity", "mcc")
        ),
        "risk_constraints_hold": constraints_pass,
        "not_equivalent_to_simple_rule": not equivalence["equivalent_rules"],
        "canonical_only": (
            screening["frozen_feature_scope"] == "canonical_only"
            and fusion_config["feature_scope"] == "canonical_only"
            and feature_schema["scope"] == "canonical_only"
        ),
        "confirmatory_labels_not_accessed": (
            fusion_config["confirmatory_labels_accessed"] is False
            and freeze_record["confirmatory_labels_accessed"] is False
        ),
        "development_macro_f1_exceeds_majority": (
            float(fusion_metrics["macro_f1"])
            > float(baselines["majority"]["metrics"]["macro_f1"])
        ),
    }
    passed = all(checks.values()) and freeze_record["stop_gate"]["passed"] is True

    bootstrap = component_bootstrap_delta(
        targets=targets,
        groups=groups,
        left_probabilities=probabilities,
        left_threshold=threshold,
        right_probabilities=baselines["majority"]["predictions"].astype(float),
        right_threshold=0.5,
        replicates=500,
        seed=20260729 + 5000000,
    )

    model = fusion_model_from_dict(fusion_config["model"])
    calibrated = {
        judge: apply_calibrator(
            fusion_config["final_development_calibrators"][judge], raw[judge]
        )
        for judge in model.judge_order
    }
    features, names = calibrated_feature_matrix(
        calibrated, model.judge_order, model.feature_family
    )
    if names != model.feature_names:
        raise RuntimeError("Frozen feature names do not reconstruct")
    importance = component_permutation_importance(
        features=features,
        targets=targets,
        groups=groups,
        model=model,
        threshold=threshold,
        replicates=200,
        seed=20260729 + 6000000,
    )

    report = {
        "schema": "softfusion-sf3-freeze-audit-v1",
        "passed": passed,
        "checks": checks,
        "development_comparison": {
            "fusion": fusion_metrics,
            "majority": baselines["majority"]["metrics"],
            "strongest_simple": baselines[strongest]["metrics"],
            "strongest_simple_name": strongest,
            "fusion_minus_majority_macro_f1": (
                float(fusion_metrics["macro_f1"])
                - float(baselines["majority"]["metrics"]["macro_f1"])
            ),
            "fusion_minus_strongest_simple_macro_f1": (
                float(fusion_metrics["macro_f1"])
                - float(baselines[strongest]["metrics"]["macro_f1"])
            ),
            "fusion_vs_majority_component_bootstrap": bootstrap,
        },
        "rule_equivalence": equivalence,
        "interpretation": (
            "development-only gate; no confirmatory performance claim"
        ),
    }
    write_json(sf3 / "permutation_importance.json", importance)
    write_json(sf3 / "freeze_audit.json", report)
    (sf3 / "FREEZE_AUDIT.md").write_text(
        audit_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "fusion_macro_f1": fusion_metrics["macro_f1"],
                "majority_macro_f1": baselines["majority"]["metrics"]["macro_f1"],
                "strongest_simple": strongest,
                "strongest_simple_macro_f1": baselines[strongest]["metrics"]["macro_f1"],
            }
        )
    )
    if not passed:
        raise RuntimeError("SF3 freeze audit failed")


def build_parser() -> argparse.ArgumentParser:
    softfusion = V15_ROOT / "results" / "mmds"
    parser = argparse.ArgumentParser(description="Audit the frozen SF3 artifacts")
    parser.add_argument(
        "--sf3-dir", type=Path, default=softfusion / "fit"
    )
    parser.add_argument(
        "--fit-manifest", type=Path, default=softfusion / "data" / "fit.jsonl"
    )
    parser.add_argument(
        "--select-manifest", type=Path, default=softfusion / "data" / "select.jsonl"
    )
    parser.add_argument(
        "--soft-root", type=Path, default=softfusion / "qualification"
    )
    parser.add_argument(
        "--screening-decision",
        type=Path,
        default=softfusion / "sf2-consistency" / "screening_decision.json",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
