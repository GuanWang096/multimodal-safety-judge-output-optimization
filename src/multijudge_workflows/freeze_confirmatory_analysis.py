from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fit_calibrators import load_split  # noqa: E402
from fit_softfusion import cross_fitted_predictions  # noqa: E402
from multijudge.baselines import (  # noqa: E402
    classification_metrics,
    evaluate_baselines,
    select_constrained_threshold,
)
from multijudge.calibration import (  # noqa: E402
    apply_calibrator,
    fit_calibrator,
    make_group_folds,
    stable_json_sha256,
)
from multijudge.fusion import fit_calibrated_fusion  # noqa: E402


FORBIDDEN_LABEL_FIELDS = {
    "assistant_label",
    "assistant_dimension",
    "expected_label",
    "expected_dimension",
    "label",
    "ground_truth",
    "target",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def subset_probabilities(
    probabilities: Mapping[str, Sequence[float]],
    indices: np.ndarray,
    judges: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        judge: np.asarray(probabilities[judge], dtype=np.float64)[indices]
        for judge in judges
    }


def cross_fitted_calibrated_mean(
    *,
    raw_probabilities: Mapping[str, Sequence[float]],
    targets: np.ndarray,
    groups: Sequence[str],
    folds: np.ndarray,
    judges: Sequence[str],
    methods: Mapping[str, str],
) -> np.ndarray:
    output = np.full(len(targets), np.nan, dtype=np.float64)
    group_values = np.asarray([str(value) for value in groups])
    for fold in sorted(set(int(value) for value in folds)):
        validation = folds == fold
        training = ~validation
        columns = []
        for judge in judges:
            raw = np.asarray(raw_probabilities[judge], dtype=np.float64)
            parameters = fit_calibrator(
                methods[judge],
                raw[training],
                targets[training],
                group_values[training].tolist(),
            )
            columns.append(apply_calibrator(parameters, raw[validation]))
        output[validation] = np.column_stack(columns).mean(axis=1)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Cross-fitted calibrated mean is incomplete")
    return output


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    fusion_config = load_json(args.fusion_config.resolve())
    freeze_record = load_json(args.freeze_record.resolve())
    selection_config = load_json(args.selection_config.resolve())
    baseline_selection = load_json(args.baseline_selection.resolve())
    if fusion_config.get("status") != "frozen":
        raise RuntimeError("Fusion config is not frozen")
    embedded_hash = str(fusion_config.get("config_sha256"))
    unhashed = dict(fusion_config)
    unhashed.pop("config_sha256", None)
    if stable_json_sha256(unhashed) != embedded_hash:
        raise RuntimeError("Fusion config embedded hash does not match")
    if freeze_record.get("fusion_config_sha256") != embedded_hash:
        raise RuntimeError("Freeze record does not link the fusion config")
    if freeze_record.get("confirmatory_run_permitted") is not True:
        raise RuntimeError("SF3 did not permit the confirmatory run")
    if fusion_config.get("feature_scope") != "canonical_only":
        raise RuntimeError("Confirmatory analysis requires canonical-only features")

    confirm_manifest = load_jsonl(args.confirm_manifest.resolve())
    forbidden = sorted(
        {
            key
            for record in confirm_manifest
            for key in record
            if key.casefold() in FORBIDDEN_LABEL_FIELDS
        }
    )
    if forbidden:
        raise RuntimeError(f"Confirmatory inference manifest contains labels: {forbidden}")
    confirm_ids = [str(value["sample_id"]) for value in confirm_manifest]
    if len(confirm_ids) != 513 or len(set(confirm_ids)) != len(confirm_ids):
        raise RuntimeError("Unexpected confirmatory manifest membership")

    judges = tuple(str(value) for value in fusion_config["judge_order"])
    if judges != ("qwen", "safework", "minicpm"):
        raise RuntimeError("Unexpected frozen judge pool")
    methods = {
        judge: str(fusion_config["calibrator_methods"][judge]) for judge in judges
    }
    fit = load_split(args.fit_manifest.resolve(), args.soft_root.resolve(), "fit")
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
        for judge in judges
    }
    folds = make_group_folds(
        targets,
        groups,
        n_splits=int(selection_config["outer_fold_count"]),
        seed=int(selection_config["seed"]) + 3000000,
    )
    baseline_evaluations = evaluate_baselines(targets, raw, judges)
    strongest = str(baseline_selection["strongest_simple_baseline"])
    if strongest != "or_1_of_n":
        raise RuntimeError("Unexpected strongest simple comparator")
    reference_metrics = baseline_evaluations[strongest]["metrics"]
    minimum_recall = max(
        0.0,
        float(reference_metrics["unsafe_recall"])
        - float(
            selection_config["threshold_constraints"][
                "unsafe_recall_margin_below_reference"
            ]
        ),
    )
    minimum_specificity = max(
        0.0,
        float(reference_metrics["safe_specificity"])
        - float(
            selection_config["threshold_constraints"][
                "safe_specificity_margin_below_reference"
            ]
        ),
    )

    calibrated_mean_oof = cross_fitted_calibrated_mean(
        raw_probabilities=raw,
        targets=targets,
        groups=groups,
        folds=folds,
        judges=judges,
        methods=methods,
    )
    calibrated_mean_threshold = select_constrained_threshold(
        targets,
        calibrated_mean_oof,
        minimum_unsafe_recall=minimum_recall,
        minimum_safe_specificity=minimum_specificity,
    )
    if not calibrated_mean_threshold.get("feasible"):
        raise RuntimeError("Calibrated soft-mean baseline has no feasible threshold")

    selected_hyperparameters = {
        "penalty": fusion_config["model"]["penalty"],
        "C": fusion_config["model"]["C"],
        "l1_ratio": fusion_config["model"].get("l1_ratio"),
    }
    full_threshold = float(fusion_config["threshold"])
    leave_one_out = {}
    for offset, omitted in enumerate(judges):
        reduced = tuple(judge for judge in judges if judge != omitted)
        reduced_oof, _ = cross_fitted_predictions(
            raw_probabilities=raw,
            targets=targets,
            groups=groups,
            folds=folds,
            judges=reduced,
            calibrator_methods=methods,
            feature_family=str(fusion_config["model"]["feature_family"]),
            hyperparameters=selected_hyperparameters,
            seed=int(selection_config["seed"]) + 3300000 + offset * 10000,
        )
        reduced_calibrators, reduced_model = fit_calibrated_fusion(
            raw,
            targets,
            groups,
            reduced,
            methods,
            selected_hyperparameters,
            feature_family=str(fusion_config["model"]["feature_family"]),
            seed=int(selection_config["seed"]) + 3500000 + offset,
        )
        leave_one_out[omitted] = {
            "included_judges": list(reduced),
            "calibrators": reduced_calibrators,
            "model": reduced_model.to_dict(),
            "threshold": full_threshold,
            "development_oof_metrics_at_full_threshold": classification_metrics(
                targets, reduced_oof >= full_threshold
            ),
        }

    plan: dict[str, Any] = {
        "schema": "softfusion-confirmatory-analysis-plan-v1",
        "status": "frozen",
        "seed_base": 20260724,
        "confirmatory_manifest_sha256": sha256_file(args.confirm_manifest.resolve()),
        "confirmatory_record_count": len(confirm_ids),
        "fusion_config_sha256": embedded_hash,
        "fusion_config_file_sha256": sha256_file(args.fusion_config.resolve()),
        "freeze_record_file_sha256": sha256_file(args.freeze_record.resolve()),
        "feature_scope": "canonical_only",
        "judge_order": list(judges),
        "adapter_versions": fusion_config["adapter_versions"],
        "main_method": {
            "family": fusion_config["model"]["feature_family"],
            "threshold": full_threshold,
            "description": "calibrated canonical soft-evidence main-effect stacker",
        },
        "confirmatory_hierarchy": [
            "fusion_vs_majority",
            "fusion_vs_or_1_of_n",
        ],
        "strongest_simple_baseline": strongest,
        "hard_baselines": [
            "majority",
            "or_1_of_n",
            "and_n_of_n",
            "single_qwen",
            "single_safework",
            "single_minicpm",
        ],
        "calibrated_soft_mean": {
            "aggregation": "arithmetic_mean_of_final_calibrated_probabilities",
            "threshold": float(calibrated_mean_threshold["threshold"]),
            "development_oof_metrics": calibrated_mean_threshold["metrics"],
            "role": "prespecified_nonparametric_soft_baseline",
        },
        "risk_constraints": {
            "reference": strongest,
            "unsafe_recall_margin_below_reference": float(
                selection_config["threshold_constraints"][
                    "unsafe_recall_margin_below_reference"
                ]
            ),
            "safe_specificity_margin_below_reference": float(
                selection_config["threshold_constraints"][
                    "safe_specificity_margin_below_reference"
                ]
            ),
        },
        "leave_one_judge_out": leave_one_out,
        "marginal_gate": selection_config["judge_marginal_gate"],
        "statistics": {
            "paired_unit": "component_id",
            "bootstrap_replicates": 10000,
            "randomization_replicates": 10000,
            "confidence_level": 0.95,
            "primary_metric": "macro_f1",
            "primary_pass_rule": "paired_component_bootstrap_ci_lower_gt_zero",
            "randomization_role": "supporting_inference",
            "mcnemar_role": "compatibility_appendix_only",
        },
        "excluded_confirmatory_features": [
            "policy_consistency",
            "visual_consistency",
            "pairwise_interactions",
        ],
        "excluded_claims": [
            "multi_view_consistency_improves_accuracy",
            "pairwise_dependence_terms_improve_accuracy",
        ],
        "confirmatory_labels_accessed": False,
        "confirmatory_predictions_generated": False,
    }
    plan["analysis_plan_sha256"] = stable_json_sha256(plan)
    write_json(args.output.resolve(), plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=SCRIPT_DIR / "configs" / "softfusion_fit.jsonl",
    )
    parser.add_argument(
        "--select-manifest",
        type=Path,
        default=SCRIPT_DIR / "configs" / "softfusion_select.jsonl",
    )
    parser.add_argument(
        "--confirm-manifest",
        type=Path,
        default=V15_ROOT / "results" / "mmds" / "data" / "confirm-unlabeled.jsonl",
    )
    parser.add_argument(
        "--soft-root",
        type=Path,
        default=V15_ROOT / "results" / "mmds" / "qualification",
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=SCRIPT_DIR / "configs" / "softfusion_model_selection_v1.json",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=V15_ROOT / "results" / "mmds" / "fit" / "fusion_config.json",
    )
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=V15_ROOT / "results" / "mmds" / "fit" / "freeze_record.json",
    )
    parser.add_argument(
        "--baseline-selection",
        type=Path,
        default=V15_ROOT / "results" / "mmds" / "fit" / "baseline_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V15_ROOT / "results" / "mmds" / "fit" / "confirmatory_analysis_plan.json",
    )
    return parser


if __name__ == "__main__":
    result = freeze(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "analysis_plan_sha256": result["analysis_plan_sha256"],
                "confirmatory_record_count": result["confirmatory_record_count"],
                "confirmatory_labels_accessed": result["confirmatory_labels_accessed"],
            }
        )
    )
