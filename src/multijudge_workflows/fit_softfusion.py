from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(path))

from fit_calibrators import load_split  # noqa: E402
from multijudge.baselines import (  # noqa: E402
    baseline_predictions,
    classification_metrics,
    evaluate_baselines,
    rule_equivalence,
    select_constrained_threshold,
    select_strongest_baseline,
    vote_pattern_report,
)
from multijudge.calibration import (  # noqa: E402
    cluster_averaged_log_loss,
    make_group_folds,
    stable_json_sha256,
)
from multijudge.fusion import (  # noqa: E402
    apply_calibrators,
    calibrated_feature_matrix,
    fit_calibrated_fusion,
    fit_fusion_model,
    predict_calibrated_fusion,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def subset_probabilities(
    probabilities: Mapping[str, Sequence[float]],
    indices: np.ndarray,
    judges: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        judge: np.asarray(probabilities[judge], dtype=np.float64)[indices]
        for judge in judges
    }


def hyperparameter_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for penalty in config["penalties"]:
        for c_value in config["C_grid"]:
            if penalty == "l2":
                output.append(
                    {"penalty": "l2", "C": float(c_value), "l1_ratio": None}
                )
            elif penalty == "elasticnet":
                for ratio in config["elasticnet_l1_ratio_grid"]:
                    output.append(
                        {
                            "penalty": "elasticnet",
                            "C": float(c_value),
                            "l1_ratio": float(ratio),
                        }
                    )
            else:
                raise ValueError(f"Unsupported configured penalty: {penalty}")
    return output


def hyperparameter_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if value["penalty"] == "l2" else 1,
        float(value["C"]),
        -1.0 if value.get("l1_ratio") is None else float(value["l1_ratio"]),
    )


def cross_fitted_predictions(
    *,
    raw_probabilities: Mapping[str, Sequence[float]],
    targets: np.ndarray,
    groups: Sequence[str],
    folds: np.ndarray,
    judges: Sequence[str],
    calibrator_methods: Mapping[str, str],
    feature_family: str,
    hyperparameters: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = np.asarray(targets, dtype=np.int64)
    group_values = np.asarray([str(value) for value in groups])
    output = np.full(len(y), np.nan, dtype=np.float64)
    records = []
    for fold in sorted(set(int(value) for value in folds)):
        validation = folds == fold
        training = ~validation
        if set(group_values[training]) & set(group_values[validation]):
            raise RuntimeError("A component crossed a fusion fold")
        parameters, model = fit_calibrated_fusion(
            subset_probabilities(raw_probabilities, training, judges),
            y[training],
            group_values[training].tolist(),
            judges,
            calibrator_methods,
            hyperparameters,
            feature_family=feature_family,
            seed=seed + fold,
        )
        output[validation] = predict_calibrated_fusion(
            subset_probabilities(raw_probabilities, validation, judges),
            parameters,
            model,
        )
        records.append(
            {
                "fold": fold,
                "training_record_count": int(training.sum()),
                "validation_record_count": int(validation.sum()),
                "training_component_count": len(set(group_values[training])),
                "validation_component_count": len(set(group_values[validation])),
                "calibrators": parameters,
                "model": model.to_dict(),
            }
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Fusion cross-fitting left non-finite predictions")
    return output, records


def select_hyperparameters(
    *,
    raw_probabilities: Mapping[str, Sequence[float]],
    targets: np.ndarray,
    groups: Sequence[str],
    judges: Sequence[str],
    calibrator_methods: Mapping[str, str],
    feature_family: str,
    candidates: Sequence[Mapping[str, Any]],
    fold_count: int,
    seed: int,
) -> dict[str, Any]:
    folds = make_group_folds(targets, groups, n_splits=fold_count, seed=seed)
    candidate_results = []
    for index, candidate in enumerate(candidates):
        predictions, _ = cross_fitted_predictions(
            raw_probabilities=raw_probabilities,
            targets=targets,
            groups=groups,
            folds=folds,
            judges=judges,
            calibrator_methods=calibrator_methods,
            feature_family=feature_family,
            hyperparameters=candidate,
            seed=seed + 1000 + index * 20,
        )
        fold_losses = []
        for fold in range(fold_count):
            members = folds == fold
            fold_losses.append(
                cluster_averaged_log_loss(
                    targets[members],
                    predictions[members],
                    np.asarray(groups)[members].tolist(),
                )
            )
        candidate_results.append(
            {
                "hyperparameters": dict(candidate),
                "mean_fold_log_loss": float(np.mean(fold_losses)),
                "standard_error_fold_log_loss": float(
                    np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses))
                ),
                "fold_log_losses": fold_losses,
                "overall_cluster_log_loss": cluster_averaged_log_loss(
                    targets, predictions, groups
                ),
            }
        )
    best = min(
        candidate_results,
        key=lambda value: (
            value["mean_fold_log_loss"],
            hyperparameter_key(value["hyperparameters"]),
        ),
    )
    limit = (
        float(best["mean_fold_log_loss"])
        + float(best["standard_error_fold_log_loss"])
    )
    eligible = [
        value
        for value in candidate_results
        if float(value["mean_fold_log_loss"]) <= limit + 1e-12
    ]
    selected = min(
        eligible,
        key=lambda value: hyperparameter_key(value["hyperparameters"]),
    )
    return {
        "selected_hyperparameters": selected["hyperparameters"],
        "one_standard_error_limit": limit,
        "minimum_loss_candidate": best["hyperparameters"],
        "candidate_results": candidate_results,
        "fold_assignments": folds,
    }


def nested_oof_predictions(
    *,
    raw_probabilities: Mapping[str, Sequence[float]],
    targets: np.ndarray,
    groups: Sequence[str],
    judges: Sequence[str],
    calibrator_methods: Mapping[str, str],
    feature_family: str,
    candidates: Sequence[Mapping[str, Any]],
    outer_fold_count: int,
    inner_fold_count: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    outer_folds = make_group_folds(
        targets, groups, n_splits=outer_fold_count, seed=seed
    )
    group_values = np.asarray([str(value) for value in groups])
    output = np.full(len(targets), np.nan, dtype=np.float64)
    records = []
    for outer_fold in range(outer_fold_count):
        validation = outer_folds == outer_fold
        training = ~validation
        inner = select_hyperparameters(
            raw_probabilities=subset_probabilities(
                raw_probabilities, training, judges
            ),
            targets=targets[training],
            groups=group_values[training].tolist(),
            judges=judges,
            calibrator_methods=calibrator_methods,
            feature_family=feature_family,
            candidates=candidates,
            fold_count=inner_fold_count,
            seed=seed + 10000 + outer_fold * 1000,
        )
        parameters, model = fit_calibrated_fusion(
            subset_probabilities(raw_probabilities, training, judges),
            targets[training],
            group_values[training].tolist(),
            judges,
            calibrator_methods,
            inner["selected_hyperparameters"],
            feature_family=feature_family,
            seed=seed + 20000 + outer_fold,
        )
        output[validation] = predict_calibrated_fusion(
            subset_probabilities(raw_probabilities, validation, judges),
            parameters,
            model,
        )
        records.append(
            {
                "outer_fold": outer_fold,
                "training_record_count": int(training.sum()),
                "validation_record_count": int(validation.sum()),
                "selected_hyperparameters": inner[
                    "selected_hyperparameters"
                ],
                "inner_one_standard_error_limit": inner[
                    "one_standard_error_limit"
                ],
            }
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Nested fusion CV left non-finite predictions")
    return output, records


def component_bootstrap_delta(
    *,
    targets: np.ndarray,
    groups: Sequence[str],
    left_probabilities: np.ndarray,
    left_threshold: float,
    right_probabilities: np.ndarray,
    right_threshold: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[str(group)].append(index)
    component_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    macro_deltas = np.empty(replicates, dtype=np.float64)
    recall_deltas = np.empty(replicates, dtype=np.float64)
    change_rates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(component_ids, size=len(component_ids), replace=True)
        indices = np.concatenate(
            [np.asarray(grouped[value], dtype=np.int64) for value in sampled]
        )
        left = left_probabilities[indices] >= left_threshold
        right = right_probabilities[indices] >= right_threshold
        left_metrics = classification_metrics(targets[indices], left)
        right_metrics = classification_metrics(targets[indices], right)
        macro_deltas[replicate] = (
            float(left_metrics["macro_f1"])
            - float(right_metrics["macro_f1"])
        )
        recall_deltas[replicate] = (
            float(left_metrics["unsafe_recall"])
            - float(right_metrics["unsafe_recall"])
        )
        change_rates[replicate] = float(np.mean(left != right))
    return {
        "replicates": replicates,
        "positive_macro_f1_fraction": float(np.mean(macro_deltas > 0)),
        "nonnegative_unsafe_recall_fraction": float(
            np.mean(recall_deltas >= 0)
        ),
        "macro_f1_delta_ci95": np.quantile(
            macro_deltas, [0.025, 0.975]
        ).tolist(),
        "unsafe_recall_delta_ci95": np.quantile(
            recall_deltas, [0.025, 0.975]
        ).tolist(),
        "prediction_change_rate_ci95": np.quantile(
            change_rates, [0.025, 0.975]
        ).tolist(),
    }


def pattern_support(
    fit_report: Mapping[str, Any], select_report: Mapping[str, Any]
) -> dict[str, Any]:
    fit_seen = {
        pattern
        for pattern, value in fit_report["patterns"].items()
        if int(value["record_count"]) > 0
    }
    unseen_records = sum(
        int(value["record_count"])
        for pattern, value in select_report["patterns"].items()
        if pattern not in fit_seen
    )
    unseen_components = sum(
        int(value["component_count"])
        for pattern, value in select_report["patterns"].items()
        if pattern not in fit_seen
    )
    total_records = sum(
        int(value["record_count"])
        for value in select_report["patterns"].values()
    )
    total_components = sum(
        int(value["component_count"])
        for value in select_report["patterns"].values()
    )
    return {
        "fit": fit_report,
        "select": select_report,
        "select_unseen_record_count": unseen_records,
        "select_unseen_record_rate": (
            0.0 if total_records == 0 else unseen_records / total_records
        ),
        "select_unseen_pattern_component_count_sum": unseen_components,
        "select_pattern_component_count_sum": total_components,
    }


def selection_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# SF3 Canonical-Only Fusion Selection",
        "",
        "The three candidate consistency views were rejected before SF3. "
        "This analysis uses canonical outputs only.",
        "",
        "## Candidate families",
        "",
        "| Family | Penalty | C | L1 ratio | Select threshold | Macro-F1 | Unsafe recall | Specificity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("F1", "F2"):
        value = report["families"][family]
        hp = value["selected_hyperparameters"]
        threshold = value["select_threshold"]
        metrics = threshold.get("metrics", {})
        lines.append(
            "| {family} | {penalty} | {c:.3g} | {ratio} | {threshold} | "
            "{f1} | {recall} | {specificity} |".format(
                family=family,
                penalty=hp["penalty"],
                c=float(hp["C"]),
                ratio=("--" if hp.get("l1_ratio") is None else hp["l1_ratio"]),
                threshold=(
                    "infeasible"
                    if not threshold["feasible"]
                    else f"{float(threshold['threshold']):.6f}"
                ),
                f1=("--" if not metrics else f"{float(metrics['macro_f1']):.4f}"),
                recall=("--" if not metrics else f"{float(metrics['unsafe_recall']):.4f}"),
                specificity=("--" if not metrics else f"{float(metrics['safe_specificity']):.4f}"),
            )
        )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            f"- Selected family: `{report['selected_family']}`",
            f"- Strongest simple reference: `{report['strongest_simple_baseline']}`",
            f"- Development threshold: `{report['development_threshold']['threshold'] if report['development_threshold'].get('feasible') else 'infeasible'}`",
            f"- Stop gate passed: `{report['stop_gate']['passed']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "stage": "SF3"})
    try:
        config = load_json(args.selection_config.resolve())
        if config.get("feature_scope") != "canonical_only":
            raise RuntimeError("SF3 must use the frozen canonical-only scope")
        screening = load_json(args.screening_decision.resolve())
        if (
            screening.get("frozen_feature_scope") != "canonical_only"
            or screening.get("run_additional_consistency_gpu") is not False
        ):
            raise RuntimeError("Consistency screening is not frozen for SF3")
        calibrators = load_json(args.calibrators.resolve())
        judges = tuple(str(value) for value in config["judge_order"])
        if tuple(calibrators["judge_order"]) != judges:
            raise RuntimeError("Judge order differs from the calibrated pool")
        methods = {
            judge: str(calibrators["selected_methods"][judge])
            for judge in judges
        }
        fit = load_split(
            args.fit_manifest.resolve(), args.soft_root.resolve(), "fit"
        )
        select = load_split(
            args.select_manifest.resolve(), args.soft_root.resolve(), "select"
        )
        if set(fit["components"]) & set(select["components"]):
            raise RuntimeError("D_fit and D_select share components")
        candidates = hyperparameter_grid(config)

        select_baselines = evaluate_baselines(
            select["targets"], select["raw_probabilities"], judges
        )
        strongest = select_strongest_baseline(select_baselines)
        reference_metrics = select_baselines[strongest]["metrics"]
        minimum_recall = max(
            0.0,
            float(reference_metrics["unsafe_recall"])
            - float(
                config["threshold_constraints"][
                    "unsafe_recall_margin_below_reference"
                ]
            ),
        )
        minimum_specificity = max(
            0.0,
            float(reference_metrics["safe_specificity"])
            - float(
                config["threshold_constraints"][
                    "safe_specificity_margin_below_reference"
                ]
            ),
        )

        family_results: dict[str, Any] = {}
        select_predictions: dict[str, np.ndarray] = {}
        for offset, family in enumerate(config["feature_families"]):
            global_selection = select_hyperparameters(
                raw_probabilities=fit["raw_probabilities"],
                targets=fit["targets"],
                groups=fit["components"],
                judges=judges,
                calibrator_methods=methods,
                feature_family=family,
                candidates=candidates,
                fold_count=int(config["outer_fold_count"]),
                seed=int(config["seed"]) + offset * 100000,
            )
            nested_oof, nested_records = nested_oof_predictions(
                raw_probabilities=fit["raw_probabilities"],
                targets=fit["targets"],
                groups=fit["components"],
                judges=judges,
                calibrator_methods=methods,
                feature_family=family,
                candidates=candidates,
                outer_fold_count=int(config["outer_fold_count"]),
                inner_fold_count=int(config["inner_fold_count"]),
                seed=int(config["seed"]) + 500000 + offset * 100000,
            )
            parameters, model = fit_calibrated_fusion(
                fit["raw_probabilities"],
                fit["targets"],
                fit["components"],
                judges,
                methods,
                global_selection["selected_hyperparameters"],
                feature_family=family,
                seed=int(config["seed"]) + 900000 + offset,
            )
            predictions = predict_calibrated_fusion(
                select["raw_probabilities"], parameters, model
            )
            select_predictions[family] = predictions
            temporary_threshold = select_constrained_threshold(
                select["targets"],
                predictions,
                minimum_unsafe_recall=minimum_recall,
                minimum_safe_specificity=minimum_specificity,
            )
            family_results[family] = {
                "selected_hyperparameters": global_selection[
                    "selected_hyperparameters"
                ],
                "hyperparameter_selection": jsonable(global_selection),
                "nested_fit_oof_cluster_log_loss": cluster_averaged_log_loss(
                    fit["targets"], nested_oof, fit["components"]
                ),
                "nested_fit_records": nested_records,
                "fit_final_calibrators": parameters,
                "fit_final_model": model.to_dict(),
                "select_cluster_log_loss": cluster_averaged_log_loss(
                    select["targets"], predictions, select["components"]
                ),
                "select_threshold": temporary_threshold,
            }

        f1_threshold = family_results["F1"]["select_threshold"]
        f2_threshold = family_results["F2"]["select_threshold"]
        if not f1_threshold["feasible"]:
            selected_family = "none"
            complexity_bootstrap = None
            complexity_reason = "F1 threshold constraints were infeasible"
        elif not f2_threshold["feasible"]:
            selected_family = "F1"
            complexity_bootstrap = None
            complexity_reason = "F2 threshold constraints were infeasible"
        else:
            f1_metrics = f1_threshold["metrics"]
            f2_metrics = f2_threshold["metrics"]
            f1_to_f2_macro = (
                float(f2_metrics["macro_f1"])
                - float(f1_metrics["macro_f1"])
            )
            f1_to_f2_recall = (
                float(f2_metrics["unsafe_recall"])
                - float(f1_metrics["unsafe_recall"])
            )
            complexity_bootstrap = component_bootstrap_delta(
                targets=select["targets"],
                groups=select["components"],
                left_probabilities=select_predictions["F2"],
                left_threshold=float(f2_threshold["threshold"]),
                right_probabilities=select_predictions["F1"],
                right_threshold=float(f1_threshold["threshold"]),
                replicates=int(
                    config["complexity_gate"]["bootstrap_replicates"]
                ),
                seed=int(config["seed"]) + 2000000,
            )
            gain_gate = (
                f1_to_f2_macro
                >= float(config["complexity_gate"]["minimum_macro_f1_gain"])
                or f1_to_f2_recall
                >= float(
                    config["complexity_gate"][
                        "minimum_unsafe_recall_gain"
                    ]
                )
            )
            stability_gate = (
                complexity_bootstrap["positive_macro_f1_fraction"]
                >= float(
                    config["complexity_gate"][
                        "minimum_positive_component_bootstrap_fraction"
                    ]
                )
            )
            selected_family = "F2" if gain_gate and stability_gate else "F1"
            complexity_reason = (
                "F2 passed the frozen gain and bootstrap gates"
                if selected_family == "F2"
                else "F2 did not pass both frozen complexity gates"
            )

        combined_targets = np.concatenate(
            [fit["targets"], select["targets"]]
        )
        combined_groups = [
            f"fit:{value}" for value in fit["components"]
        ] + [f"select:{value}" for value in select["components"]]
        combined_raw = {
            judge: np.concatenate(
                [
                    fit["raw_probabilities"][judge],
                    select["raw_probabilities"][judge],
                ]
            )
            for judge in judges
        }
        combined_baselines = evaluate_baselines(
            combined_targets, combined_raw, judges
        )
        selected_hyperparameters = (
            None
            if selected_family == "none"
            else family_results[selected_family]["selected_hyperparameters"]
        )
        development_predictions = None
        development_records = []
        development_threshold = {"feasible": False}
        final_parameters = None
        final_model = None
        equivalence = {"equivalent_rules": [], "comparisons": {}}
        judge_pool_audit: dict[str, Any] = {
            "passed": False,
            "reason": "No feasible F1 family",
        }
        if selected_family != "none":
            development_folds = make_group_folds(
                combined_targets,
                combined_groups,
                n_splits=int(config["outer_fold_count"]),
                seed=int(config["seed"]) + 3000000,
            )
            development_predictions, development_records = (
                cross_fitted_predictions(
                    raw_probabilities=combined_raw,
                    targets=combined_targets,
                    groups=combined_groups,
                    folds=development_folds,
                    judges=judges,
                    calibrator_methods=methods,
                    feature_family=selected_family,
                    hyperparameters=selected_hyperparameters,
                    seed=int(config["seed"]) + 3100000,
                )
            )
            development_reference = combined_baselines[strongest]["metrics"]
            development_threshold = select_constrained_threshold(
                combined_targets,
                development_predictions,
                minimum_unsafe_recall=max(
                    0.0,
                    float(development_reference["unsafe_recall"])
                    - float(
                        config["threshold_constraints"][
                            "unsafe_recall_margin_below_reference"
                        ]
                    ),
                ),
                minimum_safe_specificity=max(
                    0.0,
                    float(development_reference["safe_specificity"])
                    - float(
                        config["threshold_constraints"][
                            "safe_specificity_margin_below_reference"
                        ]
                    ),
                ),
            )
            final_parameters, final_model = fit_calibrated_fusion(
                combined_raw,
                combined_targets,
                combined_groups,
                judges,
                methods,
                selected_hyperparameters,
                feature_family=selected_family,
                seed=int(config["seed"]) + 3200000,
            )
            if development_threshold["feasible"]:
                baseline_values = baseline_predictions(combined_raw, judges)
                equivalence = rule_equivalence(
                    development_predictions
                    >= float(development_threshold["threshold"]),
                    baseline_values,
                )
                marginal = {}
                for offset, omitted in enumerate(judges):
                    reduced_judges = tuple(
                        judge for judge in judges if judge != omitted
                    )
                    reduced_predictions, _ = cross_fitted_predictions(
                        raw_probabilities=combined_raw,
                        targets=combined_targets,
                        groups=combined_groups,
                        folds=development_folds,
                        judges=reduced_judges,
                        calibrator_methods=methods,
                        feature_family=selected_family,
                        hyperparameters=selected_hyperparameters,
                        seed=int(config["seed"]) + 3300000 + offset * 10000,
                    )
                    threshold = float(development_threshold["threshold"])
                    full_binary = development_predictions >= threshold
                    reduced_binary = reduced_predictions >= threshold
                    full_metrics = classification_metrics(
                        combined_targets, full_binary
                    )
                    reduced_metrics = classification_metrics(
                        combined_targets, reduced_binary
                    )
                    bootstrap = component_bootstrap_delta(
                        targets=combined_targets,
                        groups=combined_groups,
                        left_probabilities=development_predictions,
                        left_threshold=threshold,
                        right_probabilities=reduced_predictions,
                        right_threshold=threshold,
                        replicates=int(
                            config["complexity_gate"]["bootstrap_replicates"]
                        ),
                        seed=int(config["seed"]) + 3400000 + offset,
                    )
                    change_rate = float(np.mean(full_binary != reduced_binary))
                    delta = {
                        "macro_f1_delta_full_minus_reduced": (
                            float(full_metrics["macro_f1"])
                            - float(reduced_metrics["macro_f1"])
                        ),
                        "unsafe_recall_delta_full_minus_reduced": (
                            float(full_metrics["unsafe_recall"])
                            - float(reduced_metrics["unsafe_recall"])
                        ),
                        "hard_prediction_change_rate": change_rate,
                        "bootstrap": bootstrap,
                    }
                    delta["gate"] = {
                        "changes_at_least_one_percent": (
                            change_rate
                            >= float(
                                config["judge_marginal_gate"][
                                    "minimum_hard_prediction_change_rate"
                                ]
                            )
                        ),
                        "positive_in_required_bootstraps": (
                            bootstrap["positive_macro_f1_fraction"]
                            >= float(
                                config["judge_marginal_gate"][
                                    "minimum_positive_component_bootstrap_fraction"
                                ]
                            )
                        ),
                        "nonnegative_point_unsafe_recall": (
                            delta["unsafe_recall_delta_full_minus_reduced"] >= 0
                        ),
                    }
                    delta["gate"]["passed"] = all(delta["gate"].values())
                    marginal[omitted] = delta
                judge_pool_audit = {
                    "schema": "softfusion-sf3-judge-pool-audit-v1",
                    "marginal_contributions": marginal,
                    "passed": all(value["gate"]["passed"] for value in marginal.values()),
                }

        fit_patterns = vote_pattern_report(
            fit["raw_probabilities"], judges, fit["components"]
        )
        select_patterns = vote_pattern_report(
            select["raw_probabilities"], judges, select["components"]
        )
        vote_patterns = pattern_support(fit_patterns, select_patterns)

        majority_metrics = select_baselines["majority"]["metrics"]
        selected_select_threshold = (
            {"feasible": False}
            if selected_family == "none"
            else family_results[selected_family]["select_threshold"]
        )
        selected_select_metrics = selected_select_threshold.get("metrics", {})
        stop_gate = {
            "canonical_only_scope_frozen": True,
            "selection_threshold_feasible": bool(
                selected_select_threshold.get("feasible")
            ),
            "development_threshold_feasible": bool(
                development_threshold.get("feasible")
            ),
            "not_equivalent_to_simple_rule": (
                bool(development_threshold.get("feasible"))
                and not equivalence["equivalent_rules"]
            ),
            "fixed_three_judge_pool_passed": bool(
                judge_pool_audit.get("passed")
            ),
            "selection_macro_f1_exceeds_majority": (
                bool(selected_select_metrics)
                and float(selected_select_metrics["macro_f1"])
                > float(majority_metrics["macro_f1"])
            ),
        }
        stop_gate["passed"] = all(stop_gate.values())

        family_selection = {
            "schema": "softfusion-family-selection-v1",
            "families": jsonable(family_results),
            "selected_family": selected_family,
            "complexity_reason": complexity_reason,
            "complexity_bootstrap": jsonable(complexity_bootstrap),
            "strongest_simple_baseline": strongest,
            "select_baselines": {
                name: value["metrics"] for name, value in select_baselines.items()
            },
            "development_threshold": jsonable(development_threshold),
            "rule_equivalence": equivalence,
            "stop_gate": stop_gate,
        }

        baseline_selection = {
            "schema": "softfusion-baseline-selection-v1",
            "primary_baseline": "majority",
            "strongest_simple_baseline": strongest,
            "selection_metrics": {
                name: value["metrics"] for name, value in select_baselines.items()
            },
            "development_metrics": {
                name: value["metrics"] for name, value in combined_baselines.items()
            },
            "selection_rule": (
                "maximum Macro-F1, then unsafe recall, specificity, and name"
            ),
        }

        feature_schema = {
            "schema": "softfusion-feature-schema-v1",
            "scope": "canonical_only",
            "selected_family": selected_family,
            "judge_order": list(judges),
            "feature_names": (
                [] if final_model is None else list(final_model.feature_names)
            ),
            "excluded_features": [
                "multi_view_mean",
                "multi_view_standard_deviation",
                "multi_view_direction_agreement",
            ],
        }

        fusion_config = {
            "schema": "softfusion-fusion-config-v1",
            "status": "frozen" if stop_gate["passed"] else "blocked",
            "seed": int(config["seed"]),
            "feature_scope": "canonical_only",
            "judge_order": list(judges),
            "adapter_versions": fit["adapter_versions"],
            "calibrator_methods": methods,
            "final_development_calibrators": final_parameters,
            "model": None if final_model is None else final_model.to_dict(),
            "threshold": (
                None
                if not development_threshold.get("feasible")
                else float(development_threshold["threshold"])
            ),
            "threshold_metrics": development_threshold.get("metrics"),
            "confirmatory_labels_accessed": False,
        }
        fusion_config["config_sha256"] = stable_json_sha256(fusion_config)

        source_paths = [
            args.selection_config.resolve(),
            args.screening_decision.resolve(),
            args.calibrators.resolve(),
            args.fit_manifest.resolve(),
            args.select_manifest.resolve(),
            Path(__file__).resolve(),
            SRC_ROOT / "multijudge" / "fusion.py",
            SRC_ROOT / "multijudge" / "baselines.py",
        ]
        provenance = {
            "schema": "softfusion-sf3-provenance-v1",
            "source_files_sha256": {
                str(path): sha256_file(path) for path in source_paths
            },
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "confirmatory_labels_accessed": False,
            "confirmatory_predictions_generated": False,
        }
        freeze_record = {
            "schema": "softfusion-freeze-record-v1",
            "status": "frozen" if stop_gate["passed"] else "blocked",
            "fusion_config_sha256": fusion_config["config_sha256"],
            "selection_config_sha256": sha256_file(
                args.selection_config.resolve()
            ),
            "screening_decision_sha256": sha256_file(
                args.screening_decision.resolve()
            ),
            "fit_record_count": len(fit["sample_ids"]),
            "select_record_count": len(select["sample_ids"]),
            "development_component_count": len(set(combined_groups)),
            "confirmatory_labels_accessed": False,
            "confirmatory_run_permitted": bool(stop_gate["passed"]),
            "stop_gate": stop_gate,
        }

        write_json(output_dir / "candidate_selection.json", family_selection)
        write_json(output_dir / "fusion_config.json", fusion_config)
        write_json(output_dir / "baseline_selection.json", baseline_selection)
        write_json(output_dir / "judge_pool_audit.json", judge_pool_audit)
        write_json(output_dir / "feature_schema.json", feature_schema)
        write_json(output_dir / "vote_patterns.json", vote_patterns)
        write_json(output_dir / "provenance.json", provenance)
        write_json(output_dir / "freeze_record.json", freeze_record)
        if development_predictions is not None:
            rows = []
            sample_ids = fit["sample_ids"] + select["sample_ids"]
            splits = ["fit"] * len(fit["sample_ids"]) + ["select"] * len(
                select["sample_ids"]
            )
            for index, sample_id in enumerate(sample_ids):
                rows.append(
                    {
                        "schema": "softfusion-development-oof-record-v1",
                        "sample_id": sample_id,
                        "split": splits[index],
                        "component_id": combined_groups[index],
                        "expected_label": (
                            "unsafe" if combined_targets[index] else "safe"
                        ),
                        "p_unsafe": float(development_predictions[index]),
                        "predicted_label": (
                            "unsafe"
                            if development_threshold.get("feasible")
                            and development_predictions[index]
                            >= float(development_threshold["threshold"])
                            else "safe"
                        ),
                    }
                )
            write_jsonl(output_dir / "development_oof_predictions.jsonl", rows)
        (output_dir / "selection_report.md").write_text(
            selection_markdown(family_selection), encoding="utf-8"
        )
        write_json(
            status_path,
            {
                "state": "completed",
                "stage": "SF3",
                "selected_family": selected_family,
                "stop_gate_passed": stop_gate["passed"],
                "confirmatory_run_permitted": stop_gate["passed"],
            },
        )
        print(
            json.dumps(
                {
                    "state": "completed",
                    "selected_family": selected_family,
                    "strongest_simple_baseline": strongest,
                    "stop_gate_passed": stop_gate["passed"],
                    "confirmatory_run_permitted": stop_gate["passed"],
                }
            )
        )
    except Exception as exc:
        write_json(
            status_path,
            {
                "state": "failed",
                "stage": "SF3",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    softfusion = V15_ROOT / "results" / "mmds"
    config_root = SCRIPT_ROOT / "configs"
    parser = argparse.ArgumentParser(
        description="Fit and freeze the canonical-only SF3 fusion model."
    )
    parser.add_argument(
        "--fit-manifest", type=Path, default=softfusion / "data" / "fit.jsonl"
    )
    parser.add_argument(
        "--select-manifest",
        type=Path,
        default=softfusion / "data" / "select.jsonl",
    )
    parser.add_argument(
        "--soft-root", type=Path, default=softfusion / "qualification"
    )
    parser.add_argument(
        "--calibrators",
        type=Path,
        default=softfusion / "calibration" / "calibrators.json",
    )
    parser.add_argument(
        "--screening-decision",
        type=Path,
        default=softfusion / "sf2-consistency" / "screening_decision.json",
    )
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=config_root / "softfusion_model_selection_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=softfusion / "fit"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
