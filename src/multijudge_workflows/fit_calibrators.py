from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from multijudge.calibration import (  # noqa: E402
    apply_calibrator,
    calibration_metrics,
    cluster_averaged_log_loss,
    fit_final_probe,
    make_group_folds,
    nested_probe_predictions,
    pairwise_dependence,
    predict_final_probe,
    select_calibrator,
    stable_json_sha256,
)
from multijudge.metrics import binary_classification_metrics  # noqa: E402


JUDGES = ("qwen", "safework", "minicpm")
SEED = 20260729
FOLD_COUNT = 5
BOOTSTRAP_REPLICATES = 500
PROBE_L2_STRENGTH = 0.1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_split(
    manifest_path: Path,
    soft_root: Path,
    split: str,
) -> dict[str, Any]:
    manifest = load_jsonl(manifest_path)
    if not manifest:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    manifest_by_id = {str(value["sample_id"]): value for value in manifest}
    if len(manifest_by_id) != len(manifest):
        raise RuntimeError(f"Duplicate sample IDs in {manifest_path}")
    ordered_ids = [str(value["sample_id"]) for value in manifest]
    raw_probabilities: dict[str, np.ndarray] = {}
    input_files = [manifest_path]
    adapter_versions = {}
    for judge in JUDGES:
        path = soft_root / judge / split / "judgments.jsonl"
        records = load_jsonl(path)
        input_files.append(path)
        by_id = {str(value["sample_id"]): value for value in records}
        if len(by_id) != len(records) or set(by_id) != set(ordered_ids):
            raise RuntimeError(
                f"{judge}/{split} does not exactly cover the frozen manifest"
            )
        probabilities = []
        versions = set()
        for sample_id in ordered_ids:
            record = by_id[sample_id]
            manifest_record = manifest_by_id[sample_id]
            if str(record["expected_label"]) != str(
                manifest_record["assistant_label"]
            ):
                raise RuntimeError(
                    f"Ground-truth mismatch for {judge}/{sample_id}"
                )
            if str(record["conversation_sha256"]) != str(
                manifest_record["conversation_sha256"]
            ):
                raise RuntimeError(
                    f"Conversation mismatch for {judge}/{sample_id}"
                )
            probabilities.append(float(record["soft_evidence"]["p_unsafe"]))
            versions.add(str(record["judge_service"]["adapter_version"]))
        if len(versions) != 1:
            raise RuntimeError(f"Mixed adapter versions for {judge}/{split}")
        adapter_versions[judge] = next(iter(versions))
        values = np.asarray(probabilities, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite probabilities for {judge}/{split}")
        raw_probabilities[judge] = values

    return {
        "sample_ids": ordered_ids,
        "components": [str(value["component_id"]) for value in manifest],
        "targets": np.asarray(
            [
                1 if str(value["assistant_label"]) == "unsafe" else 0
                for value in manifest
            ],
            dtype=np.int64,
        ),
        "raw_probabilities": raw_probabilities,
        "manifest": manifest,
        "adapter_versions": adapter_versions,
        "input_files": input_files,
    }


def classification_summary(
    targets: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    expected = ["unsafe" if int(value) else "safe" for value in targets]
    predicted = [
        "unsafe" if float(value) >= 0.5 else "safe"
        for value in probabilities
    ]
    return binary_classification_metrics(expected, predicted)


def method_metrics(
    targets: Sequence[int],
    probabilities: Sequence[float],
    groups: Sequence[str],
) -> dict[str, Any]:
    return {
        "classification_at_0_5": classification_summary(
            targets, probabilities
        ),
        "calibration": calibration_metrics(
            targets, probabilities, groups
        ),
    }


def confusion_counts(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    predicted = probabilities >= 0.5
    positive = targets == 1
    return np.asarray(
        [
            np.logical_and(positive, predicted).sum(),
            np.logical_and(~positive, ~predicted).sum(),
            np.logical_and(~positive, predicted).sum(),
            np.logical_and(positive, ~predicted).sum(),
        ],
        dtype=np.int64,
    )


def macro_f1_from_counts(counts: np.ndarray) -> float:
    unsafe_tp, safe_tp, unsafe_fp, unsafe_fn = [
        int(value) for value in counts
    ]
    unsafe_denominator = 2 * unsafe_tp + unsafe_fp + unsafe_fn
    safe_denominator = 2 * safe_tp + unsafe_fn + unsafe_fp
    unsafe_f1 = (
        0.0
        if unsafe_denominator == 0
        else 2 * unsafe_tp / unsafe_denominator
    )
    safe_f1 = (
        0.0 if safe_denominator == 0 else 2 * safe_tp / safe_denominator
    )
    return (unsafe_f1 + safe_f1) / 2


def unsafe_recall_from_counts(counts: np.ndarray) -> float:
    unsafe_tp, _, _, unsafe_fn = [int(value) for value in counts]
    denominator = unsafe_tp + unsafe_fn
    return 0.0 if denominator == 0 else unsafe_tp / denominator


def cluster_bootstrap_marginal(
    *,
    targets: np.ndarray,
    groups: Sequence[str],
    full_probabilities: np.ndarray,
    reduced_probabilities: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped_indices[str(group)].append(index)
    component_ids = sorted(grouped_indices)
    component_loss_delta = []
    component_full_confusion = []
    component_reduced_confusion = []
    component_change_count = []
    component_record_count = []
    component_abs_change = []
    for component_id in component_ids:
        indices = np.asarray(grouped_indices[component_id], dtype=np.int64)
        y = targets[indices]
        full = np.clip(full_probabilities[indices], 1e-6, 1 - 1e-6)
        reduced = np.clip(reduced_probabilities[indices], 1e-6, 1 - 1e-6)
        full_loss = -(y * np.log(full) + (1 - y) * np.log(1 - full))
        reduced_loss = -(
            y * np.log(reduced) + (1 - y) * np.log(1 - reduced)
        )
        component_loss_delta.append(float(np.mean(reduced_loss - full_loss)))
        component_full_confusion.append(confusion_counts(y, full))
        component_reduced_confusion.append(confusion_counts(y, reduced))
        component_change_count.append(
            int(((full >= 0.5) != (reduced >= 0.5)).sum())
        )
        component_record_count.append(len(indices))
        component_abs_change.append(float(np.mean(np.abs(full - reduced))))

    loss_values = np.asarray(component_loss_delta, dtype=np.float64)
    full_counts = np.stack(component_full_confusion)
    reduced_counts = np.stack(component_reduced_confusion)
    changes = np.asarray(component_change_count, dtype=np.float64)
    record_counts = np.asarray(component_record_count, dtype=np.float64)
    absolute_changes = np.asarray(component_abs_change, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot_loss = np.empty(replicates, dtype=np.float64)
    boot_macro_f1 = np.empty(replicates, dtype=np.float64)
    boot_unsafe_recall = np.empty(replicates, dtype=np.float64)
    boot_change = np.empty(replicates, dtype=np.float64)
    boot_abs_change = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(component_ids), len(component_ids))
        boot_loss[replicate] = float(loss_values[sampled].mean())
        sampled_full = full_counts[sampled].sum(axis=0)
        sampled_reduced = reduced_counts[sampled].sum(axis=0)
        boot_macro_f1[replicate] = (
            macro_f1_from_counts(sampled_full)
            - macro_f1_from_counts(sampled_reduced)
        )
        boot_unsafe_recall[replicate] = (
            unsafe_recall_from_counts(sampled_full)
            - unsafe_recall_from_counts(sampled_reduced)
        )
        boot_change[replicate] = (
            changes[sampled].sum() / record_counts[sampled].sum()
        )
        boot_abs_change[replicate] = float(
            absolute_changes[sampled].mean()
        )
    return {
        "schema": "softfusion-cluster-bootstrap-marginal-v1",
        "seed": seed,
        "replicates": replicates,
        "component_count": len(component_ids),
        "positive_log_loss_fraction": float(np.mean(boot_loss > 0)),
        "positive_macro_f1_fraction": float(np.mean(boot_macro_f1 > 0)),
        "nonnegative_unsafe_recall_fraction": float(
            np.mean(boot_unsafe_recall >= 0)
        ),
        "log_loss_delta_ci95": [
            float(value) for value in np.quantile(boot_loss, [0.025, 0.975])
        ],
        "macro_f1_delta_ci95": [
            float(value)
            for value in np.quantile(boot_macro_f1, [0.025, 0.975])
        ],
        "unsafe_recall_delta_ci95": [
            float(value)
            for value in np.quantile(boot_unsafe_recall, [0.025, 0.975])
        ],
        "hard_prediction_change_rate_ci95": [
            float(value)
            for value in np.quantile(boot_change, [0.025, 0.975])
        ],
        "mean_absolute_probability_change_ci95": [
            float(value)
            for value in np.quantile(boot_abs_change, [0.025, 0.975])
        ],
    }


def build_probe_audit(
    *,
    fit: dict[str, Any],
    select: dict[str, Any],
    folds: np.ndarray,
    calibrator_methods: dict[str, str],
    final_calibrators: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fit_targets = fit["targets"]
    fit_groups = fit["components"]
    select_targets = select["targets"]
    select_groups = select["components"]
    combined_targets = np.concatenate([fit_targets, select_targets])
    combined_groups = [
        f"fit:{value}" for value in fit_groups
    ] + [f"select:{value}" for value in select_groups]

    model_predictions: dict[str, np.ndarray] = {}
    model_details: dict[str, Any] = {}
    included_sets = {"all": list(JUDGES)}
    for omitted in JUDGES:
        included_sets[f"without_{omitted}"] = [
            judge for judge in JUDGES if judge != omitted
        ]
    for name, included in included_sets.items():
        fit_oof, fold_records = nested_probe_predictions(
            fit["raw_probabilities"],
            fit_targets,
            fit_groups,
            folds,
            calibrator_methods,
            included,
            l2_strength=PROBE_L2_STRENGTH,
        )
        probe = fit_final_probe(
            fit["raw_probabilities"],
            fit_targets,
            fit_groups,
            final_calibrators,
            included,
            l2_strength=PROBE_L2_STRENGTH,
        )
        select_predictions = predict_final_probe(
            probe,
            select["raw_probabilities"],
            final_calibrators,
        )
        combined = np.concatenate([fit_oof, select_predictions])
        model_predictions[name] = combined
        model_details[name] = {
            "included_judges": included,
            "probe_role": (
                "fixed_l2_preliminary_complementarity_probe_not_final_fusion"
            ),
            "l2_strength": PROBE_L2_STRENGTH,
            "fit_oof_metrics": method_metrics(
                fit_targets, fit_oof, fit_groups
            ),
            "select_frozen_metrics": method_metrics(
                select_targets, select_predictions, select_groups
            ),
            "combined_development_metrics": method_metrics(
                combined_targets, combined, combined_groups
            ),
            "fold_records": fold_records,
            "final_fit_probe": probe.to_dict(),
        }

    full = model_predictions["all"]
    full_classification = classification_summary(combined_targets, full)
    full_log_loss = cluster_averaged_log_loss(
        combined_targets, full, combined_groups
    )
    marginal = {}
    for offset, omitted in enumerate(JUDGES):
        reduced = model_predictions[f"without_{omitted}"]
        reduced_classification = classification_summary(
            combined_targets, reduced
        )
        reduced_log_loss = cluster_averaged_log_loss(
            combined_targets, reduced, combined_groups
        )
        hard_change_rate = float(
            np.mean((full >= 0.5) != (reduced >= 0.5))
        )
        bootstrap = cluster_bootstrap_marginal(
            targets=combined_targets,
            groups=combined_groups,
            full_probabilities=full,
            reduced_probabilities=reduced,
            replicates=BOOTSTRAP_REPLICATES,
            seed=SEED + 100 + offset,
        )
        delta = {
            "log_loss_delta_reduced_minus_full": (
                reduced_log_loss - full_log_loss
            ),
            "macro_f1_delta_full_minus_reduced": (
                float(full_classification["macro_f1"])
                - float(reduced_classification["macro_f1"])
            ),
            "unsafe_recall_delta_full_minus_reduced": (
                float(full_classification["unsafe_recall"])
                - float(reduced_classification["unsafe_recall"])
            ),
            "hard_prediction_change_rate": hard_change_rate,
            "mean_absolute_probability_change": float(
                np.mean(np.abs(full - reduced))
            ),
            "bootstrap": bootstrap,
        }
        delta["probability_quality_marginal_gate"] = {
            "positive_point_log_loss_delta": (
                delta["log_loss_delta_reduced_minus_full"] > 0
            ),
            "positive_in_at_least_70_percent_bootstraps": (
                bootstrap["positive_log_loss_fraction"] >= 0.70
            ),
        }
        delta["probability_quality_marginal_gate"]["passed"] = all(
            delta["probability_quality_marginal_gate"].values()
        )
        delta["classification_marginal_gate"] = {
            "positive_point_macro_f1_delta": (
                delta["macro_f1_delta_full_minus_reduced"] > 0
            ),
            "positive_macro_f1_in_at_least_70_percent_bootstraps": (
                bootstrap["positive_macro_f1_fraction"] >= 0.70
            ),
            "nonnegative_point_unsafe_recall_delta": (
                delta["unsafe_recall_delta_full_minus_reduced"] >= 0
            ),
            "changes_at_least_1_percent_hard_predictions": (
                hard_change_rate >= 0.01
            ),
        }
        delta["classification_marginal_gate"]["passed"] = all(
            delta["classification_marginal_gate"].values()
        )
        delta["preliminary_marginal_gate"] = {
            "basis": (
                "primary_macro_f1_with_nonnegative_unsafe_recall_and_"
                "prediction_change"
            ),
            "passed": delta["classification_marginal_gate"]["passed"],
        }
        marginal[omitted] = delta

    return (
        {
            "schema": "softfusion-fixed-pool-audit-v1",
            "status": "completed",
            "judge_pool": list(JUDGES),
            "probe": {
                "family": "main_effects_logistic",
                "feature": "logit_of_selected_calibrated_probability",
                "l2_strength": PROBE_L2_STRENGTH,
                "selection_role": (
                    "preliminary_gate_only_no_model_or_subset_selection"
                ),
            },
            "models": model_details,
            "marginal_contributions": marginal,
            "gate": {
                "rule": (
                    "all_three_frozen_judges_must_improve_primary_macro_f1_"
                    "in_at_least_70_percent_component_bootstraps_without_"
                    "reducing_point_unsafe_recall_and_must_change_at_least_"
                    "one_percent_of_hard_predictions"
                ),
                "passed": all(
                    value["preliminary_marginal_gate"]["passed"]
                    for value in marginal.values()
                ),
            },
        },
        model_predictions,
    )


def calibration_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M2.2 Cross-Fitted Calibration Report",
        "",
        "**Status:** completed without confirmatory-label access.",
        "",
        "## Selected Calibrators",
        "",
        "| Judge | Selected | Raw OOF log loss | Calibrated OOF log loss | Raw OOF Brier | Calibrated OOF Brier |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for judge in JUDGES:
        value = report["judges"][judge]
        raw = value["raw_fit_metrics"]
        selected = value["selected_fit_oof_metrics"]
        lines.append(
            "| {judge} | {method} | {raw_loss:.6f} | {selected_loss:.6f} | "
            "{raw_brier:.6f} | {selected_brier:.6f} |".format(
                judge=judge,
                method=value["selected_method"],
                raw_loss=raw["cluster_averaged_log_loss"],
                selected_loss=selected["cluster_averaged_log_loss"],
                raw_brier=raw["cluster_averaged_brier"],
                selected_brier=selected["cluster_averaged_brier"],
            )
        )
    lines.extend(
        [
            "",
            "Calibrator type selection used only five-fold image-component "
            "cross-fitted predictions from `D_fit`. `D_select` was transformed "
            "once with calibrators refitted on all of `D_fit`; its labels did "
            "not influence calibrator selection.",
            "",
        ]
    )
    return "\n".join(lines)


def pool_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# M2.2 Fixed Judge-Pool Complementarity Audit",
        "",
        "**Role:** preliminary CPU gate; this is not the final fusion model.",
        "",
        "| Removed judge | Delta log loss | Delta Macro-F1 | Delta unsafe recall | Hard prediction change | Macro-F1 positive bootstrap | Classification gate | Log-loss gate |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for judge in JUDGES:
        value = audit["marginal_contributions"][judge]
        lines.append(
            "| {judge} | {loss:.6f} | {f1:.6f} | {recall:.6f} | "
            "{change:.2%} | {positive:.2%} | {classification_gate} | "
            "{logloss_gate} |".format(
                judge=judge,
                loss=value["log_loss_delta_reduced_minus_full"],
                f1=value["macro_f1_delta_full_minus_reduced"],
                recall=value["unsafe_recall_delta_full_minus_reduced"],
                change=value["hard_prediction_change_rate"],
                positive=value["bootstrap"]["positive_macro_f1_fraction"],
                classification_gate=(
                    "PASS"
                    if value["classification_marginal_gate"]["passed"]
                    else "FAIL"
                ),
                logloss_gate=(
                    "PASS"
                    if value["probability_quality_marginal_gate"]["passed"]
                    else "FAIL"
                ),
            )
        )
    lines.extend(
        [
            "",
            "The fixed probe uses the same three-judge pool for every "
            "comparison. Leave-one-out variants are deletion ablations, not "
            "committee candidates. The preliminary continuation gate follows "
            "the plan's primary classification endpoint: each judge must "
            "improve Macro-F1 in at least 70% of 500 component bootstraps, "
            "must not reduce point unsafe recall, and must change at least 1% "
            "of hard predictions. The separate log-loss gate records whether "
            "the judge also improves probability quality; it is not silently "
            "substituted for the primary classification endpoint.",
            "",
            "## Decision",
            "",
            (
                "M2.2 PASSED its preliminary classification-marginal gate. "
                "The consistency-view pilot may proceed. Any failed log-loss "
                "marginal gate remains a required caveat for SF3."
                if audit["gate"]["passed"]
                else (
                    "M2.2 FAILED. Do not start consistency-view GPU runs; the "
                    "current evidence does not support a three-judge "
                    "complementary-fusion claim."
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(
        status_path,
        {
            "schema": "softfusion-m2-calibration-status-v1",
            "state": "running",
        },
    )
    try:
        fit = load_split(
            args.fit_manifest.resolve(), args.soft_root.resolve(), "fit"
        )
        select = load_split(
            args.select_manifest.resolve(), args.soft_root.resolve(), "select"
        )
        if fit["adapter_versions"] != select["adapter_versions"]:
            raise RuntimeError("Adapter versions differ between fit and select")
        if set(fit["components"]) & set(select["components"]):
            raise RuntimeError("Fit and select share image components")
        folds = make_group_folds(
            fit["targets"],
            fit["components"],
            n_splits=FOLD_COUNT,
            seed=SEED,
        )
        fold_by_component: dict[str, int] = {}
        for component, fold in zip(fit["components"], folds):
            previous = fold_by_component.setdefault(component, int(fold))
            if previous != int(fold):
                raise RuntimeError("A component was split across folds")

        calibration_report: dict[str, Any] = {
            "schema": "softfusion-calibration-report-v1",
            "status": "completed",
            "seed": SEED,
            "fold_count": FOLD_COUNT,
            "selection_data": "D_fit_only",
            "confirmatory_labels_accessed": False,
            "judges": {},
        }
        selected_methods = {}
        final_parameters = {}
        fit_calibrated = {}
        select_calibrated = {}
        for judge in JUDGES:
            raw_fit = fit["raw_probabilities"][judge]
            result = select_calibrator(
                raw_fit,
                fit["targets"],
                fit["components"],
                folds,
            )
            selected_method = str(result["selected_method"])
            selected_oof = np.asarray(
                result["selected_oof_probabilities"], dtype=np.float64
            )
            parameters = result["final_parameters"]
            transformed_select = apply_calibrator(
                parameters, select["raw_probabilities"][judge]
            )
            selected_methods[judge] = selected_method
            final_parameters[judge] = parameters
            fit_calibrated[judge] = selected_oof
            select_calibrated[judge] = transformed_select
            candidate_results = {}
            for method, candidate in result["candidate_results"].items():
                candidate_results[method] = {
                    key: jsonable(value)
                    for key, value in candidate.items()
                    if key != "oof_probabilities"
                }
            calibration_report["judges"][judge] = {
                "adapter_version": fit["adapter_versions"][judge],
                "selected_method": selected_method,
                "selection_rule": result["selection_rule"],
                "raw_fit_metrics": calibration_metrics(
                    fit["targets"], raw_fit, fit["components"]
                ),
                "selected_fit_oof_metrics": calibration_metrics(
                    fit["targets"], selected_oof, fit["components"]
                ),
                "frozen_select_metrics": calibration_metrics(
                    select["targets"],
                    transformed_select,
                    select["components"],
                ),
                "candidate_results": candidate_results,
                "final_fit_parameters": parameters,
            }

        calibrator_config = {
            "schema": "softfusion-calibrator-config-v1",
            "seed": SEED,
            "fold_count": FOLD_COUNT,
            "judge_order": list(JUDGES),
            "adapter_versions": fit["adapter_versions"],
            "selected_methods": selected_methods,
            "final_fit_parameters": final_parameters,
            "selection_source": "D_fit_five_fold_component_cross_fitting",
            "select_labels_used_for_selection": False,
            "confirmatory_labels_accessed": False,
        }
        calibrator_config["config_sha256"] = stable_json_sha256(
            calibrator_config
        )
        fit_rows = []
        for index, sample_id in enumerate(fit["sample_ids"]):
            fit_rows.append(
                {
                    "schema": "softfusion-calibrated-record-v1",
                    "split": "fit",
                    "sample_id": sample_id,
                    "component_id": fit["components"][index],
                    "expected_label": (
                        "unsafe" if fit["targets"][index] else "safe"
                    ),
                    "fold": int(folds[index]),
                    "raw_p_unsafe": {
                        judge: float(fit["raw_probabilities"][judge][index])
                        for judge in JUDGES
                    },
                    "calibrated_p_unsafe": {
                        judge: float(fit_calibrated[judge][index])
                        for judge in JUDGES
                    },
                }
            )
        select_rows = []
        for index, sample_id in enumerate(select["sample_ids"]):
            select_rows.append(
                {
                    "schema": "softfusion-calibrated-record-v1",
                    "split": "select",
                    "sample_id": sample_id,
                    "component_id": select["components"][index],
                    "expected_label": (
                        "unsafe" if select["targets"][index] else "safe"
                    ),
                    "fold": None,
                    "raw_p_unsafe": {
                        judge: float(
                            select["raw_probabilities"][judge][index]
                        )
                        for judge in JUDGES
                    },
                    "calibrated_p_unsafe": {
                        judge: float(select_calibrated[judge][index])
                        for judge in JUDGES
                    },
                }
            )

        pool_audit, probe_predictions = build_probe_audit(
            fit=fit,
            select=select,
            folds=folds,
            calibrator_methods=selected_methods,
            final_calibrators=final_parameters,
        )
        combined_targets = np.concatenate(
            [fit["targets"], select["targets"]]
        )
        combined_calibrated = {
            judge: np.concatenate(
                [fit_calibrated[judge], select_calibrated[judge]]
            )
            for judge in JUDGES
        }
        dependence_report = {
            "schema": "softfusion-dependence-report-v1",
            "judge_order": list(JUDGES),
            "fit_oof_pairwise": pairwise_dependence(
                fit["targets"], fit_calibrated
            ),
            "select_frozen_pairwise": pairwise_dependence(
                select["targets"], select_calibrated
            ),
            "combined_development_pairwise": pairwise_dependence(
                combined_targets, combined_calibrated
            ),
            "interpretation": (
                "descriptive_dependence_only_not_causal_or_independence_proof"
            ),
        }
        fold_report = {
            "schema": "softfusion-component-folds-v1",
            "seed": SEED,
            "fold_count": FOLD_COUNT,
            "component_to_fold": dict(sorted(fold_by_component.items())),
            "fold_record_counts": {
                str(fold): int(np.sum(folds == fold))
                for fold in range(FOLD_COUNT)
            },
            "fold_component_counts": {
                str(fold): len(
                    {
                        component
                        for component, value in fold_by_component.items()
                        if value == fold
                    }
                )
                for fold in range(FOLD_COUNT)
            },
            "fold_unsafe_record_counts": {
                str(fold): int(fit["targets"][folds == fold].sum())
                for fold in range(FOLD_COUNT)
            },
        }
        source_files = {
            str(path.resolve()): sha256_file(path.resolve())
            for path in fit["input_files"] + select["input_files"]
        }
        provenance = {
            "schema": "softfusion-m2-provenance-v1",
            "source_files_sha256": source_files,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "calibration_module_sha256": sha256_file(
                SRC_ROOT / "multijudge" / "calibration.py"
            ),
            "confirmatory_labels_accessed": False,
            "confirmatory_predictions_generated": False,
        }

        write_json(
            output_dir / "calibration_report.json",
            jsonable(calibration_report),
        )
        write_json(output_dir / "calibrators.json", calibrator_config)
        write_json(output_dir / "fold_assignments.json", fold_report)
        write_json(
            output_dir / "dependence_report.json", dependence_report
        )
        write_json(
            output_dir / "judge_pool_audit.json", jsonable(pool_audit)
        )
        write_json(output_dir / "provenance.json", provenance)
        write_jsonl(output_dir / "fit_oof_calibrated.jsonl", fit_rows)
        write_jsonl(
            output_dir / "select_frozen_calibrated.jsonl", select_rows
        )
        write_json(
            output_dir / "probe_predictions.json",
            {
                "schema": "softfusion-preliminary-probe-predictions-v1",
                "sample_ids": fit["sample_ids"] + select["sample_ids"],
                "splits": (
                    ["fit"] * len(fit["sample_ids"])
                    + ["select"] * len(select["sample_ids"])
                ),
                "predictions": {
                    name: values.tolist()
                    for name, values in probe_predictions.items()
                },
            },
        )
        (output_dir / "CALIBRATION_REPORT.md").write_text(
            calibration_markdown(calibration_report),
            encoding="utf-8",
        )
        (output_dir / "JUDGE_POOL_AUDIT.md").write_text(
            pool_markdown(pool_audit),
            encoding="utf-8",
        )
        write_json(
            status_path,
            {
                "schema": "softfusion-m2-calibration-status-v1",
                "state": "completed",
                "calibration_completed": True,
                "fixed_pool_gate_passed": pool_audit["gate"]["passed"],
                "confirmatory_labels_accessed": False,
            },
        )
        print(
            json.dumps(
                {
                    "state": "completed",
                    "selected_calibrators": selected_methods,
                    "fixed_pool_gate_passed": pool_audit["gate"]["passed"],
                }
            )
        )
    except Exception as exc:
        write_json(
            status_path,
            {
                "schema": "softfusion-m2-calibration-status-v1",
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    softfusion_root = (
        V15_ROOT / "results" / "mmds"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Fit D_fit-only cross-fitted calibrators and run the frozen "
            "three-judge preliminary complementarity gate."
        )
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=softfusion_root / "data" / "fit.jsonl",
    )
    parser.add_argument(
        "--select-manifest",
        type=Path,
        default=softfusion_root / "data" / "select.jsonl",
    )
    parser.add_argument(
        "--soft-root",
        type=Path,
        default=softfusion_root / "qualification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=softfusion_root / "calibration",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
