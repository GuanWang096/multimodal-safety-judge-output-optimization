from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cluster_inference import (  # noqa: E402
    paired_component_bootstrap,
    weighted_classification_metrics,
)
from multijudge.schema import canonical_json, sha256_file, sha256_text  # noqa: E402


JUDGES = ("qwen", "safework", "minicpm")
EPSILON = 1e-6
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
SOFT_FEATURE_FAMILIES = (
    "soft_logits",
    "raw_logits",
    "dual_logits",
    "beta_features",
    "fusion_plus_vote",
    "soft_plus_vote",
)
HARD_BASELINE_FAMILIES = ("binary_judges", "vote_count")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_fold(group: str, fold_count: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


def feature_matrices(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    calibrated = np.column_stack(
        [
            [float(row["calibrated_p_unsafe"][judge]) for row in rows]
            for judge in JUDGES
        ]
    )
    calibrated_logits = logit(np.clip(calibrated, EPSILON, 1.0 - EPSILON))
    clipped_calibrated = np.clip(calibrated, EPSILON, 1.0 - EPSILON)
    beta_features = np.column_stack(
        [np.log(clipped_calibrated), -np.log1p(-clipped_calibrated)]
    )
    raw = np.column_stack(
        [
            [float(row["raw_p_unsafe"][judge]) for row in rows]
            for judge in JUDGES
        ]
    )
    votes = (raw >= 0.5).sum(axis=1)
    binary_judges = (raw >= 0.5).astype(float)
    raw_logits = logit(np.clip(raw, EPSILON, 1.0 - EPSILON))
    vote_fraction = votes[:, None] / len(JUDGES)
    vote_one_hot = np.column_stack([votes == count for count in range(4)]).astype(float)
    fusion = np.asarray([float(row["fusion_p_unsafe"]) for row in rows])
    fusion_logit = logit(np.clip(fusion, EPSILON, 1.0 - EPSILON))[:, None]
    return {
        "soft_logits": calibrated_logits,
        "raw_logits": raw_logits,
        "dual_logits": np.column_stack([calibrated_logits, raw_logits]),
        "beta_features": beta_features,
        "fusion_plus_vote": np.column_stack(
            [fusion_logit, vote_fraction, vote_one_hot]
        ),
        "soft_plus_vote": np.column_stack(
            [calibrated_logits, vote_fraction, vote_one_hot]
        ),
        "binary_judges": binary_judges,
        "vote_count": np.column_stack([vote_fraction, vote_one_hot]),
    }


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    result = weighted_classification_metrics(targets, predictions)
    return {
        key: result[key]
        for key in (
            "accuracy",
            "macro_f1",
            "unsafe_recall",
            "safe_specificity",
            "mcc",
            "unsafe_true_positive",
            "unsafe_false_positive",
            "unsafe_false_negative",
        )
    }


def metric_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["unsafe_recall"]),
        float(metrics["safe_specificity"]),
    )


def select_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(
        np.concatenate(
            (
                [0.0],
                np.quantile(scores, np.linspace(0.0, 1.0, 401)),
                [1.0],
            )
        )
    )
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        current = classification_metrics(targets, scores >= threshold)
        current_key = metric_key(current)
        if current_key > best_key:
            best_key = current_key
            best_threshold = float(threshold)
    return best_threshold


def fit_model(features: np.ndarray, targets: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-9,
        random_state=20260802,
    )
    model.fit(features, targets)
    return model


def inner_select(
    feature_sets: dict[str, np.ndarray],
    targets: np.ndarray,
    groups: Sequence[str],
    training_indices: np.ndarray,
    outer_fold: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    training_groups = np.asarray([groups[index] for index in training_indices])
    inner_folds = np.asarray(
        [stable_fold(group, 4, 20260820 + outer_fold) for group in training_groups]
    )
    for family in SOFT_FEATURE_FAMILIES:
        all_features = feature_sets[family]
        features = all_features[training_indices]
        for c_value in C_GRID:
            oof = np.full(len(training_indices), np.nan, dtype=float)
            for fold in range(4):
                validation = inner_folds == fold
                training = ~validation
                if not validation.any() or len(np.unique(targets[training_indices][training])) < 2:
                    continue
                model = fit_model(
                    features[training], targets[training_indices][training], c_value
                )
                oof[validation] = model.predict_proba(features[validation])[:, 1]
            if not np.all(np.isfinite(oof)):
                continue
            threshold = select_threshold(targets[training_indices], oof)
            current_metrics = classification_metrics(
                targets[training_indices], oof >= threshold
            )
            candidate = {
                "feature_family": family,
                "C": c_value,
                "threshold": threshold,
                "inner_oof_metrics": current_metrics,
            }
            if best is None or metric_key(current_metrics) > metric_key(
                best["inner_oof_metrics"]
            ):
                best = candidate
    if best is None:
        raise RuntimeError("No target-adaptation candidate completed inner selection")
    return best


def inner_select_c_for_family(
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    training_indices: np.ndarray,
    outer_fold: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    training_groups = np.asarray([groups[index] for index in training_indices])
    inner_folds = np.asarray(
        [stable_fold(group, 4, 20260820 + outer_fold) for group in training_groups]
    )
    selected_features = features[training_indices]
    selected_targets = targets[training_indices]
    for c_value in C_GRID:
        oof = np.full(len(training_indices), np.nan, dtype=float)
        for fold in range(4):
            validation = inner_folds == fold
            training = ~validation
            if not validation.any() or len(np.unique(selected_targets[training])) < 2:
                continue
            model = fit_model(
                selected_features[training], selected_targets[training], c_value
            )
            oof[validation] = model.predict_proba(selected_features[validation])[:, 1]
        if not np.all(np.isfinite(oof)):
            continue
        threshold = select_threshold(selected_targets, oof)
        current_metrics = classification_metrics(
            selected_targets, oof >= threshold
        )
        candidate = {
            "C": c_value,
            "threshold": threshold,
            "inner_oof_metrics": current_metrics,
        }
        if best is None or metric_key(current_metrics) > metric_key(
            best["inner_oof_metrics"]
        ):
            best = candidate
    if best is None:
        raise RuntimeError("Hard-baseline inner selection did not complete")
    return best


def inner_threshold_for_fixed_model(
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str],
    training_indices: np.ndarray,
    outer_fold: int,
    c_value: float,
) -> float:
    training_groups = np.asarray([groups[index] for index in training_indices])
    inner_folds = np.asarray(
        [stable_fold(group, 4, 20260820 + outer_fold) for group in training_groups]
    )
    selected_features = features[training_indices]
    selected_targets = targets[training_indices]
    oof = np.full(len(training_indices), np.nan, dtype=float)
    for fold in range(4):
        validation = inner_folds == fold
        training = ~validation
        if not validation.any() or len(np.unique(selected_targets[training])) < 2:
            continue
        model = fit_model(
            selected_features[training], selected_targets[training], c_value
        )
        oof[validation] = model.predict_proba(selected_features[validation])[:, 1]
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("Fixed-model inner OOF predictions are incomplete")
    return select_threshold(selected_targets, oof)


def run() -> dict[str, Any]:
    predictions_path = (
        V15_ROOT
        / "results/external/beavertails_v/predictions.jsonl"
    )
    rows = load_jsonl(predictions_path)
    targets = np.asarray(
        [1 if row["expected_label"] == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )
    groups = [str(row["group_id"]) for row in rows]
    feature_sets = feature_matrices(rows)
    fusion = np.asarray([float(row["fusion_p_unsafe"]) for row in rows])
    soft_mean = np.asarray(
        [
            np.mean(
                [float(row["calibrated_p_unsafe"][judge]) for judge in JUDGES]
            )
            for row in rows
        ]
    )
    majority = np.asarray(
        [1 if row["predictions"]["majority"] == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )
    frozen = np.asarray(
        [1 if row["predictions"]["frozen_fusion"] == "unsafe" else 0 for row in rows],
        dtype=np.int64,
    )
    outer_folds = np.asarray(
        [stable_fold(group, 5, 20260802) for group in groups], dtype=np.int64
    )
    threshold_predictions = np.full(len(rows), -1, dtype=np.int64)
    soft_mean_threshold_predictions = np.full(len(rows), -1, dtype=np.int64)
    model_predictions = np.full(len(rows), -1, dtype=np.int64)
    fixed_model_predictions = {
        family: np.full(len(rows), -1, dtype=np.int64)
        for family in SOFT_FEATURE_FAMILIES
    }
    adapted_hard_predictions = {
        family: np.full(len(rows), -1, dtype=np.int64)
        for family in HARD_BASELINE_FAMILIES
    }
    fold_records = []
    for fold in range(5):
        validation = np.flatnonzero(outer_folds == fold)
        training = np.flatnonzero(outer_folds != fold)
        threshold = select_threshold(targets[training], fusion[training])
        threshold_predictions[validation] = (
            fusion[validation] >= threshold
        ).astype(np.int64)
        soft_mean_threshold = select_threshold(targets[training], soft_mean[training])
        soft_mean_threshold_predictions[validation] = (
            soft_mean[validation] >= soft_mean_threshold
        ).astype(np.int64)

        selected = inner_select(
            feature_sets, targets, groups, training, outer_fold=fold
        )
        family = str(selected["feature_family"])
        model = fit_model(
            feature_sets[family][training],
            targets[training],
            float(selected["C"]),
        )
        scores = model.predict_proba(feature_sets[family][validation])[:, 1]
        model_predictions[validation] = (
            scores >= float(selected["threshold"])
        ).astype(np.int64)
        selected["outer_refit_intercept"] = float(model.intercept_[0])
        selected["outer_refit_coefficients"] = model.coef_[0].tolist()
        fixed_records = {}
        for fixed_family in SOFT_FEATURE_FAMILIES:
            fixed_features = feature_sets[fixed_family]
            fixed_threshold = inner_threshold_for_fixed_model(
                fixed_features,
                targets,
                groups,
                training,
                outer_fold=fold,
                c_value=0.03,
            )
            fixed_model = fit_model(
                fixed_features[training], targets[training], 0.03
            )
            fixed_scores = fixed_model.predict_proba(
                fixed_features[validation]
            )[:, 1]
            fixed_model_predictions[fixed_family][validation] = (
                fixed_scores >= fixed_threshold
            ).astype(np.int64)
            fixed_records[fixed_family] = {
                "C": 0.03,
                "inner_oof_threshold": fixed_threshold,
                "outer_refit_intercept": float(fixed_model.intercept_[0]),
                "outer_refit_coefficients": fixed_model.coef_[0].tolist(),
            }
        hard_records = {}
        for hard_family in HARD_BASELINE_FAMILIES:
            hard_features = feature_sets[hard_family]
            hard_selected = inner_select_c_for_family(
                hard_features,
                targets,
                groups,
                training,
                outer_fold=fold,
            )
            hard_model = fit_model(
                hard_features[training],
                targets[training],
                float(hard_selected["C"]),
            )
            hard_scores = hard_model.predict_proba(
                hard_features[validation]
            )[:, 1]
            adapted_hard_predictions[hard_family][validation] = (
                hard_scores >= float(hard_selected["threshold"])
            ).astype(np.int64)
            hard_selected["outer_refit_intercept"] = float(
                hard_model.intercept_[0]
            )
            hard_selected["outer_refit_coefficients"] = (
                hard_model.coef_[0].tolist()
            )
            hard_records[hard_family] = hard_selected
        fold_records.append(
            {
                "fold": fold,
                "training_count": len(training),
                "validation_count": len(validation),
                "threshold_only": threshold,
                "soft_mean_threshold_only": soft_mean_threshold,
                "selected_model": selected,
                "fixed_models": fixed_records,
                "adapted_hard_baselines": hard_records,
            }
        )
    if (
        (threshold_predictions < 0).any()
        or (model_predictions < 0).any()
        or (soft_mean_threshold_predictions < 0).any()
        or any((value < 0).any() for value in fixed_model_predictions.values())
        or any((value < 0).any() for value in adapted_hard_predictions.values())
    ):
        raise RuntimeError("Cross-fitted target-adaptation predictions are incomplete")

    methods = {
        "majority": majority,
        "frozen_fusion": frozen,
        "cross_fitted_threshold_only": threshold_predictions,
        "cross_fitted_soft_mean_threshold": soft_mean_threshold_predictions,
        "nested_cross_fitted_target_fusion": model_predictions,
        **{
            f"fixed_cross_fitted_{family}": predictions
            for family, predictions in fixed_model_predictions.items()
        },
        **{
            f"nested_cross_fitted_adapted_{family}": predictions
            for family, predictions in adapted_hard_predictions.items()
        },
    }
    method_metrics = {
        name: classification_metrics(targets, predictions)
        for name, predictions in methods.items()
    }
    comparisons = {}
    for name in (
        "cross_fitted_threshold_only",
        "nested_cross_fitted_target_fusion",
        "cross_fitted_soft_mean_threshold",
        *(
            f"fixed_cross_fitted_{family}"
            for family in fixed_model_predictions
        ),
        *(
            f"nested_cross_fitted_adapted_{family}"
            for family in adapted_hard_predictions
        ),
    ):
        comparisons[name] = paired_component_bootstrap(
            targets=targets,
            groups=groups,
            left_predictions=methods[name],
            right_predictions=majority,
            replicates=10000,
            seed=20260900 + len(comparisons),
        )
    pairwise_adaptation_comparisons = {
        "soft_logits_minus_adapted_binary_judges": paired_component_bootstrap(
            targets=targets,
            groups=groups,
            left_predictions=methods["fixed_cross_fitted_soft_logits"],
            right_predictions=methods[
                "nested_cross_fitted_adapted_binary_judges"
            ],
            replicates=10000,
            seed=20260920,
        ),
        "soft_logits_minus_target_tuned_soft_mean": paired_component_bootstrap(
            targets=targets,
            groups=groups,
            left_predictions=methods["fixed_cross_fitted_soft_logits"],
            right_predictions=methods["cross_fitted_soft_mean_threshold"],
            replicates=10000,
            seed=20260921,
        ),
    }

    prediction_path = (
        V15_ROOT
        / "results/external/beavertails_v/target_adaptation_predictions.jsonl"
    )
    prediction_lines = []
    for index, row in enumerate(rows):
        prediction_lines.append(
            canonical_json(
                {
                    "schema": "beavertails-v-target-adaptation-prediction-v1",
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "expected_label": row["expected_label"],
                    "outer_fold": int(outer_folds[index]),
                    "predictions": {
                        name: "unsafe" if values[index] else "safe"
                        for name, values in methods.items()
                    },
                }
            )
        )
    prediction_path.write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")

    output = {
        "schema": "beavertails-v-target-adaptation-diagnostic-v1",
        "role": "posthoc_supervised_target_adaptation_not_frozen_transfer",
        "sample_count": len(rows),
        "group_count": len(set(groups)),
        "prediction_file": prediction_path.name,
        "prediction_file_sha256": sha256_file(prediction_path),
        "protocol": {
            "outer_folds": 5,
            "outer_split_unit": "shared image-question pair",
            "inner_folds": 4,
            "inner_selection": "feature family, L2 C, and threshold by inner OOF Macro-F1",
            "outer_assessment": "predictions only on held-out groups",
            "target_labels_required": True,
        },
        "metrics": method_metrics,
        "paired_bootstrap_vs_majority": comparisons,
        "paired_bootstrap_adaptation_comparisons": pairwise_adaptation_comparisons,
        "fold_records": fold_records,
        "interpretation_constraints": [
            "This analysis estimates supervised target-domain adaptation, not zero-shot transfer.",
            "BeaverTails-V labels were observed before the protocol was implemented.",
            "The result is exploratory and requires confirmation on a new untouched dataset or a prospectively frozen split.",
        ],
    }
    output["content_sha256"] = sha256_text(canonical_json(output))
    output_path = (
        V15_ROOT
        / "results/external/beavertails_v/target_adaptation_diagnostic.json"
    )
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
