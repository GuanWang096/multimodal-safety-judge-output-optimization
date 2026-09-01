from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr


EPSILON = 1e-6
CALIBRATOR_COMPLEXITY = {
    "identity": 0,
    "temperature": 1,
    "beta": 3,
}


def clip_probabilities(values: Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite")
    return np.clip(probabilities, EPSILON, 1.0 - EPSILON)


def binary_targets(values: Sequence[int | bool | str]) -> np.ndarray:
    converted = []
    for value in values:
        if value in (1, True, "unsafe"):
            converted.append(1.0)
        elif value in (0, False, "safe"):
            converted.append(0.0)
        else:
            raise ValueError(f"Unsupported binary target: {value!r}")
    targets = np.asarray(converted, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("Targets must be a nonempty one-dimensional sequence")
    return targets


def cluster_weights(groups: Sequence[str]) -> np.ndarray:
    if not groups:
        raise ValueError("At least one group is required")
    counts = Counter(str(value) for value in groups)
    weights = np.asarray(
        [1.0 / counts[str(value)] for value in groups],
        dtype=np.float64,
    )
    return weights / weights.sum()


def cluster_averaged_log_loss(
    targets: Sequence[int | bool | str],
    probabilities: Sequence[float],
    groups: Sequence[str],
) -> float:
    y = binary_targets(targets)
    p = clip_probabilities(probabilities)
    if len(y) != len(p) or len(y) != len(groups):
        raise ValueError("Targets, probabilities, and groups must align")
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.dot(cluster_weights(groups), losses))


def cluster_averaged_brier(
    targets: Sequence[int | bool | str],
    probabilities: Sequence[float],
    groups: Sequence[str],
) -> float:
    y = binary_targets(targets)
    p = clip_probabilities(probabilities)
    if len(y) != len(p) or len(y) != len(groups):
        raise ValueError("Targets, probabilities, and groups must align")
    return float(np.dot(cluster_weights(groups), np.square(p - y)))


def adaptive_ece(
    targets: Sequence[int | bool | str],
    probabilities: Sequence[float],
    groups: Sequence[str],
    *,
    bin_count: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    y = binary_targets(targets)
    p = clip_probabilities(probabilities)
    if len(y) != len(p) or len(y) != len(groups):
        raise ValueError("Targets, probabilities, and groups must align")
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    weights = cluster_weights(groups)
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, min(bin_count, len(order)))
    ece = 0.0
    details: list[dict[str, float | int]] = []
    for index, members in enumerate(bins):
        if not len(members):
            continue
        bin_weights = weights[members]
        weight = float(bin_weights.sum())
        normalized = bin_weights / weight
        mean_probability = float(np.dot(normalized, p[members]))
        observed_rate = float(np.dot(normalized, y[members]))
        gap = abs(mean_probability - observed_rate)
        ece += weight * gap
        details.append(
            {
                "bin": index,
                "record_count": int(len(members)),
                "weight": weight,
                "minimum_probability": float(p[members].min()),
                "maximum_probability": float(p[members].max()),
                "mean_probability": mean_probability,
                "observed_unsafe_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    return float(ece), details


def calibration_metrics(
    targets: Sequence[int | bool | str],
    probabilities: Sequence[float],
    groups: Sequence[str],
) -> dict[str, Any]:
    ece, bins = adaptive_ece(targets, probabilities, groups)
    return {
        "cluster_averaged_log_loss": cluster_averaged_log_loss(
            targets, probabilities, groups
        ),
        "cluster_averaged_brier": cluster_averaged_brier(
            targets, probabilities, groups
        ),
        "adaptive_ece": ece,
        "reliability_bins": bins,
    }


def _weighted_log_loss(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    p = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    losses = -(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p))
    return float(np.dot(weights, losses) / weights.sum())


def apply_calibrator(
    parameters: dict[str, Any],
    probabilities: Sequence[float],
) -> np.ndarray:
    p = clip_probabilities(probabilities)
    method = str(parameters["method"])
    if method == "identity":
        return p
    if method == "temperature":
        temperature = float(parameters["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        return expit(logit(p) / temperature)
    if method == "beta":
        a = float(parameters["a"])
        b = float(parameters["b"])
        c = float(parameters["c"])
        if min(a, b) <= 0 or not all(math.isfinite(v) for v in (a, b, c)):
            raise ValueError("Invalid beta-calibration parameters")
        return expit(a * np.log(p) - b * np.log1p(-p) + c)
    raise ValueError(f"Unknown calibration method: {method}")


def fit_calibrator(
    method: str,
    probabilities: Sequence[float],
    targets: Sequence[int | bool | str],
    groups: Sequence[str],
) -> dict[str, Any]:
    p = clip_probabilities(probabilities)
    y = binary_targets(targets)
    if len(p) != len(y) or len(p) != len(groups):
        raise ValueError("Calibration inputs must align")
    weights = cluster_weights(groups)

    if method == "identity":
        return {"method": "identity"}

    if method == "temperature":
        def objective(theta: np.ndarray) -> float:
            calibrated = expit(logit(p) / math.exp(float(theta[0])))
            return _weighted_log_loss(y, calibrated, weights)

        result = minimize(
            objective,
            x0=np.asarray([0.0]),
            method="L-BFGS-B",
            bounds=[(-5.0, 5.0)],
        )
        if not result.success:
            raise RuntimeError(f"Temperature scaling failed: {result.message}")
        return {
            "method": "temperature",
            "temperature": math.exp(float(result.x[0])),
            "optimizer": "L-BFGS-B",
            "objective": float(result.fun),
        }

    if method == "beta":
        def objective(theta: np.ndarray) -> float:
            a = math.exp(float(theta[0]))
            b = math.exp(float(theta[1]))
            calibrated = expit(
                a * np.log(p) - b * np.log1p(-p) + float(theta[2])
            )
            penalty = 1e-8 * float(np.dot(theta[:2], theta[:2]))
            return _weighted_log_loss(y, calibrated, weights) + penalty

        result = minimize(
            objective,
            x0=np.asarray([0.0, 0.0, 0.0]),
            method="L-BFGS-B",
            bounds=[(-5.0, 5.0), (-5.0, 5.0), (-10.0, 10.0)],
        )
        if not result.success:
            raise RuntimeError(f"Beta calibration failed: {result.message}")
        return {
            "method": "beta",
            "a": math.exp(float(result.x[0])),
            "b": math.exp(float(result.x[1])),
            "c": float(result.x[2]),
            "optimizer": "L-BFGS-B",
            "objective": float(result.fun),
            "monotonic": True,
        }

    raise ValueError(f"Unknown calibration method: {method}")


def make_group_folds(
    targets: Sequence[int | bool | str],
    groups: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 20260729,
) -> np.ndarray:
    y = binary_targets(targets)
    group_values = [str(value) for value in groups]
    if len(y) != len(group_values):
        raise ValueError("Targets and groups must align")
    unique_groups = sorted(set(group_values))
    if len(unique_groups) < n_splits:
        raise ValueError("Fewer groups than requested folds")

    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_values):
        group_indices[group].append(index)
    rng = np.random.default_rng(seed)
    tie_breakers = {
        group: float(rng.random()) for group in unique_groups
    }
    summaries = []
    for group, indices in group_indices.items():
        positive = float(y[indices].sum())
        negative = float(len(indices) - positive)
        summaries.append(
            (group, positive, negative, len(indices), tie_breakers[group])
        )
    summaries.sort(
        key=lambda value: (
            -(value[1] > 0),
            -value[1],
            -value[3],
            value[4],
            value[0],
        )
    )

    fold_positive = np.zeros(n_splits, dtype=np.float64)
    fold_negative = np.zeros(n_splits, dtype=np.float64)
    fold_records = np.zeros(n_splits, dtype=np.float64)
    fold_groups = np.zeros(n_splits, dtype=np.float64)
    target_positive = max(float(y.sum()) / n_splits, 1.0)
    target_negative = max(float((1.0 - y).sum()) / n_splits, 1.0)
    target_records = max(float(len(y)) / n_splits, 1.0)
    target_groups = max(float(len(unique_groups)) / n_splits, 1.0)
    assignment: dict[str, int] = {}

    for group, positive, negative, size, _ in summaries:
        candidates = []
        for fold in range(n_splits):
            candidate_positive = fold_positive.copy()
            candidate_negative = fold_negative.copy()
            candidate_records = fold_records.copy()
            candidate_groups = fold_groups.copy()
            candidate_positive[fold] += positive
            candidate_negative[fold] += negative
            candidate_records[fold] += size
            candidate_groups[fold] += 1
            score = (
                np.var(candidate_positive / target_positive)
                + np.var(candidate_negative / target_negative)
                + 0.25 * np.var(candidate_records / target_records)
                + 0.05 * np.var(candidate_groups / target_groups)
            )
            candidates.append(
                (
                    float(score),
                    fold_records[fold],
                    fold_groups[fold],
                    fold,
                )
            )
        selected = min(candidates)[-1]
        assignment[group] = selected
        fold_positive[selected] += positive
        fold_negative[selected] += negative
        fold_records[selected] += size
        fold_groups[selected] += 1

    folds = np.asarray(
        [assignment[group] for group in group_values],
        dtype=np.int64,
    )
    for fold in range(n_splits):
        members = folds == fold
        if not np.any(members):
            raise RuntimeError(f"Fold {fold} is empty")
        if len(set(np.asarray(group_values)[members])) == 0:
            raise RuntimeError(f"Fold {fold} has no groups")
    return folds


def cross_fit_calibrator(
    method: str,
    probabilities: Sequence[float],
    targets: Sequence[int | bool | str],
    groups: Sequence[str],
    folds: Sequence[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    p = clip_probabilities(probabilities)
    y = binary_targets(targets)
    group_values = np.asarray([str(value) for value in groups])
    fold_values = np.asarray(folds, dtype=np.int64)
    if not (len(p) == len(y) == len(group_values) == len(fold_values)):
        raise ValueError("Cross-fitting inputs must align")
    output = np.full(len(p), np.nan, dtype=np.float64)
    fold_parameters: list[dict[str, Any]] = []
    for fold in sorted(set(int(value) for value in fold_values)):
        validation = fold_values == fold
        training = ~validation
        train_groups = set(group_values[training])
        validation_groups = set(group_values[validation])
        if train_groups & validation_groups:
            raise RuntimeError("A component was split across calibration folds")
        parameters = fit_calibrator(
            method,
            p[training],
            y[training],
            group_values[training].tolist(),
        )
        output[validation] = apply_calibrator(parameters, p[validation])
        fold_parameters.append(
            {
                "fold": fold,
                "training_record_count": int(training.sum()),
                "validation_record_count": int(validation.sum()),
                "training_component_count": len(train_groups),
                "validation_component_count": len(validation_groups),
                "parameters": parameters,
            }
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Cross-fitting left non-finite predictions")
    return output, fold_parameters


def select_calibrator(
    probabilities: Sequence[float],
    targets: Sequence[int | bool | str],
    groups: Sequence[str],
    folds: Sequence[int],
    *,
    tie_tolerance: float = 0.005,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for method in ("identity", "temperature", "beta"):
        oof, fold_parameters = cross_fit_calibrator(
            method, probabilities, targets, groups, folds
        )
        candidates[method] = {
            "method": method,
            "parameter_count": CALIBRATOR_COMPLEXITY[method],
            "metrics": calibration_metrics(targets, oof, groups),
            "fold_parameters": fold_parameters,
            "oof_probabilities": oof,
        }
    best_loss = min(
        value["metrics"]["cluster_averaged_log_loss"]
        for value in candidates.values()
    )
    near_best = [
        value
        for value in candidates.values()
        if value["metrics"]["cluster_averaged_log_loss"] - best_loss
        < tie_tolerance
    ]
    selected = min(
        near_best,
        key=lambda value: (
            value["parameter_count"],
            value["metrics"]["cluster_averaged_log_loss"],
            value["method"],
        ),
    )
    identity = candidates["identity"]
    if (
        selected["method"] != "identity"
        and selected["metrics"]["cluster_averaged_log_loss"]
        >= identity["metrics"]["cluster_averaged_log_loss"]
        and selected["metrics"]["cluster_averaged_brier"]
        >= identity["metrics"]["cluster_averaged_brier"]
    ):
        selected = identity
    final_parameters = fit_calibrator(
        selected["method"], probabilities, targets, groups
    )
    return {
        "selected_method": selected["method"],
        "selection_rule": {
            "primary": "minimum_cluster_averaged_log_loss",
            "tie_tolerance": tie_tolerance,
            "tie_break": "fewer_parameters",
            "identity_fallback": (
                "retain_identity_if_neither_log_loss_nor_brier_improves"
            ),
        },
        "candidate_results": candidates,
        "selected_oof_probabilities": selected["oof_probabilities"],
        "final_parameters": final_parameters,
    }


@dataclass(frozen=True)
class LogisticProbe:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    l2_strength: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        standardized = (values - self.mean) / self.scale
        return expit(standardized @ self.coefficients + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "l2_strength": self.l2_strength,
        }


def fit_logistic_probe(
    features: np.ndarray,
    targets: Sequence[int | bool | str],
    groups: Sequence[str],
    feature_names: Sequence[str],
    *,
    l2_strength: float = 0.1,
) -> LogisticProbe:
    x = np.asarray(features, dtype=np.float64)
    y = binary_targets(targets)
    if x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError("Feature matrix and targets must align")
    if x.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match feature columns")
    weights = cluster_weights(groups)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average(np.square(x - mean), axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (x - mean) / scale

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = theta[:-1]
        intercept = float(theta[-1])
        scores = standardized @ coefficients + intercept
        probabilities = expit(scores)
        loss = _weighted_log_loss(y, probabilities, weights)
        penalty = 0.5 * l2_strength * float(
            np.dot(coefficients, coefficients)
        )
        residual = probabilities - y
        gradient_coefficients = (
            standardized.T @ (weights * residual) / weights.sum()
            + l2_strength * coefficients
        )
        gradient_intercept = float(
            np.dot(weights, residual) / weights.sum()
        )
        gradient = np.concatenate(
            [gradient_coefficients, [gradient_intercept]]
        )
        return loss + penalty, gradient

    initial_rate = float(np.dot(weights, y) / weights.sum())
    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[-1] = float(logit(np.clip(initial_rate, EPSILON, 1 - EPSILON)))
    result = minimize(
        lambda theta: objective(theta),
        initial,
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success:
        raise RuntimeError(f"Logistic probe failed: {result.message}")
    return LogisticProbe(
        feature_names=tuple(str(value) for value in feature_names),
        mean=mean,
        scale=scale,
        coefficients=np.asarray(result.x[:-1], dtype=np.float64),
        intercept=float(result.x[-1]),
        l2_strength=l2_strength,
    )


def nested_probe_predictions(
    raw_probabilities: dict[str, Sequence[float]],
    targets: Sequence[int | bool | str],
    groups: Sequence[str],
    folds: Sequence[int],
    calibrator_methods: dict[str, str],
    included_judges: Sequence[str],
    *,
    l2_strength: float = 0.1,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    judges = tuple(str(value) for value in included_judges)
    y = binary_targets(targets)
    group_values = np.asarray([str(value) for value in groups])
    fold_values = np.asarray(folds, dtype=np.int64)
    raw = {
        judge: clip_probabilities(raw_probabilities[judge])
        for judge in judges
    }
    if any(len(value) != len(y) for value in raw.values()):
        raise ValueError("Raw judge probabilities do not align")
    output = np.full(len(y), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for fold in sorted(set(int(value) for value in fold_values)):
        validation = fold_values == fold
        training = ~validation
        train_features = []
        validation_features = []
        calibration_parameters = {}
        for judge in judges:
            parameters = fit_calibrator(
                calibrator_methods[judge],
                raw[judge][training],
                y[training],
                group_values[training].tolist(),
            )
            calibration_parameters[judge] = parameters
            train_features.append(
                logit(apply_calibrator(parameters, raw[judge][training]))
            )
            validation_features.append(
                logit(apply_calibrator(parameters, raw[judge][validation]))
            )
        train_matrix = np.column_stack(train_features)
        validation_matrix = np.column_stack(validation_features)
        probe = fit_logistic_probe(
            train_matrix,
            y[training],
            group_values[training].tolist(),
            judges,
            l2_strength=l2_strength,
        )
        output[validation] = probe.predict(validation_matrix)
        records.append(
            {
                "fold": fold,
                "calibrators": calibration_parameters,
                "probe": probe.to_dict(),
            }
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Nested probe produced non-finite predictions")
    return output, records


def fit_final_probe(
    raw_probabilities: dict[str, Sequence[float]],
    targets: Sequence[int | bool | str],
    groups: Sequence[str],
    calibrator_parameters: dict[str, dict[str, Any]],
    included_judges: Sequence[str],
    *,
    l2_strength: float = 0.1,
) -> LogisticProbe:
    judges = tuple(str(value) for value in included_judges)
    features = np.column_stack(
        [
            logit(
                apply_calibrator(
                    calibrator_parameters[judge],
                    raw_probabilities[judge],
                )
            )
            for judge in judges
        ]
    )
    return fit_logistic_probe(
        features,
        targets,
        groups,
        judges,
        l2_strength=l2_strength,
    )


def predict_final_probe(
    probe: LogisticProbe,
    raw_probabilities: dict[str, Sequence[float]],
    calibrator_parameters: dict[str, dict[str, Any]],
) -> np.ndarray:
    features = np.column_stack(
        [
            logit(
                apply_calibrator(
                    calibrator_parameters[judge],
                    raw_probabilities[judge],
                )
            )
            for judge in probe.feature_names
        ]
    )
    return probe.predict(features)


def pairwise_dependence(
    targets: Sequence[int | bool | str],
    probabilities: dict[str, Sequence[float]],
) -> list[dict[str, Any]]:
    y = binary_targets(targets)
    judges = sorted(probabilities)
    output = []
    for left_index, left in enumerate(judges):
        left_p = clip_probabilities(probabilities[left])
        left_error = (left_p >= 0.5).astype(int) != y.astype(int)
        for right in judges[left_index + 1 :]:
            right_p = clip_probabilities(probabilities[right])
            right_error = (right_p >= 0.5).astype(int) != y.astype(int)
            pearson_score = float(np.corrcoef(left_p, right_p)[0, 1])
            spearman_score = float(spearmanr(left_p, right_p).statistic)
            if np.std(left_error) == 0 or np.std(right_error) == 0:
                error_phi = None
            else:
                error_phi = float(
                    np.corrcoef(
                        left_error.astype(float),
                        right_error.astype(float),
                    )[0, 1]
                )
            output.append(
                {
                    "left": left,
                    "right": right,
                    "soft_pearson": pearson_score,
                    "soft_spearman": spearman_score,
                    "error_phi": error_phi,
                    "both_wrong_count": int(
                        np.logical_and(left_error, right_error).sum()
                    ),
                    "hard_disagreement_count": int(
                        ((left_p >= 0.5) != (right_p >= 0.5)).sum()
                    ),
                }
            )
    return output


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
