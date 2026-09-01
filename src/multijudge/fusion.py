from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .calibration import apply_calibrator, cluster_weights, fit_calibrator


EPSILON = 1e-6
FEATURE_FAMILIES = ("F1", "F2")


def calibrated_feature_matrix(
    probabilities: Mapping[str, Sequence[float]],
    judge_order: Sequence[str],
    feature_family: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if feature_family not in FEATURE_FAMILIES:
        raise ValueError(f"Unsupported feature family: {feature_family}")
    judges = tuple(str(value) for value in judge_order)
    logits = []
    for judge in judges:
        values = np.asarray(probabilities[judge], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid calibrated probabilities for {judge}")
        logits.append(logit(np.clip(values, EPSILON, 1.0 - EPSILON)))
    if not logits:
        raise ValueError("At least one judge is required")
    matrix = np.column_stack(logits)
    names = list(judges)
    if feature_family == "F2":
        interactions = []
        for left in range(len(judges)):
            for right in range(left + 1, len(judges)):
                interactions.append(matrix[:, left] * matrix[:, right])
                names.append(f"{judges[left]}_x_{judges[right]}")
        if interactions:
            matrix = np.column_stack([matrix, *interactions])
    return matrix, tuple(names)


@dataclass(frozen=True)
class FusionModel:
    feature_family: str
    feature_names: tuple[str, ...]
    judge_order: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    penalty: str
    c_value: float
    l1_ratio: float | None

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("Prediction feature matrix has the wrong shape")
        standardized = (values - self.mean) / self.scale
        return expit(standardized @ self.coefficients + self.intercept)

    def predict(
        self, calibrated_probabilities: Mapping[str, Sequence[float]]
    ) -> np.ndarray:
        features, names = calibrated_feature_matrix(
            calibrated_probabilities, self.judge_order, self.feature_family
        )
        if names != self.feature_names:
            raise RuntimeError("Frozen feature schema does not match")
        return self.predict_features(features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_family": self.feature_family,
            "feature_names": list(self.feature_names),
            "judge_order": list(self.judge_order),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "penalty": self.penalty,
            "C": self.c_value,
            "l1_ratio": self.l1_ratio,
        }


def fusion_model_from_dict(value: Mapping[str, Any]) -> FusionModel:
    return FusionModel(
        feature_family=str(value["feature_family"]),
        feature_names=tuple(str(item) for item in value["feature_names"]),
        judge_order=tuple(str(item) for item in value["judge_order"]),
        mean=np.asarray(value["mean"], dtype=np.float64),
        scale=np.asarray(value["scale"], dtype=np.float64),
        coefficients=np.asarray(value["coefficients"], dtype=np.float64),
        intercept=float(value["intercept"]),
        penalty=str(value["penalty"]),
        c_value=float(value["C"]),
        l1_ratio=(
            None if value.get("l1_ratio") is None else float(value["l1_ratio"])
        ),
    )


def fit_fusion_model(
    features: np.ndarray,
    targets: Sequence[int | bool],
    groups: Sequence[str],
    *,
    feature_family: str,
    feature_names: Sequence[str],
    judge_order: Sequence[str],
    penalty: str,
    c_value: float,
    l1_ratio: float | None,
    seed: int,
) -> FusionModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] != len(feature_names):
        raise ValueError("Fusion training inputs do not align")
    if penalty not in {"l2", "elasticnet"}:
        raise ValueError(f"Unsupported penalty: {penalty}")
    if c_value <= 0:
        raise ValueError("C must be positive")
    if penalty == "elasticnet" and (
        l1_ratio is None or not 0.0 < l1_ratio < 1.0
    ):
        raise ValueError("Elastic-net requires 0 < l1_ratio < 1")
    weights = cluster_weights(groups) * len(y)
    normalized_weights = weights / weights.sum()
    mean = np.average(x, axis=0, weights=normalized_weights)
    variance = np.average(np.square(x - mean), axis=0, weights=normalized_weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (x - mean) / scale
    estimator = LogisticRegression(
        penalty=penalty,
        C=float(c_value),
        solver="lbfgs" if penalty == "l2" else "saga",
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        max_iter=10000,
        tol=1e-9,
        random_state=int(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(standardized, y, sample_weight=weights)
    return FusionModel(
        feature_family=feature_family,
        feature_names=tuple(str(value) for value in feature_names),
        judge_order=tuple(str(value) for value in judge_order),
        mean=mean,
        scale=scale,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
        penalty=penalty,
        c_value=float(c_value),
        l1_ratio=None if penalty == "l2" else float(l1_ratio),
    )


def fit_calibrators_for_split(
    raw_probabilities: Mapping[str, Sequence[float]],
    targets: Sequence[int | bool],
    groups: Sequence[str],
    judge_order: Sequence[str],
    calibrator_methods: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    parameters = {}
    calibrated = {}
    for judge in judge_order:
        selected = fit_calibrator(
            calibrator_methods[judge],
            raw_probabilities[judge],
            targets,
            groups,
        )
        parameters[judge] = selected
        calibrated[judge] = apply_calibrator(
            selected, raw_probabilities[judge]
        )
    return parameters, calibrated


def apply_calibrators(
    raw_probabilities: Mapping[str, Sequence[float]],
    parameters: Mapping[str, Mapping[str, Any]],
    judge_order: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        judge: apply_calibrator(
            dict(parameters[judge]), raw_probabilities[judge]
        )
        for judge in judge_order
    }


def fit_calibrated_fusion(
    raw_probabilities: Mapping[str, Sequence[float]],
    targets: Sequence[int | bool],
    groups: Sequence[str],
    judge_order: Sequence[str],
    calibrator_methods: Mapping[str, str],
    hyperparameters: Mapping[str, Any],
    *,
    feature_family: str,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], FusionModel]:
    parameters, calibrated = fit_calibrators_for_split(
        raw_probabilities,
        targets,
        groups,
        judge_order,
        calibrator_methods,
    )
    features, names = calibrated_feature_matrix(
        calibrated, judge_order, feature_family
    )
    model = fit_fusion_model(
        features,
        targets,
        groups,
        feature_family=feature_family,
        feature_names=names,
        judge_order=judge_order,
        penalty=str(hyperparameters["penalty"]),
        c_value=float(hyperparameters["C"]),
        l1_ratio=hyperparameters.get("l1_ratio"),
        seed=seed,
    )
    return parameters, model


def predict_calibrated_fusion(
    raw_probabilities: Mapping[str, Sequence[float]],
    calibrator_parameters: Mapping[str, Mapping[str, Any]],
    model: FusionModel,
) -> np.ndarray:
    calibrated = apply_calibrators(
        raw_probabilities, calibrator_parameters, model.judge_order
    )
    return model.predict(calibrated)
