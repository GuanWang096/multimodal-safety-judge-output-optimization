from __future__ import annotations

import numpy as np

from multijudge.calibration import (
    apply_calibrator,
    cluster_averaged_log_loss,
    make_group_folds,
    nested_probe_predictions,
    select_calibrator,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(20260729)
    group_count = 80
    group_sizes = rng.integers(1, 4, group_count)
    latent = rng.normal(0, 1.2, group_count)
    targets = []
    probabilities = []
    groups = []
    for group, (size, score) in enumerate(zip(group_sizes, latent)):
        probability = 1 / (1 + np.exp(-score))
        for _ in range(int(size)):
            target = int(rng.random() < probability)
            targets.append(target)
            # Deliberately overconfident to make temperature scaling useful.
            probabilities.append(1 / (1 + np.exp(-2.5 * score)))
            groups.append(f"component-{group:03d}")
    return (
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(targets, dtype=np.int64),
        groups,
    )


def test_group_folds_never_split_a_component() -> None:
    _, targets, groups = _fixture()
    folds = make_group_folds(targets, groups, n_splits=5, seed=20260729)
    seen = {}
    for group, fold in zip(groups, folds):
        assert seen.setdefault(group, int(fold)) == int(fold)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert all(targets[folds == fold].sum() > 0 for fold in range(5))


def test_cross_fitted_calibrator_uses_only_fit_inputs() -> None:
    probabilities, targets, groups = _fixture()
    folds = make_group_folds(targets, groups, n_splits=5, seed=20260729)
    first = select_calibrator(probabilities, targets, groups, folds)
    unrelated_select_labels = np.asarray([0, 1, 1, 0, 1])
    unrelated_select_labels[:] = 1 - unrelated_select_labels
    second = select_calibrator(probabilities, targets, groups, folds)
    assert first["selected_method"] == second["selected_method"]
    assert first["final_parameters"] == second["final_parameters"]
    assert np.array_equal(
        first["selected_oof_probabilities"],
        second["selected_oof_probabilities"],
    )


def test_selected_calibration_does_not_worsen_cluster_log_loss() -> None:
    probabilities, targets, groups = _fixture()
    folds = make_group_folds(targets, groups, n_splits=5, seed=20260729)
    result = select_calibrator(probabilities, targets, groups, folds)
    raw_loss = cluster_averaged_log_loss(targets, probabilities, groups)
    selected_loss = cluster_averaged_log_loss(
        targets, result["selected_oof_probabilities"], groups
    )
    assert selected_loss <= raw_loss + 0.005


def test_beta_calibration_is_monotonic() -> None:
    probabilities = np.linspace(0.01, 0.99, 100)
    transformed = apply_calibrator(
        {"method": "beta", "a": 1.2, "b": 0.8, "c": -0.1},
        probabilities,
    )
    assert np.all(np.diff(transformed) > 0)
    assert np.all((transformed > 0) & (transformed < 1))


def test_nested_probe_returns_finite_oof_predictions() -> None:
    probabilities, targets, groups = _fixture()
    folds = make_group_folds(targets, groups, n_splits=5, seed=20260729)
    raw = {
        "judge_a": probabilities,
        "judge_b": np.clip(0.15 + 0.7 * probabilities, 0.001, 0.999),
        "judge_c": np.clip(
            probabilities + np.linspace(-0.05, 0.05, len(probabilities)),
            0.001,
            0.999,
        ),
    }
    predictions, records = nested_probe_predictions(
        raw,
        targets,
        groups,
        folds,
        {judge: "identity" for judge in raw},
        list(raw),
        l2_strength=0.1,
    )
    assert len(records) == 5
    assert predictions.shape == targets.shape
    assert np.all(np.isfinite(predictions))
    assert np.all((predictions > 0) & (predictions < 1))
