from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from multijudge.calibration import make_group_folds
from multijudge.fusion import (
    calibrated_feature_matrix,
    fit_fusion_model,
    fusion_model_from_dict,
)


SCRIPT_DIR = Path(__file__).resolve().parents[1]
CONFIG = SCRIPT_DIR / "configs" / "softfusion_model_selection_v1.json"


def _load_script(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    rng = np.random.default_rng(20260729)
    group_count = 50
    groups = [f"component-{index:03d}" for index in range(group_count) for _ in range(2)]
    latent = np.repeat(rng.normal(size=group_count), 2)
    targets = (latent + rng.normal(scale=0.7, size=len(latent)) > 0).astype(int)
    raw = {
        "qwen": 1 / (1 + np.exp(-(latent + rng.normal(scale=0.5, size=len(latent))))),
        "safework": 1 / (1 + np.exp(-(0.8 * latent + rng.normal(scale=0.7, size=len(latent))))),
        "minicpm": 1 / (1 + np.exp(-(0.6 * latent + rng.normal(scale=0.9, size=len(latent))))),
    }
    return raw, targets, groups


def test_frozen_selection_config_is_canonical_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["feature_scope"] == "canonical_only"
    assert config["feature_families"] == ["F1", "F2"]
    assert config["confirmatory_labels_accessed"] is False


def test_feature_families_have_expected_columns() -> None:
    raw, _, _ = _fixture()
    f1, f1_names = calibrated_feature_matrix(raw, tuple(raw), "F1")
    f2, f2_names = calibrated_feature_matrix(raw, tuple(raw), "F2")
    assert f1.shape[1] == 3
    assert f2.shape[1] == 6
    assert f2_names[:3] == f1_names
    assert f2_names[3:] == (
        "qwen_x_safework",
        "qwen_x_minicpm",
        "safework_x_minicpm",
    )


def test_cross_fitted_fusion_preserves_component_boundaries() -> None:
    runner = _load_script("fit_softfusion.py")
    raw, targets, groups = _fixture()
    folds = make_group_folds(targets, groups, n_splits=5, seed=20260729)
    for group in set(groups):
        members = [index for index, value in enumerate(groups) if value == group]
        assert len(set(int(folds[index]) for index in members)) == 1
    predictions, records = runner.cross_fitted_predictions(
        raw_probabilities=raw,
        targets=targets,
        groups=groups,
        folds=folds,
        judges=tuple(raw),
        calibrator_methods={judge: "identity" for judge in raw},
        feature_family="F1",
        hyperparameters={"penalty": "l2", "C": 0.1, "l1_ratio": None},
        seed=20260729,
    )
    assert np.all(np.isfinite(predictions))
    assert len(records) == 5
    assert sum(value["validation_record_count"] for value in records) == len(targets)


def test_fusion_fit_is_deterministic() -> None:
    raw, targets, groups = _fixture()
    features, names = calibrated_feature_matrix(raw, tuple(raw), "F2")
    kwargs = dict(
        feature_family="F2",
        feature_names=names,
        judge_order=tuple(raw),
        penalty="elasticnet",
        c_value=0.1,
        l1_ratio=0.5,
        seed=20260729,
    )
    first = fit_fusion_model(features, targets, groups, **kwargs)
    second = fit_fusion_model(features, targets, groups, **kwargs)
    assert np.allclose(first.coefficients, second.coefficients, atol=1e-10)
    assert abs(first.intercept - second.intercept) < 1e-10
    restored = fusion_model_from_dict(first.to_dict())
    assert np.allclose(first.predict_features(features), restored.predict_features(features))


def test_default_sf3_inputs_do_not_include_confirmatory_manifest() -> None:
    runner = _load_script("fit_softfusion.py")
    args = runner.build_parser().parse_args([])
    paths = [
        args.fit_manifest,
        args.select_manifest,
        args.soft_root,
        args.calibrators,
        args.screening_decision,
        args.selection_config,
        args.output_dir,
    ]
    assert all("confirm" not in str(path).lower() for path in paths)
