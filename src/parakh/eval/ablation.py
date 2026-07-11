from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from parakh.eval.delong import delong_auc_delta
from parakh.eval.metrics import bootstrap_auc_ci
from parakh.scoring.model import HealthModel
from parakh.synth.persona import BANK_FEATURES, EPFO_FEATURES, GST_FEATURES, TARGET

STAGES: list[tuple[str, list[str]]] = [
    ("GST only", GST_FEATURES),
    ("+ AA bank statements", GST_FEATURES + BANK_FEATURES),
    ("+ EPFO", GST_FEATURES + BANK_FEATURES + EPFO_FEATURES),
]

ALL_FEATURES = GST_FEATURES + BANK_FEATURES + EPFO_FEATURES


def _three_way(n: int, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train, calibrate and test folds.

    The calibrate and test folds each take ``valid_fraction`` of the population.
    Calibration is fit on the calibrate fold and every reported metric is read
    off the untouched test fold.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    hold = int(n * valid_fraction)
    test_idx = order[:hold]
    calib_idx = order[hold : 2 * hold]
    train_idx = order[2 * hold :]
    return train_idx, calib_idx, test_idx


def source_ablation(df: pd.DataFrame, valid_fraction: float = 0.2, seed: int = 42) -> dict[str, float]:
    """Test-fold AUC as consented data sources are stacked. Each source should add signal."""
    train_idx, calib_idx, test_idx = _three_way(len(df), valid_fraction, seed)
    y = df[TARGET].to_numpy()

    results: dict[str, float] = {}
    for name, columns in STAGES:
        x = df[columns]
        model = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[calib_idx], y[calib_idx])
        auc = roc_auc_score(y[test_idx], model.predict_pd(x.iloc[test_idx]))
        results[name] = round(float(auc), 3)
    return results


def source_ablation_ci(
    df: pd.DataFrame,
    valid_fraction: float = 0.2,
    seed: int = 42,
    resamples: int = 500,
) -> dict[str, dict[str, float]]:
    """Source-ablation ladder with a 95% bootstrap confidence interval per rung."""
    train_idx, calib_idx, test_idx = _three_way(len(df), valid_fraction, seed)
    y = df[TARGET].to_numpy()

    results: dict[str, dict[str, float]] = {}
    for name, columns in STAGES:
        x = df[columns]
        model = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[calib_idx], y[calib_idx])
        pd_test = model.predict_pd(x.iloc[test_idx])
        auc = roc_auc_score(y[test_idx], pd_test)
        ci_low, ci_high = bootstrap_auc_ci(y[test_idx], pd_test, resamples=resamples, seed=seed)
        results[name] = {
            "auc": round(float(auc), 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
        }
    return results


def model_comparison(df: pd.DataFrame, valid_fraction: float = 0.2, seed: int = 42) -> dict[str, float]:
    """Test-fold AUC for a plain logistic regression versus the gradient-boosted model.

    Reported to pre-empt the objection that a linear model would score as well;
    the boosted model earns its place only if it stays ahead here. The GBM-minus-
    logistic AUC delta carries a DeLong 95% CI and two-sided p-value on the shared
    test fold, so the significance of any lift is read off directly. When the CI
    brackets zero the lift is not statistically distinguishable and the deployed
    value is the calibrated score, not a black-box edge.
    """
    train_idx, calib_idx, test_idx = _three_way(len(df), valid_fraction, seed)
    fit_idx = np.concatenate([train_idx, calib_idx])
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    logistic.fit(x.iloc[fit_idx], y[fit_idx])
    log_pd = logistic.predict_proba(x.iloc[test_idx])[:, 1]
    log_auc = roc_auc_score(y[test_idx], log_pd)

    gbm = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[calib_idx], y[calib_idx])
    gbm_pd = gbm.predict_pd(x.iloc[test_idx])
    gbm_auc = roc_auc_score(y[test_idx], gbm_pd)

    delong = delong_auc_delta(y[test_idx], gbm_pd, log_pd)
    return {
        "logistic_auc": round(float(log_auc), 3),
        "gbm_auc": round(float(gbm_auc), 3),
        "gbm_lift": round(float(gbm_auc - log_auc), 3),
        "delta_ci_low": delong.delta_ci_low,
        "delta_ci_high": delong.delta_ci_high,
        "p_value": delong.p_value,
        "delta_ci_includes_zero": delong.includes_zero,
    }
