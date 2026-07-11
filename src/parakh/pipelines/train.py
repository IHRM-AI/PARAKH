from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from parakh.config import settings
from parakh.eval.ablation import model_comparison, source_ablation, source_ablation_ci
from parakh.eval.metrics import ScoreReport, bootstrap_auc_ci, evaluate, reliability_curve
from parakh.scoring.model import HealthModel
from parakh.synth.persona import (
    BANK_FEATURES,
    COHORTS,
    EPFO_FEATURES,
    GST_FEATURES,
    TARGET,
    generate,
)

logger = logging.getLogger("parakh.train")

ALL_FEATURES = GST_FEATURES + BANK_FEATURES + EPFO_FEATURES


def _random_split(n: int, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    hold = int(n * valid_fraction)
    return order[2 * hold :], order[hold : 2 * hold], order[:hold]


def _fit_and_score(
    df: pd.DataFrame, train_idx: np.ndarray, calib_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[HealthModel, ScoreReport, np.ndarray]:
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[calib_idx], y[calib_idx])
    pd_test = model.predict_pd(x.iloc[test_idx])
    report = evaluate(y[test_idx], pd_test)
    return model, report, pd_test


def _report_with_uncertainty(
    report: ScoreReport, y_test: np.ndarray, pd_test: np.ndarray, seed: int
) -> dict[str, object]:
    ci_low, ci_high = bootstrap_auc_ci(y_test, pd_test, seed=seed)
    payload = report.as_dict()
    payload["auc_ci_low"] = round(ci_low, 4)
    payload["auc_ci_high"] = round(ci_high, 4)
    payload["reliability_curve"] = reliability_curve(y_test, pd_test, bins=10)
    return payload


def _oot_indices(df: pd.DataFrame, calib_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-time split: train on early cohorts, test on the latest cohorts.

    A slice of the early cohorts is held back for calibration so isotonic never
    sees the out-of-time test firms.
    """
    cutoff = int(COHORTS * 0.7)
    early = np.where(df["cohort"].to_numpy() < cutoff)[0]
    test_idx = np.where(df["cohort"].to_numpy() >= cutoff)[0]
    rng = np.random.default_rng(seed)
    early = rng.permutation(early)
    hold = int(len(early) * calib_fraction)
    return early[hold:], early[:hold], test_idx


def run(n: int = 8000) -> dict[str, object]:
    df = generate(n=n, seed=settings.random_seed)
    logger.info("generated %d firms, default rate %.3f", len(df), df[TARGET].mean())

    ladder = source_ablation(df, valid_fraction=settings.validation_fraction, seed=settings.random_seed)
    logger.info("source-ablation AUC ladder (held-out test fold): %s", ladder)

    ladder_ci = source_ablation_ci(df, valid_fraction=settings.validation_fraction, seed=settings.random_seed)
    logger.info("source-ablation ladder with 95%% CI: %s", ladder_ci)

    comparison = model_comparison(df, valid_fraction=settings.validation_fraction, seed=settings.random_seed)
    logger.info(
        "logistic vs GBM (test fold): logistic=%.3f gbm=%.3f lift=%+.3f",
        comparison["logistic_auc"], comparison["gbm_auc"], comparison["gbm_lift"],
    )

    y = df[TARGET].to_numpy()

    train_idx, calib_idx, test_idx = _random_split(len(df), settings.validation_fraction, settings.random_seed)
    model, report, pd_test = _fit_and_score(df, train_idx, calib_idx, test_idx)
    random_split = _report_with_uncertainty(report, y[test_idx], pd_test, settings.random_seed)
    logger.info(
        "random-split test AUC=%.3f [%.3f, %.3f] Gini=%.3f KS=%.3f Brier=%.3f",
        report.auc, random_split["auc_ci_low"], random_split["auc_ci_high"],
        report.gini, report.ks, report.brier,
    )

    oot_train, oot_calib, oot_test = _oot_indices(df, settings.validation_fraction, settings.random_seed)
    _, oot_report, oot_pd = _fit_and_score(df, oot_train, oot_calib, oot_test)
    out_of_time = _report_with_uncertainty(oot_report, y[oot_test], oot_pd, settings.random_seed)
    logger.info(
        "out-of-time test AUC=%.3f [%.3f, %.3f] Gini=%.3f KS=%.3f Brier=%.3f",
        oot_report.auc, out_of_time["auc_ci_low"], out_of_time["auc_ci_high"],
        oot_report.gini, oot_report.ks, oot_report.brier,
    )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.artifacts_dir / "health_model.joblib")
    artifact = {
        "source_ablation": ladder,
        "source_ablation_ci": ladder_ci,
        "model_comparison": comparison,
        "random_split": random_split,
        "out_of_time": out_of_time,
        "monotone_constraints": True,
    }
    (settings.artifacts_dir / "ablation.json").write_text(json.dumps(artifact, indent=2))
    return artifact


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
