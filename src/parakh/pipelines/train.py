from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from parakh.config import settings
from parakh.eval.ablation import model_comparison, source_ablation
from parakh.eval.metrics import ScoreReport, evaluate
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
) -> tuple[HealthModel, ScoreReport]:
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[calib_idx], y[calib_idx])
    report = evaluate(y[test_idx], model.predict_pd(x.iloc[test_idx]))
    return model, report


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

    comparison = model_comparison(df, valid_fraction=settings.validation_fraction, seed=settings.random_seed)
    logger.info(
        "logistic vs GBM (test fold): logistic=%.3f gbm=%.3f lift=%+.3f",
        comparison["logistic_auc"], comparison["gbm_auc"], comparison["gbm_lift"],
    )

    train_idx, calib_idx, test_idx = _random_split(len(df), settings.validation_fraction, settings.random_seed)
    model, report = _fit_and_score(df, train_idx, calib_idx, test_idx)
    logger.info(
        "random-split test AUC=%.3f Gini=%.3f KS=%.3f Brier=%.3f",
        report.auc, report.gini, report.ks, report.brier,
    )

    oot_train, oot_calib, oot_test = _oot_indices(df, settings.validation_fraction, settings.random_seed)
    _, oot_report = _fit_and_score(df, oot_train, oot_calib, oot_test)
    logger.info(
        "out-of-time test AUC=%.3f Gini=%.3f KS=%.3f Brier=%.3f",
        oot_report.auc, oot_report.gini, oot_report.ks, oot_report.brier,
    )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.artifacts_dir / "health_model.joblib")
    artifact = {
        "source_ablation": ladder,
        "model_comparison": comparison,
        "random_split": report.as_dict(),
        "out_of_time": oot_report.as_dict(),
    }
    (settings.artifacts_dir / "ablation.json").write_text(json.dumps(artifact, indent=2))
    return artifact


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
