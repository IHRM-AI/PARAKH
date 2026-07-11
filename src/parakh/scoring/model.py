from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 120,
    "verbosity": -1,
}

# Credit-sensible monotone direction of each feature on the probability of
# default. +1: higher feature raises PD; -1: higher feature lowers PD. The
# constraint makes the booster a glass-box scorecard: a reason-code direction
# can never flip against its documented credit meaning, which is what a model-
# risk reviewer needs to sign off. Any feature absent from this map is left
# unconstrained (0).
MONOTONE_DIRECTIONS: dict[str, int] = {
    # Higher is safer -> lower PD.
    "gst_avg_monthly_turnover": -1,
    "gst_turnover_growth": -1,
    "gst_filing_punctuality": -1,
    "bank_avg_monthly_credits": -1,
    "bank_cash_buffer_days": -1,
    "bank_credit_debit_ratio": -1,
    "epfo_headcount": -1,
    "epfo_headcount_trend": -1,
    # Higher is riskier -> higher PD.
    "gst_turnover_volatility": 1,
    "gst_turnover_decline_3m": 1,
    "bank_balance_dip_count": 1,
    "bank_bounce_count": 1,
    "xf_gst_bank_gap": 1,
}


def monotone_vector(feature_names: list[str]) -> list[int]:
    """Constraint vector aligned to ``feature_names`` for LightGBM's ``monotone_constraints``."""
    return [MONOTONE_DIRECTIONS.get(name, 0) for name in feature_names]


@dataclass
class HealthModel:
    num_boost_round: int = 800
    early_stopping_rounds: int = 60
    booster: lgb.Booster | None = None
    calibrator: IsotonicRegression | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_valid: pd.DataFrame,
        y_valid: np.ndarray,
    ) -> "HealthModel":
        self.feature_names = list(x_train.columns)
        params = {**PARAMS, "monotone_constraints": monotone_vector(self.feature_names)}
        train_set = lgb.Dataset(x_train, label=y_train)
        valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        raw = self._raw(x_valid)
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y_valid)
        return self

    def _raw(self, x: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not trained.")
        return self.booster.predict(x, num_iteration=self.booster.best_iteration)

    def predict_pd(self, x: pd.DataFrame) -> np.ndarray:
        raw = self._raw(x)
        return raw if self.calibrator is None else self.calibrator.predict(raw)


SCORE_REFERENCE = 55
ODDS_REFERENCE = 3.0
PDO = 15
SCORE_FLOOR = 10
SCORE_CEILING = 95


def pd_to_score(pd_value: float | np.ndarray) -> np.ndarray:
    """Map a probability of default to a 0-100 health score via a points-to-double-the-odds scale.

    Scores are affine in the log-odds of repayment. The scale is anchored so that
    a score of ``SCORE_REFERENCE`` (55) corresponds to odds of ``ODDS_REFERENCE``
    (3:1 good-to-bad, PD = 0.25, near the population default mean), and every
    ``PDO`` (15) points added doubles those odds. This keeps the score responsive
    across the whole realistic risk range: the previous ``100 - 130 * PD``
    collapsed every PD at or above 0.69 to the floor, whereas the log-odds form
    still separates firms at PD 0.1 (score 79), 0.3 (50) and 0.6 (22) before the
    [10, 95] display clip binds.
    """
    p = np.clip(np.asarray(pd_value, dtype=float), 1e-6, 1 - 1e-6)
    factor = PDO / np.log(2.0)
    offset = SCORE_REFERENCE - factor * np.log(ODDS_REFERENCE)
    score = offset + factor * np.log((1 - p) / p)
    return np.clip(np.round(score), SCORE_FLOOR, SCORE_CEILING).astype(int)
