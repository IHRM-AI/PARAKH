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
        train_set = lgb.Dataset(x_train, label=y_train)
        valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
        self.booster = lgb.train(
            PARAMS,
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
