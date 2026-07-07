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


def pd_to_score(pd_value: float | np.ndarray) -> np.ndarray:
    """Map a probability of default to a 0-100 health score.

    A linear risk-to-score mapping bounded to [10, 95] so that even a very low
    predicted PD does not read as a perfect, implausible 100.
    """
    return np.clip(np.round(100 - 130 * np.asarray(pd_value)), 10, 95).astype(int)
