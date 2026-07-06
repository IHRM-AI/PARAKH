from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class ScoreReport:
    n: int
    default_rate: float
    auc: float
    gini: float
    ks: float
    brier: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_sorted = np.asarray(y_true)[order]
    positives = y_sorted.cumsum() / max(y_sorted.sum(), 1)
    negatives = (1 - y_sorted).cumsum() / max((1 - y_sorted).sum(), 1)
    return float(np.abs(positives - negatives).max())


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> ScoreReport:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = roc_auc_score(y_true, y_score)
    return ScoreReport(
        n=int(y_true.size),
        default_rate=float(y_true.mean()),
        auc=float(auc),
        gini=float(2 * auc - 1),
        ks=ks_statistic(y_true, y_score),
        brier=float(brier_score_loss(y_true, y_score)),
    )
