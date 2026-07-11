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


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, resamples: int = 500, seed: int = 42
) -> tuple[float, float]:
    """95% percentile bootstrap confidence interval for AUC over case resampling."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = y_true.size
    aucs: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        sample = y_true[idx]
        if sample.min() == sample.max():
            continue
        aucs.append(roc_auc_score(sample, y_score[idx]))
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def reliability_curve(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Calibration curve as equal-width bins of predicted versus observed default rate."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(y_score, edges[1:-1]), 0, bins - 1)
    curve: list[dict[str, float]] = []
    for b in range(bins):
        mask = assignment == b
        count = int(mask.sum())
        if count == 0:
            continue
        curve.append(
            {
                "bin": b,
                "count": count,
                "mean_predicted": round(float(y_score[mask].mean()), 4),
                "observed_default_rate": round(float(y_true[mask].mean()), 4),
            }
        )
    return curve
