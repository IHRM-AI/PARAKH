from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DelongResult:
    auc_a: float
    auc_b: float
    delta_auc: float
    delta_ci_low: float
    delta_ci_high: float
    p_value: float
    includes_zero: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Column-wise midranks, averaging ranks over ties (fractional ranking)."""
    order = np.argsort(x, kind="mergesort")
    ranked = x[order]
    n = len(x)
    mid = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        mid[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = mid
    return out


def _structural_components(
    scores: np.ndarray, n_pos: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast DeLong structural components for a stack of predictors on one test set.

    ``scores`` is ``(k, n)`` with positives in the first ``n_pos`` columns. Returns
    the AUC vector, and the V10 / V01 placement matrices per Sun & Xu (2014).
    """
    m = n_pos
    n = scores.shape[1] - m
    pos = scores[:, :m]
    neg = scores[:, m:]
    k = scores.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(scores[r])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v10, v01


def delong_auc_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
) -> DelongResult:
    """DeLong test for the difference of two correlated ROC AUCs on the same labels.

    ``score_a`` and ``score_b`` are two predictors scored on identical
    ``y_true`` (higher score means higher predicted default). Returns each AUC,
    ``delta = auc_a - auc_b``, its 95% confidence interval, a two-sided p-value
    against the null of equal AUCs, and whether the interval brackets zero. The
    covariance uses the fast Sun & Xu (2014) algorithm.
    """
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if not (y_true.shape == score_a.shape == score_b.shape):
        raise ValueError("y_true, score_a and score_b must share one shape.")
    if set(np.unique(y_true).tolist()) - {0, 1}:
        raise ValueError("y_true must be binary 0/1.")
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        raise ValueError("y_true must contain both classes.")

    pos_mask = y_true == 1
    order = np.concatenate([np.where(pos_mask)[0], np.where(~pos_mask)[0]])
    stacked = np.vstack([score_a[order], score_b[order]])
    m = int(pos_mask.sum())
    n = int((~pos_mask).sum())

    aucs, v10, v01 = _structural_components(stacked, m)
    s10 = np.cov(v10)
    s01 = np.cov(v01)
    cov = s10 / m + s01 / n

    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))

    if se == 0.0:
        p_value = 1.0 if delta == 0.0 else 0.0
        ci_low = ci_high = delta
    else:
        z = delta / se
        p_value = float(2 * stats.norm.sf(abs(z)))
        half = 1.959963984540054 * se
        ci_low, ci_high = delta - half, delta + half

    return DelongResult(
        auc_a=round(float(aucs[0]), 4),
        auc_b=round(float(aucs[1]), 4),
        delta_auc=round(delta, 4),
        delta_ci_low=round(float(ci_low), 4),
        delta_ci_high=round(float(ci_high), 4),
        p_value=round(p_value, 4),
        includes_zero=bool(ci_low <= 0.0 <= ci_high),
    )
