from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HEADCOUNT_BANDS: list[tuple[str, int, int]] = [
    ("micro (1-9)", 1, 9),
    ("small (10-24)", 10, 24),
    ("mid (25+)", 25, 10**9),
]


def _headcount_band(headcount: pd.Series) -> pd.Series:
    labels = pd.Series(index=headcount.index, dtype=object)
    for name, low, high in HEADCOUNT_BANDS:
        labels[(headcount >= low) & (headcount <= high)] = name
    return labels


def _slice_metrics(
    y_true: np.ndarray, pd_score: np.ndarray, approved: np.ndarray
) -> dict[str, float | int | None]:
    n = int(len(y_true))
    both_classes = 0 < y_true.sum() < n
    return {
        "n": n,
        "auc": round(float(roc_auc_score(y_true, pd_score)), 4) if both_classes else None,
        "approval_rate": round(float(approved.mean()), 4),
        "observed_default_rate": round(float(y_true.mean()), 4),
    }


def _dimension(
    frame: pd.DataFrame, group_col: str, y_true: np.ndarray, pd_score: np.ndarray, approved: np.ndarray
) -> dict[str, object]:
    slices: dict[str, dict[str, float | int | None]] = {}
    groups = frame[group_col]
    for value in sorted(str(v) for v in groups.dropna().unique()):
        mask = (groups == value).to_numpy()
        slices[value] = _slice_metrics(y_true[mask], pd_score[mask], approved[mask])

    rates = [s["approval_rate"] for s in slices.values() if s["n"] > 0]
    max_rate = max(rates) if rates else 0.0
    di_ratio = round(min(rates) / max_rate, 4) if max_rate > 0 else None

    aucs = {k: s["auc"] for k, s in slices.items() if s["auc"] is not None}
    weakest = min(aucs, key=aucs.get) if aucs else None

    return {
        "slices": slices,
        "disparate_impact_ratio": di_ratio,
        "min_approval_rate": round(min(rates), 4) if rates else None,
        "max_approval_rate": round(max_rate, 4) if rates else None,
        "weakest_auc_slice": weakest,
        "weakest_auc": aucs[weakest] if weakest else None,
    }


def fairness_slices(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    pd_score: np.ndarray,
    approval_cutoff: float = 0.25,
) -> dict[str, object]:
    """Sliced AUC, approval rate and observed default rate on a held-out fold.

    ``frame`` must carry ``sector``, ``state`` and ``epfo_headcount`` columns
    aligned to ``y_true`` and ``pd_score``. Slices by sector, state and three
    headcount-size bands. The disparate-impact ratio per dimension is the minimum
    slice approval rate over the maximum (1.0 = parity; the four-fifths rule flags
    below 0.80). Approval uses the deployed calibrated-PD sanction cutoff. Values
    are reported as measured.
    """
    y_true = np.asarray(y_true).astype(int)
    pd_score = np.asarray(pd_score, dtype=float)
    approved = pd_score < approval_cutoff

    work = frame.copy()
    work["_headcount_band"] = _headcount_band(work["epfo_headcount"])

    return {
        "approval_cutoff": approval_cutoff,
        "sector": _dimension(work, "sector", y_true, pd_score, approved),
        "state": _dimension(work, "state", y_true, pd_score, approved),
        "headcount_band": _dimension(work, "_headcount_band", y_true, pd_score, approved),
    }
