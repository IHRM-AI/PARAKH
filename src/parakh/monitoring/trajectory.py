from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from parakh.scoring.model import HealthModel, pd_to_score

HORIZON = 12
WATCH_THRESHOLD = 55
MONTHS = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]


@dataclass(frozen=True)
class Trajectory:
    months: list[str]
    scores: list[int]
    flag_month: int | None
    watch_threshold: int
    deteriorating: bool


def _stress(features: dict[str, float]) -> float:
    decline = max(0.0, float(features.get("gst_turnover_decline_3m", 0.0)))
    headcount = max(0.0, -float(features.get("epfo_headcount_trend", 0.0)))
    gap = max(0.0, float(features.get("xf_gst_bank_gap", 0.0)) - 0.1)
    return float(np.clip(decline * 3.0 + headcount * 2.0 + gap * 1.5, 0.0, 1.0))


def _month_snapshot(features: dict[str, float], relief: float) -> dict[str, float]:
    """Roll the observed features back by ``relief`` of one firm's accumulated stress.

    ``relief`` runs from 1 at the first month (healthiest prior state) to 0 at the
    latest month (the observed snapshot). Each source is unwound along its own
    axis so the resulting snapshot is a coherent input the model can re-score.
    """
    snapshot = dict(features)
    decline = float(features.get("gst_turnover_decline_3m", 0.0))
    snapshot["gst_turnover_decline_3m"] = decline - relief * max(decline, 0.0) * 1.4
    snapshot["gst_turnover_growth"] = float(features.get("gst_turnover_growth", 0.0)) + relief * 0.10
    snapshot["gst_turnover_volatility"] = max(
        0.02, float(features.get("gst_turnover_volatility", 0.14)) - relief * 0.06
    )
    snapshot["bank_cash_buffer_days"] = float(features.get("bank_cash_buffer_days", 30.0)) + relief * 18.0
    snapshot["bank_balance_dip_count"] = max(
        0.0, float(features.get("bank_balance_dip_count", 0.0)) - round(relief * 2.0)
    )
    snapshot["bank_bounce_count"] = max(
        0.0, float(features.get("bank_bounce_count", 0.0)) - round(relief * 1.0)
    )
    snapshot["bank_credit_debit_ratio"] = float(features.get("bank_credit_debit_ratio", 1.0)) + relief * 0.10
    gap = float(features.get("xf_gst_bank_gap", 0.0))
    snapshot["xf_gst_bank_gap"] = max(0.0, gap - relief * max(gap - 0.1, 0.0))
    trend = float(features.get("epfo_headcount_trend", 0.0))
    snapshot["epfo_headcount_trend"] = trend - relief * min(trend, 0.0)
    return snapshot


def simulate(model: HealthModel, features: dict[str, float]) -> Trajectory:
    """Re-score 12 monthly feature snapshots through the model.

    Each month applies a share of the firm's accumulated per-source stress, then
    calls ``model.predict_pd`` on that snapshot. The trajectory is the sequence of
    model scores and ``flag_month`` is the first month the model score crosses
    below the watch threshold, not an interpolated line.
    """
    stress = _stress(features)
    scores: list[int] = []
    for t in range(HORIZON):
        relief = stress * (1 - t / (HORIZON - 1))
        snapshot = _month_snapshot(features, relief)
        frame = pd.DataFrame([snapshot]).reindex(columns=model.feature_names)
        pd_value = float(model.predict_pd(frame)[0])
        scores.append(int(pd_to_score(pd_value)))

    flag_month: int | None = None
    for index, value in enumerate(scores):
        if value < WATCH_THRESHOLD:
            flag_month = index + 1
            break

    return Trajectory(
        months=MONTHS,
        scores=scores,
        flag_month=flag_month,
        watch_threshold=WATCH_THRESHOLD,
        deteriorating=scores[-1] < scores[0] and stress > 0.35,
    )
