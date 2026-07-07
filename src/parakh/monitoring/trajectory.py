from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def simulate(current_score: int, features: dict[str, float]) -> Trajectory:
    """Reconstruct a 12-month score trajectory ending at the current score.

    A stressed firm started healthier and slides towards its current score; the
    flag month is the first crossing below the watch threshold.
    """
    stress = _stress(features)
    start = int(np.clip(current_score + round(stress * 22), 5, 100))
    slope = (current_score - start) / (HORIZON - 1)
    scores = [int(round(np.clip(start + slope * t, 5, 100))) for t in range(HORIZON)]

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
        deteriorating=stress > 0.35,
    )
