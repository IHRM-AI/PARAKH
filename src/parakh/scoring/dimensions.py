from __future__ import annotations

import numpy as np
import pandas as pd

DIMENSIONS = [
    "Cash Flow",
    "GST Compliance",
    "Banking Discipline",
    "Growth",
    "Stability",
    "Leverage",
]


def _scale(series: pd.Series, low: float, high: float, invert: bool = False) -> pd.Series:
    normalized = ((series - low) / (high - low)).clip(0, 1)
    if invert:
        normalized = 1 - normalized
    return (normalized * 100).round().astype(int)


def sub_scores(row: pd.Series) -> dict[str, int]:
    """Interpretable 0-100 sub-scores for each health dimension."""
    frame = pd.to_numeric(row, errors="coerce").to_frame().T
    return {
        "Cash Flow": int(_scale(frame["bank_cash_buffer_days"], 1, 60).iloc[0]),
        "GST Compliance": int((frame["gst_filing_punctuality"].iloc[0] / 12 * 100)),
        "Banking Discipline": int(
            _scale(frame["bank_bounce_count"] + frame["bank_balance_dip_count"], 0, 10, invert=True).iloc[0]
        ),
        "Growth": int(_scale(frame["gst_turnover_growth"], -0.2, 0.3).iloc[0]),
        "Stability": int(_scale(frame["gst_turnover_volatility"], 0.02, 0.6, invert=True).iloc[0]),
        "Leverage": int(_scale(frame["xf_gst_bank_gap"], 0.0, 0.6, invert=True).iloc[0]),
    }


def grade(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 68:
        return "B+"
    if score >= 55:
        return "B"
    if score >= 45:
        return "C"
    return "D"


def prequalified_limit(score: int, avg_monthly_credits: float) -> float:
    multiple = np.interp(score, [40, 55, 68, 80, 100], [0.0, 0.6, 1.1, 1.6, 2.2])
    return round(avg_monthly_credits * multiple, -3)
