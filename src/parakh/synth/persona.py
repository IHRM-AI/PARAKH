from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = [
    "Retail trade",
    "Textiles",
    "Auto components",
    "Food processing",
    "Pharma distribution",
    "Logistics",
    "Light engineering",
]
STATES = ["MH", "GJ", "TN", "KA", "UP", "MP", "RJ", "WB"]

GST_FEATURES = [
    "gst_avg_monthly_turnover",
    "gst_turnover_growth",
    "gst_turnover_volatility",
    "gst_filing_punctuality",
    "gst_turnover_decline_3m",
]
BANK_FEATURES = [
    "bank_avg_monthly_credits",
    "bank_balance_dip_count",
    "bank_bounce_count",
    "bank_cash_buffer_days",
    "bank_credit_debit_ratio",
    "xf_gst_bank_gap",
]
EPFO_FEATURES = ["epfo_headcount", "epfo_headcount_trend"]

TARGET = "default"


def generate(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic MSME population with a 12-month default label.

    Default propensity depends on drivers observable across GST, bank and EPFO
    data, so each consented source adds genuine predictive signal.
    """
    rng = np.random.default_rng(seed)
    quality = rng.beta(2.2, 2.2, n)

    # Independent per-source stress shocks (higher means more stressed). Each shock
    # drives its source's observed features and the default label, so a consented
    # source adds signal that the others cannot fully recover.
    gst_shock = rng.normal(0, 1, n)
    bank_shock = rng.normal(0, 1, n)
    epfo_shock = rng.normal(0, 1, n)

    size = np.exp(rng.normal(13.0, 0.8, n))
    turnover = size * (1 + rng.normal(0, 0.05, n))
    growth = (quality - 0.5) * 0.40 - 0.09 * gst_shock + rng.normal(0, 0.06, n)
    volatility = np.clip(0.14 + 0.06 * gst_shock + (1 - quality) * 0.15, 0.02, 1.0)
    decline_3m = np.clip(0.09 * gst_shock - (quality - 0.5) * 0.18 + rng.normal(0, 0.05, n), -0.5, 0.8)
    punctuality = np.clip(
        np.round(12 * np.clip(quality - 0.16 * gst_shock + rng.normal(0, 0.08, n), 0, 1)), 0, 12
    )

    reconciliation_gap = np.clip(
        0.08 + 0.11 * bank_shock + (0.5 - quality) * 0.18 + rng.normal(0, 0.04, n), 0.0, 0.85
    )
    bank_credits = turnover * (1 - reconciliation_gap)
    balance_dips = rng.poisson(np.clip(1.4 + 0.9 * bank_shock + (1 - quality) * 1.6, 0.05, None))
    bounces = rng.poisson(np.clip(0.5 + 0.5 * bank_shock + (1 - quality) * 0.9, 0.03, None))
    cash_buffer = np.clip(40 - 8 * bank_shock + quality * 12 + rng.normal(0, 4, n), 1, 120)
    credit_debit_ratio = 1 - 0.09 * bank_shock + (quality - 0.5) * 0.30 + rng.normal(0, 0.06, n)

    headcount = np.clip(rng.poisson(np.clip(quality * 22, 1, None)), 1, None)
    headcount_trend = -0.10 * epfo_shock + (quality - 0.5) * 0.14 + rng.normal(0, 0.05, n)

    logit = (
        -1.75
        + 0.95 * gst_shock
        + 0.95 * bank_shock
        + 0.75 * epfo_shock
        + 1.6 * (0.5 - quality)
        + rng.normal(0, 0.4, n)
    )
    default = rng.binomial(1, _sigmoid(logit))

    return pd.DataFrame(
        {
            "firm_id": [f"MSME{100000 + i}" for i in range(n)],
            "sector": rng.choice(SECTORS, n),
            "state": rng.choice(STATES, n),
            "vintage_years": np.clip(rng.gamma(3.0, 1.5, n), 0.5, 25).round(1),
            "gst_avg_monthly_turnover": turnover.round(0),
            "gst_turnover_growth": growth.round(4),
            "gst_turnover_volatility": volatility.round(4),
            "gst_filing_punctuality": punctuality.astype(int),
            "gst_turnover_decline_3m": decline_3m.round(4),
            "bank_avg_monthly_credits": bank_credits.round(0),
            "bank_balance_dip_count": balance_dips,
            "bank_bounce_count": bounces,
            "bank_cash_buffer_days": cash_buffer.round(1),
            "bank_credit_debit_ratio": credit_debit_ratio.round(4),
            "xf_gst_bank_gap": reconciliation_gap.round(4),
            "epfo_headcount": headcount,
            "epfo_headcount_trend": headcount_trend.round(4),
            TARGET: default,
        }
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
