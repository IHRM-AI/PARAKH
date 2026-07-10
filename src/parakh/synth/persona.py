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

COHORTS = 18


def generate(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic MSME population with a 12-month default label.

    The label is deliberately not a linear read-off of the observed features:

    * A hidden ``management`` latent moves default risk but is measured by no
      single feature, leaving irreducible variance a linear model on features
      cannot recover.
    * Each source carries its own driver. GST reflects demand, bank statements
      reflect a distinct liquidity-stress latent, and EPFO reflects workforce
      stability. Bank credits are turnover plus independent off-book cash and
      noise rather than a deterministic function of turnover.
    * Default risk combines these through thresholds and interactions, so a
      gradient-boosted model has a legitimate edge over logistic regression.

    Each firm is stamped with an origination cohort; later cohorts face a
    harsher macro backdrop, which supports an out-of-time evaluation split.
    """
    rng = np.random.default_rng(seed)

    quality = rng.beta(2.2, 2.2, n)
    management = rng.normal(0, 1, n)
    liquidity_stress = rng.normal(0, 1, n)
    workforce_stability = rng.normal(0, 1, n)
    demand_shock = rng.normal(0, 1, n)

    cohort = rng.integers(0, COHORTS, n)
    macro = (cohort / (COHORTS - 1)) - 0.5
    macro_stress = rng.normal(0.9 * macro, 0.35)

    size = np.exp(rng.normal(13.0, 0.8, n))
    turnover = size * (1 + rng.normal(0, 0.05, n))
    growth = (quality - 0.5) * 0.40 + 0.10 * demand_shock - 0.12 * macro_stress + rng.normal(0, 0.06, n)
    volatility = np.clip(0.14 - 0.05 * demand_shock + (1 - quality) * 0.15 + 0.04 * macro_stress, 0.02, 1.0)
    decline_3m = np.clip(
        -0.10 * demand_shock - (quality - 0.5) * 0.18 + 0.08 * macro_stress + rng.normal(0, 0.05, n),
        -0.5,
        0.8,
    )
    punctuality = np.clip(
        np.round(12 * np.clip(quality + 0.10 * management + rng.normal(0, 0.08, n), 0, 1)), 0, 12
    )

    off_book = rng.normal(0, 0.09, n)
    reported_share = np.clip(0.9 - 0.12 * liquidity_stress + off_book, 0.35, 1.15)
    bank_credits = turnover * reported_share * (1 + rng.normal(0, 0.06, n))
    reconciliation_gap = np.clip(1 - bank_credits / turnover, 0.0, 0.85)

    balance_dips = rng.poisson(np.clip(1.2 + 1.0 * liquidity_stress + 0.6 * macro_stress, 0.05, None))
    bounces = rng.poisson(np.clip(0.4 + 0.6 * liquidity_stress + 0.3 * macro_stress, 0.03, None))
    cash_buffer = np.clip(42 - 11 * liquidity_stress + quality * 8 + rng.normal(0, 4, n), 1, 120)
    credit_debit_ratio = 1 - 0.11 * liquidity_stress + (quality - 0.5) * 0.12 + rng.normal(0, 0.06, n)

    headcount = np.clip(rng.poisson(np.clip(quality * 22, 1, None)), 1, None)
    headcount_trend = -0.12 * workforce_stability + (quality - 0.5) * 0.08 + rng.normal(0, 0.05, n)

    thin_buffer = (cash_buffer < 25).astype(float)
    demand_break = (decline_3m > 0.02).astype(float)
    late_filing = (punctuality < 11).astype(float)
    shrinking = (growth < 0).astype(float)
    high_volatility = (volatility > 0.32).astype(float)
    attrition = np.clip(-headcount_trend, 0, 0.6)

    logit = (
        -3.05
        + 0.40 * liquidity_stress
        + 0.28 * demand_shock
        + 0.22 * workforce_stability
        + 0.40 * (0.5 - quality)
        + 0.95 * management
        + 0.42 * macro_stress
        + 0.80 * volatility
        + 2.8 * thin_buffer * demand_break
        + 2.1 * late_filing * shrinking
        + 1.9 * attrition
        + 1.7 * high_volatility * thin_buffer
        + rng.normal(0, 0.35, n)
    )
    default = rng.binomial(1, _sigmoid(logit))

    return pd.DataFrame(
        {
            "firm_id": [f"MSME{100000 + i}" for i in range(n)],
            "sector": rng.choice(SECTORS, n),
            "state": rng.choice(STATES, n),
            "cohort": cohort.astype(int),
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
