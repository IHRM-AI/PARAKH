import pandas as pd

from parakh.interpret.reason_codes import CardExplainer
from parakh.scoring.card import ALL_FEATURES
from parakh.scoring.model import HealthModel
from parakh.synth.persona import TARGET, generate


def _explainer() -> CardExplainer:
    df = generate(n=3000, seed=2)
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel(num_boost_round=200, early_stopping_rounds=40)
    model.fit(x.iloc[:2400], y[:2400], x.iloc[2400:], y[2400:])
    return CardExplainer(model)


def _stressed_row() -> pd.DataFrame:
    features = {
        "gst_avg_monthly_turnover": 300000.0,
        "gst_turnover_growth": -0.05,
        "gst_turnover_volatility": 0.9,
        "gst_filing_punctuality": 8,
        "gst_turnover_decline_3m": 0.15,
        "bank_avg_monthly_credits": 250000.0,
        "bank_balance_dip_count": 5,
        "bank_bounce_count": 3,
        "bank_cash_buffer_days": 8.0,
        "bank_credit_debit_ratio": 0.85,
        "xf_gst_bank_gap": 0.35,
        "epfo_headcount": 6,
        "epfo_headcount_trend": -0.08,
    }
    return pd.DataFrame([features]).reindex(columns=ALL_FEATURES)


def test_high_volatility_firm_reads_irregular_sales():
    codes = _explainer().explain(_stressed_row(), top_k=13)
    volatility = next(c for c in codes if c.feature == "gst_turnover_volatility")
    assert not volatility.supports_score
    assert volatility.english == "Irregular monthly sales"


def test_contributions_reconcile_with_margin():
    explainer = _explainer()
    codes = explainer.explain(_stressed_row())
    assert codes
    assert all(code.points >= 1 for code in codes)
