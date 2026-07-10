from parakh.monitoring.trajectory import HORIZON, WATCH_THRESHOLD, simulate
from parakh.scoring.card import ALL_FEATURES
from parakh.scoring.model import HealthModel
from parakh.synth.persona import TARGET, generate


def _trained_model() -> HealthModel:
    df = generate(n=3000, seed=5)
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel(num_boost_round=200, early_stopping_rounds=40)
    model.fit(x.iloc[:2400], y[:2400], x.iloc[2400:], y[2400:])
    return model


def _healthy_firm() -> dict[str, float]:
    return {
        "gst_avg_monthly_turnover": 600000.0,
        "gst_turnover_growth": 0.10,
        "gst_turnover_volatility": 0.08,
        "gst_filing_punctuality": 12,
        "gst_turnover_decline_3m": -0.05,
        "bank_avg_monthly_credits": 560000.0,
        "bank_balance_dip_count": 0,
        "bank_bounce_count": 0,
        "bank_cash_buffer_days": 55.0,
        "bank_credit_debit_ratio": 1.15,
        "xf_gst_bank_gap": 0.05,
        "epfo_headcount": 18,
        "epfo_headcount_trend": 0.06,
    }


def _stressed_firm() -> dict[str, float]:
    return {
        "gst_avg_monthly_turnover": 300000.0,
        "gst_turnover_growth": -0.14,
        "gst_turnover_volatility": 0.45,
        "gst_filing_punctuality": 6,
        "gst_turnover_decline_3m": 0.22,
        "bank_avg_monthly_credits": 190000.0,
        "bank_balance_dip_count": 6,
        "bank_bounce_count": 3,
        "bank_cash_buffer_days": 9.0,
        "bank_credit_debit_ratio": 0.82,
        "xf_gst_bank_gap": 0.40,
        "epfo_headcount": 7,
        "epfo_headcount_trend": -0.10,
    }


def test_healthy_firm_stays_flat_and_unflagged():
    traj = simulate(_trained_model(), _healthy_firm())
    assert len(traj.scores) == HORIZON
    assert traj.flag_month is None
    assert not traj.deteriorating


def test_stressed_firm_slides_and_flags():
    traj = simulate(_trained_model(), _stressed_firm())
    assert traj.deteriorating
    assert traj.scores[0] > traj.scores[-1]
    assert traj.flag_month is not None
    assert traj.scores[traj.flag_month - 1] < WATCH_THRESHOLD
