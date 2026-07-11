from __future__ import annotations

import numpy as np

from parakh.eval.fairness import HEADCOUNT_BANDS, _headcount_band, fairness_slices
from parakh.synth.persona import ALL_FEATURES, TARGET, generate
from parakh.scoring.model import HealthModel


def _scored_fold(n: int = 2500, seed: int = 5):
    df = generate(n=n, seed=seed)
    y = df[TARGET].to_numpy()
    x = df[ALL_FEATURES]
    model = HealthModel(num_boost_round=120, early_stopping_rounds=25)
    model.fit(x.iloc[: n - 800], y[: n - 800], x.iloc[n - 800 :], y[n - 800 :])
    test = df.iloc[n - 800 :].reset_index(drop=True)
    pd_test = model.predict_pd(test[ALL_FEATURES])
    return test, y[n - 800 :], pd_test


def test_headcount_bands_cover_the_range():
    counts = np.array([1, 5, 9, 10, 24, 25, 200])
    bands = _headcount_band(__import__("pandas").Series(counts))
    assert bands.notna().all()
    assert bands.iloc[0] == "micro (1-9)"
    assert bands.iloc[3] == "small (10-24)"
    assert bands.iloc[6] == "mid (25+)"
    assert len(HEADCOUNT_BANDS) == 3


def test_fairness_dimensions_present():
    test, y, pd_test = _scored_fold()
    result = fairness_slices(test, y, pd_test)
    assert set(result) == {"approval_cutoff", "sector", "state", "headcount_band"}
    for dim in ("sector", "state", "headcount_band"):
        d = result[dim]
        assert d["slices"]
        assert d["disparate_impact_ratio"] is None or 0.0 <= d["disparate_impact_ratio"] <= 1.0


def test_slice_metrics_are_bounded():
    test, y, pd_test = _scored_fold()
    result = fairness_slices(test, y, pd_test)
    for value in result["sector"]["slices"].values():
        assert value["n"] > 0
        assert 0.0 <= value["approval_rate"] <= 1.0
        assert 0.0 <= value["observed_default_rate"] <= 1.0
        assert value["auc"] is None or 0.0 <= value["auc"] <= 1.0


def test_disparate_impact_is_min_over_max():
    test, y, pd_test = _scored_fold()
    result = fairness_slices(test, y, pd_test)
    d = result["state"]
    rates = [s["approval_rate"] for s in d["slices"].values()]
    assert d["min_approval_rate"] == round(min(rates), 4)
    assert d["max_approval_rate"] == round(max(rates), 4)
    assert abs(d["disparate_impact_ratio"] - round(min(rates) / max(rates), 4)) < 1e-9


def test_single_class_slice_reports_none_auc():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "sector": ["A", "A", "B", "B"],
            "state": ["X", "X", "Y", "Y"],
            "epfo_headcount": [5, 5, 30, 30],
        }
    )
    y = np.array([0, 0, 1, 1])
    pd_score = np.array([0.1, 0.2, 0.3, 0.4])
    result = fairness_slices(frame, y, pd_score)
    assert result["sector"]["slices"]["A"]["auc"] is None
    assert result["sector"]["slices"]["B"]["auc"] is None
