from __future__ import annotations

import numpy as np

from parakh.eval.metrics import bootstrap_auc_ci, evaluate, ks_statistic, reliability_curve


def test_auc_matches_hand_computed_value():
    # Four pairs of (score, label). Of the 4 positive-negative pairs, the
    # positive outranks the negative in 3 and ties in 0, so AUC = 3/4.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    report = evaluate(y_true, y_score)
    assert abs(report.auc - 0.75) < 1e-9
    assert abs(report.gini - 0.5) < 1e-9


def test_perfect_separation_gives_auc_one():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    report = evaluate(y_true, y_score)
    assert abs(report.auc - 1.0) < 1e-9
    assert abs(report.ks - 1.0) < 1e-9


def test_ks_matches_hand_computed_value():
    # Perfectly separated: all negatives below all positives, so the maximum
    # gap between the cumulative positive and negative distributions is 1.0.
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(ks_statistic(y_true, y_score) - 1.0) < 1e-9


def test_brier_matches_hand_computed_value():
    # Mean squared error of probabilities against labels.
    # ((0.2-0)^2 + (0.3-0)^2 + (0.9-1)^2 + (0.4-1)^2) / 4 = 0.125
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.3, 0.9, 0.4])
    report = evaluate(y_true, y_score)
    assert abs(report.brier - 0.125) < 1e-9


def test_report_records_size_and_default_rate():
    y_true = np.array([0, 1, 1, 1])
    y_score = np.array([0.2, 0.6, 0.7, 0.9])
    report = evaluate(y_true, y_score)
    assert report.n == 4
    assert abs(report.default_rate - 0.75) < 1e-9
    assert set(report.as_dict()) == {"n", "default_rate", "auc", "gini", "ks", "brier"}


def test_bootstrap_ci_brackets_point_auc():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 400)
    y_score = np.clip(0.5 * y_true + rng.normal(0, 0.3, 400), 0, 1)
    point = evaluate(y_true, y_score).auc
    low, high = bootstrap_auc_ci(y_true, y_score, resamples=200, seed=0)
    assert low <= point <= high
    assert 0.0 <= low <= high <= 1.0


def test_reliability_curve_bins_track_observed_rate():
    y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_score = np.array([0.05, 0.08, 0.12, 0.15, 0.85, 0.88, 0.92, 0.95])
    curve = reliability_curve(y_true, y_score, bins=10)
    assert curve
    assert all(0.0 <= point["observed_default_rate"] <= 1.0 for point in curve)
    assert sum(point["count"] for point in curve) == len(y_true)
    low_bin = min(curve, key=lambda p: p["mean_predicted"])
    high_bin = max(curve, key=lambda p: p["mean_predicted"])
    assert high_bin["observed_default_rate"] >= low_bin["observed_default_rate"]
