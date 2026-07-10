from __future__ import annotations

import numpy as np

from parakh.eval.metrics import evaluate, ks_statistic


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
