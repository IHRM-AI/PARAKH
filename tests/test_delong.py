from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from parakh.eval.delong import _midrank, delong_auc_delta


def _correlated_scores(seed: int = 0, n: int = 1500):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    strong = np.clip(0.6 * y + rng.normal(0, 0.4, n), 0, 1)
    weak = np.clip(0.3 * y + rng.normal(0, 0.5, n), 0, 1)
    return y, strong, weak


def test_component_auc_matches_sklearn():
    y, strong, weak = _correlated_scores()
    result = delong_auc_delta(y, strong, weak)
    assert abs(result.auc_a - roc_auc_score(y, strong)) < 1e-4
    assert abs(result.auc_b - roc_auc_score(y, weak)) < 1e-4
    assert abs(result.delta_auc - (result.auc_a - result.auc_b)) < 1e-4


def test_strong_lift_is_significant_and_excludes_zero():
    y, strong, weak = _correlated_scores()
    result = delong_auc_delta(y, strong, weak)
    assert result.delta_auc > 0
    assert result.p_value < 0.01
    assert not result.includes_zero
    assert result.delta_ci_low <= result.delta_auc <= result.delta_ci_high


def test_identical_predictors_give_zero_delta_and_p_one():
    y, strong, _ = _correlated_scores()
    result = delong_auc_delta(y, strong, strong.copy())
    assert result.delta_auc == 0.0
    assert result.p_value == 1.0
    assert result.includes_zero


def test_delta_is_antisymmetric_and_p_matches():
    y, strong, weak = _correlated_scores()
    forward = delong_auc_delta(y, strong, weak)
    reverse = delong_auc_delta(y, weak, strong)
    assert abs(forward.delta_auc + reverse.delta_auc) < 1e-9
    assert abs(forward.p_value - reverse.p_value) < 1e-9


def test_midrank_averages_ties():
    ranks = _midrank(np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0]))
    assert list(ranks) == [1.5, 1.5, 3.0, 5.0, 5.0, 5.0]


def test_shape_and_binary_validation():
    y, strong, weak = _correlated_scores(n=200)
    with pytest.raises(ValueError):
        delong_auc_delta(y, strong[:100], weak)
    with pytest.raises(ValueError):
        delong_auc_delta(np.full(200, 1), strong, weak)
    with pytest.raises(ValueError):
        delong_auc_delta(np.full(200, 2), strong, weak)


def test_result_serialises():
    y, strong, weak = _correlated_scores(n=300)
    payload = delong_auc_delta(y, strong, weak).as_dict()
    assert set(payload) == {
        "auc_a",
        "auc_b",
        "delta_auc",
        "delta_ci_low",
        "delta_ci_high",
        "p_value",
        "includes_zero",
    }
