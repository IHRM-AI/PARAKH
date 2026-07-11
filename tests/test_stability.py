from __future__ import annotations

import numpy as np

from parakh.eval.stability import population_stability_index, score_stability_by_cohort


def test_identical_distributions_give_near_zero_psi():
    rng = np.random.default_rng(0)
    base = rng.normal(50, 10, 5000)
    comp = rng.normal(50, 10, 5000)
    assert population_stability_index(base, comp) < 0.02


def test_shifted_distribution_raises_psi():
    rng = np.random.default_rng(0)
    base = rng.normal(50, 10, 5000)
    shifted = rng.normal(65, 10, 5000)
    same = rng.normal(50, 10, 5000)
    assert population_stability_index(base, shifted) > population_stability_index(base, same)
    assert population_stability_index(base, shifted) > 0.1


def test_psi_is_non_negative_and_finite():
    rng = np.random.default_rng(1)
    base = rng.normal(50, 8, 2000)
    comp = rng.normal(55, 12, 2000)
    psi = population_stability_index(base, comp)
    assert psi >= 0.0
    assert np.isfinite(psi)


def test_empty_bucket_stays_finite():
    base = np.concatenate([np.zeros(500), np.ones(500) * 100])
    comp = np.zeros(500)
    psi = population_stability_index(base, comp, buckets=10)
    assert np.isfinite(psi)
    assert psi > 0.0


def test_cohort_report_shape_and_baseline():
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(60, 10, 1000), rng.normal(45, 10, 1000)])
    cohorts = np.concatenate([np.zeros(1000, dtype=int), np.ones(1000, dtype=int)])
    report = score_stability_by_cohort(scores, cohorts)
    assert report["baseline_cohort"] == 0
    assert "1" in report["per_cohort_psi"]
    assert "0" not in report["per_cohort_psi"]
    assert report["max_psi"] == report["per_cohort_psi"]["1"]
    assert 0.0 <= report["share_within_threshold"] <= 1.0


def test_cohort_over_threshold_is_flagged():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(60, 6, 2000), rng.normal(30, 6, 2000)])
    cohorts = np.concatenate([np.zeros(2000, dtype=int), np.ones(2000, dtype=int)])
    report = score_stability_by_cohort(scores, cohorts, threshold=0.10)
    assert "1" in report["cohorts_over_threshold"]
    assert report["max_psi"] >= 0.10


def test_explicit_baseline_cohort():
    rng = np.random.default_rng(4)
    scores = np.concatenate([rng.normal(60, 8, 800)] * 3)
    cohorts = np.repeat([0, 1, 2], 800)
    report = score_stability_by_cohort(scores, cohorts, baseline_cohort=1)
    assert report["baseline_cohort"] == 1
    assert "1" not in report["per_cohort_psi"]
    assert set(report["per_cohort_psi"]) == {"0", "2"}
