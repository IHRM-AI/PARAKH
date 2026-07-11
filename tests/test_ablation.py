from __future__ import annotations

from parakh.eval.ablation import model_comparison, source_ablation, source_ablation_ci
from parakh.synth.persona import generate


def test_source_ablation_ladder_adds_signal():
    df = generate(n=1500, seed=7)
    ladder = source_ablation(df, valid_fraction=0.2, seed=7)
    assert set(ladder) == {"GST only", "+ AA bank statements", "+ EPFO"}
    assert all(0.5 <= auc <= 1.0 for auc in ladder.values())


def test_source_ablation_ci_brackets_point_auc():
    df = generate(n=1500, seed=7)
    ladder = source_ablation_ci(df, valid_fraction=0.2, seed=7, resamples=100)
    for stage in ("GST only", "+ AA bank statements", "+ EPFO"):
        row = ladder[stage]
        assert row["ci_low"] <= row["auc"] <= row["ci_high"]


def test_model_comparison_reports_lift():
    df = generate(n=1500, seed=7)
    comparison = model_comparison(df, valid_fraction=0.2, seed=7)
    assert set(comparison) == {
        "logistic_auc",
        "gbm_auc",
        "gbm_lift",
        "delta_ci_low",
        "delta_ci_high",
        "p_value",
        "delta_ci_includes_zero",
    }
    assert abs(comparison["gbm_lift"] - (comparison["gbm_auc"] - comparison["logistic_auc"])) < 1e-6
    assert comparison["delta_ci_low"] <= comparison["gbm_lift"] <= comparison["delta_ci_high"]
    assert 0.0 <= comparison["p_value"] <= 1.0
    assert comparison["delta_ci_includes_zero"] == (
        comparison["delta_ci_low"] <= 0.0 <= comparison["delta_ci_high"]
    )
