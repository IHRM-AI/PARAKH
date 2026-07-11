from __future__ import annotations

from parakh.pipelines import reject_inference
from parakh.synth.persona import BUREAU_FEATURE, TARGET, generate_through_the_door


def test_through_the_door_has_bureau_marker():
    df = generate_through_the_door(n=1500, seed=3, thin_file_share=0.4)
    assert BUREAU_FEATURE in df.columns
    assert set(df[BUREAU_FEATURE].unique()) <= {0, 1}
    thin_share = 1 - df[BUREAU_FEATURE].mean()
    assert 0.3 < thin_share < 0.5
    assert TARGET in df.columns


def test_reject_inference_expands_approvals_and_reports_incremental_cohort(tmp_path, monkeypatch):
    from parakh.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    result = reject_inference.run(n=3000, seed=7)

    assert result["method"] == "fuzzy augmentation (parcelling)"
    assert 0.0 <= result["approval_rate_bureau"] <= 1.0
    assert 0.0 <= result["approval_rate_parakh"] <= 1.0
    assert result["approval_rate_parakh"] > result["approval_rate_bureau"]
    assert result["incremental_approval_pp"] > 0
    assert result["incremental_cohort_n"] > 0
    assert 0.0 <= result["default_rate_incremental_cohort"] <= 1.0
    assert (tmp_path / "reject_inference.json").exists()


def test_reject_inference_marks_synthetic(tmp_path, monkeypatch):
    from parakh.config import settings

    monkeypatch.setattr(settings, "artifacts_dir", tmp_path)
    result = reject_inference.run(n=2500, seed=1)
    assert "SYNTHETIC" in result["notes"]
    assert result["default_rate_incremental_cohort"] < result["approval_rate_bureau"] + 1.0
