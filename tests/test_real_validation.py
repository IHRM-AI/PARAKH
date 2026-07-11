from __future__ import annotations

import numpy as np
import pandas as pd

from parakh.pipelines.real_validation import (
    _load_sba,
    _load_sba_foia,
    _sba_loader_for,
    load_real_dataset,
)


def _write_foia(path, rows: int = 400) -> None:
    rng = np.random.default_rng(0)
    status = rng.choice(["PIF", "CHGOFF"], size=rows, p=[0.85, 0.15])
    frame = pd.DataFrame(
        {
            "LoanStatus": status,
            "TermInMonths": rng.integers(12, 240, rows),
            "JobsSupported": rng.integers(0, 50, rows),
            "GrossApproval": rng.integers(20000, 500000, rows),
            "SBAGuaranteedApproval": rng.integers(10000, 400000, rows),
            "InitialInterestRate": rng.uniform(3, 12, rows).round(2),
            "BusinessType": rng.choice(["INDIVIDUAL", "CORPORATION", "PARTNERSHIP"], rows),
            "NAICSCode": rng.integers(110000, 990000, rows),
            "ApprovalFiscalYear": rng.integers(2005, 2015, rows),
        }
    )
    frame.to_csv(path, index=False)


def _write_kaggle(path, rows: int = 400) -> None:
    rng = np.random.default_rng(1)
    status = rng.choice(["P I F", "CHGOFF"], size=rows, p=[0.85, 0.15])
    frame = pd.DataFrame(
        {
            "MIS_Status": status,
            "Term": rng.integers(12, 240, rows),
            "NoEmp": rng.integers(0, 50, rows),
            "NewExist": rng.choice([1, 2], rows),
            "CreateJob": rng.integers(0, 20, rows),
            "RetainedJob": rng.integers(0, 20, rows),
            "UrbanRural": rng.choice([0, 1, 2], rows),
            "RevLineCr": rng.choice(["Y", "N"], rows),
            "LowDoc": rng.choice(["Y", "N"], rows),
            "DisbursementGross": [f"${v:,}" for v in rng.integers(20000, 500000, rows)],
            "GrAppv": [f"${v:,}" for v in rng.integers(20000, 500000, rows)],
            "SBA_Appv": [f"${v:,}" for v in rng.integers(10000, 400000, rows)],
            "NAICS": rng.integers(110000, 990000, rows),
            "ApprovalFY": rng.integers(2005, 2015, rows),
        }
    )
    frame.to_csv(path, index=False)


def test_dispatcher_detects_foia_schema(tmp_path):
    path = tmp_path / "foia.csv"
    _write_foia(path)
    assert _sba_loader_for(path) is _load_sba_foia


def test_dispatcher_detects_kaggle_schema(tmp_path):
    path = tmp_path / "SBAnational.csv"
    _write_kaggle(path)
    assert _sba_loader_for(path) is _load_sba


def test_dispatcher_ignores_unknown_schema(tmp_path):
    path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
    assert _sba_loader_for(path) is None


def test_foia_loader_labels_and_features(tmp_path):
    path = tmp_path / "foia.csv"
    _write_foia(path)
    dataset = _load_sba_foia(path, sample_rows=1000, seed=0)
    assert dataset.name == "SBA 7(a) FOIA"
    assert set(np.unique(dataset.target)) <= {0, 1}
    assert "sba_guarantee_share" in dataset.features.columns
    assert "term_months" in dataset.features.columns
    assert len(dataset.features) == len(dataset.target)


def test_load_real_dataset_prefers_sba_and_reads_foia(tmp_path, monkeypatch):
    import parakh.pipelines.real_validation as rv

    sba_dir = tmp_path / "sba"
    sba_dir.mkdir(parents=True)
    _write_foia(sba_dir / "7a_foia.csv")
    monkeypatch.setattr(rv, "DATA_ROOT", tmp_path)

    dataset = load_real_dataset(sample_rows=1000, seed=0)
    assert dataset is not None
    assert dataset.name == "SBA 7(a) FOIA"


def test_load_real_dataset_none_when_absent(tmp_path, monkeypatch):
    import parakh.pipelines.real_validation as rv

    monkeypatch.setattr(rv, "DATA_ROOT", tmp_path)
    assert load_real_dataset(sample_rows=100, seed=0) is None
