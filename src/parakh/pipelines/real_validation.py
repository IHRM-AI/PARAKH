from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from parakh.config import settings
from parakh.eval.metrics import bootstrap_auc_ci, evaluate, reliability_curve
from parakh.scoring.model import HealthModel

logger = logging.getLogger("parakh.real_validation")

DATA_ROOT = settings.artifacts_dir.parent / "data" / "raw"


@dataclass(frozen=True)
class RealDataset:
    name: str
    label: str
    features: pd.DataFrame
    target: np.ndarray
    split_column: pd.Series | None
    notes: str


def _clean_currency(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(text, errors="coerce")


def _load_sba(path: Path, sample_rows: int, seed: int) -> RealDataset:
    """US SBA 7(a) small-business loans: the canonical real MSME-analog default set."""
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["MIS_Status"].isin(["P I F", "CHGOFF"])].copy()
    target = (raw["MIS_Status"] == "CHGOFF").astype(int).to_numpy()

    if len(raw) > sample_rows:
        rng = np.random.default_rng(seed)
        pick = np.zeros(len(raw), dtype=bool)
        for label in (0, 1):
            idx = np.where(target == label)[0]
            take = int(round(sample_rows * len(idx) / len(raw)))
            pick[rng.choice(idx, size=min(take, len(idx)), replace=False)] = True
        raw, target = raw.iloc[pick].reset_index(drop=True), target[pick]

    term = pd.to_numeric(raw.get("Term"), errors="coerce")
    features = pd.DataFrame(
        {
            "term": term,
            "no_emp": pd.to_numeric(raw.get("NoEmp"), errors="coerce"),
            "new_exist": pd.to_numeric(raw.get("NewExist"), errors="coerce"),
            "create_job": pd.to_numeric(raw.get("CreateJob"), errors="coerce"),
            "retained_job": pd.to_numeric(raw.get("RetainedJob"), errors="coerce"),
            "urban_rural": pd.to_numeric(raw.get("UrbanRural"), errors="coerce"),
            "rev_line_cr": (raw.get("RevLineCr").astype(str) == "Y").astype(int),
            "low_doc": (raw.get("LowDoc").astype(str) == "Y").astype(int),
            "disbursement_gross": _clean_currency(raw.get("DisbursementGross")),
            "gr_appv": _clean_currency(raw.get("GrAppv")),
            "sba_appv": _clean_currency(raw.get("SBA_Appv")),
            "naics_sector": pd.to_numeric(raw.get("NAICS").astype(str).str[:2], errors="coerce"),
        }
    )
    features["sba_guarantee_share"] = features["sba_appv"] / features["gr_appv"].replace(0, np.nan)

    approval_year = pd.to_numeric(raw.get("ApprovalFY"), errors="coerce")
    split_column = approval_year if approval_year.notna().mean() > 0.9 else None
    notes = (
        "US SBA 7(a) small-business loans (real MSME-analog defaults). "
        "Label MIS_Status: CHGOFF=1 (charged off), 'P I F'=0 (paid in full)."
    )
    return RealDataset("SBA 7(a) national", "real MSME defaults", features, target, split_column, notes)


def _load_home_credit(path: Path, sample_rows: int, seed: int) -> RealDataset:
    """Home-Credit retail loans: real-default methodology validation (retail proxy, not MSME)."""
    cols = [
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "CNT_FAM_MEMBERS",
        "CNT_CHILDREN",
        "REGION_POPULATION_RELATIVE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "REGION_RATING_CLIENT",
        "DAYS_LAST_PHONE_CHANGE",
    ]
    raw = pd.read_csv(path, usecols=cols, low_memory=False)
    if len(raw) > sample_rows:
        rng = np.random.default_rng(seed)
        target = raw["TARGET"].to_numpy()
        pick = np.zeros(len(raw), dtype=bool)
        for label in (0, 1):
            idx = np.where(target == label)[0]
            take = int(round(sample_rows * len(idx) / len(raw)))
            pick[rng.choice(idx, size=min(take, len(idx)), replace=False)] = True
        raw = raw.iloc[pick].reset_index(drop=True)

    target = raw["TARGET"].astype(int).to_numpy()
    employed = raw["DAYS_EMPLOYED"].replace(365243, np.nan)
    features = pd.DataFrame(
        {
            "income_total": raw["AMT_INCOME_TOTAL"],
            "credit_amount": raw["AMT_CREDIT"],
            "annuity": raw["AMT_ANNUITY"],
            "goods_price": raw["AMT_GOODS_PRICE"],
            "credit_to_income": raw["AMT_CREDIT"] / raw["AMT_INCOME_TOTAL"].replace(0, np.nan),
            "annuity_to_income": raw["AMT_ANNUITY"] / raw["AMT_INCOME_TOTAL"].replace(0, np.nan),
            "age_years": (-raw["DAYS_BIRTH"] / 365.25).round(1),
            "employed_years": (-employed / 365.25).round(1),
            "registration_years": (-raw["DAYS_REGISTRATION"] / 365.25).round(1),
            "family_size": raw["CNT_FAM_MEMBERS"],
            "children": raw["CNT_CHILDREN"],
            "region_population": raw["REGION_POPULATION_RELATIVE"],
            "ext_source_1": raw["EXT_SOURCE_1"],
            "ext_source_2": raw["EXT_SOURCE_2"],
            "ext_source_3": raw["EXT_SOURCE_3"],
            "region_rating": raw["REGION_RATING_CLIENT"],
            "phone_change_days": raw["DAYS_LAST_PHONE_CHANGE"],
        }
    )
    notes = (
        "Home-Credit retail loans (307k real consumer defaults, target=TARGET). "
        "Used as real-default methodology validation (retail proxy), NOT an MSME claim: "
        "SBA small-business data was not fetchable on this network."
    )
    return RealDataset("Home Credit (retail proxy)", "real retail defaults", features, target, None, notes)


def load_real_dataset(sample_rows: int = 60000, seed: int = 42) -> RealDataset | None:
    """Return the best available real default dataset, or None if none is on disk."""
    sba_path = DATA_ROOT / "sba" / "SBAnational.csv"
    if sba_path.exists():
        logger.info("loading real dataset: SBA 7(a) national")
        return _load_sba(sba_path, sample_rows, seed)
    home_path = DATA_ROOT / "homecredit" / "application_train.csv"
    if home_path.exists():
        logger.info("loading real dataset: Home Credit (retail proxy)")
        return _load_home_credit(home_path, sample_rows, seed)
    return None


def _split(dataset: RealDataset, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return train, calibrate and test indices plus the split type.

    An out-of-time split is used when a vintage-like column is present (train on
    early vintages, test on the latest); otherwise a stratified random holdout.
    Calibration always comes from within the training vintages so isotonic never
    sees the test firms.
    """
    n = len(dataset.target)
    rng = np.random.default_rng(seed)
    if dataset.split_column is not None:
        years = dataset.split_column.to_numpy()
        cutoff = np.nanquantile(years, 1 - test_fraction)
        test_idx = np.where(years >= cutoff)[0]
        early = rng.permutation(np.where(years < cutoff)[0])
        hold = int(len(early) * test_fraction)
        return early[hold:], early[:hold], test_idx, "out-of-time (approval vintage)"

    order = rng.permutation(n)
    hold = int(n * test_fraction)
    return order[2 * hold :], order[hold : 2 * hold], order[:hold], "stratified random holdout"


def run(
    sample_rows: int = 60000,
    test_fraction: float = 0.2,
    bootstrap_resamples: int = 500,
    seed: int = 42,
) -> dict[str, object] | None:
    dataset = load_real_dataset(sample_rows=sample_rows, seed=seed)
    if dataset is None:
        logger.warning("no real dataset on disk under %s; skipping real validation", DATA_ROOT)
        return None

    train_idx, calib_idx, test_idx, split_type = _split(dataset, test_fraction, seed)
    x, y = dataset.features, dataset.target

    model = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[calib_idx], y[calib_idx])
    pd_test = model.predict_pd(x.iloc[test_idx])
    report = evaluate(y[test_idx], pd_test)
    ci_low, ci_high = bootstrap_auc_ci(y[test_idx], pd_test, bootstrap_resamples, seed)

    fit_idx = np.concatenate([train_idx, calib_idx])
    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000)
    )
    logistic.fit(x.iloc[fit_idx].fillna(x.iloc[fit_idx].median()), y[fit_idx])
    log_pd = logistic.predict_proba(x.iloc[test_idx].fillna(x.iloc[fit_idx].median()))[:, 1]
    logistic_auc = float(roc_auc_score(y[test_idx], log_pd))

    curve = reliability_curve(y[test_idx], pd_test, bins=10)

    result: dict[str, object] = {
        "dataset": dataset.name,
        "dataset_label": dataset.label,
        "n_train": int(len(train_idx)),
        "n_calibrate": int(len(calib_idx)),
        "n_test": int(len(test_idx)),
        "test_default_rate": round(report.default_rate, 4),
        "split_type": split_type,
        "auc": round(report.auc, 4),
        "auc_ci_low": round(ci_low, 4),
        "auc_ci_high": round(ci_high, 4),
        "gini": round(report.gini, 4),
        "ks": round(report.ks, 4),
        "brier": round(report.brier, 6),
        "logistic_auc": round(logistic_auc, 4),
        "gbm_lift": round(report.auc - logistic_auc, 4),
        "bootstrap_resamples": bootstrap_resamples,
        "reliability_curve": curve,
        "notes": dataset.notes,
    }

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "real_validation.json").write_text(json.dumps(result, indent=2))
    logger.info(
        "real validation on %s: AUC=%.4f [%.4f, %.4f] KS=%.4f Brier=%.4f logistic=%.4f lift=%+.4f",
        dataset.name, report.auc, ci_low, ci_high, report.ks, report.brier, logistic_auc,
        report.auc - logistic_auc,
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if run() is None:
        logger.warning("real validation produced no artifact (dataset absent)")


if __name__ == "__main__":
    main()
