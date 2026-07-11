from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from parakh.config import settings
from parakh.scoring.card import PD_CUTOFF
from parakh.scoring.model import PARAMS, HealthModel, monotone_vector
from parakh.synth.persona import (
    ALL_FEATURES,
    BUREAU_FEATURE,
    TARGET,
    generate_through_the_door,
)

logger = logging.getLogger("parakh.reject_inference")

METHOD = "fuzzy augmentation (parcelling)"


def _fit_weighted(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[lgb.Booster, IsotonicRegression]:
    """Weighted-sample twin of :meth:`HealthModel.fit` for the parcelled AGB model.

    Kept local to the pipeline so the deployed scorer in ``scoring/model.py`` is
    untouched. Same monotone constraints, same isotonic calibration on a clean
    booked-only validation fold.
    """
    params = {**PARAMS, "monotone_constraints": monotone_vector(list(x_train.columns))}
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=800,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    raw = booster.predict(x_valid, num_iteration=booster.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y_valid)
    return booster, calibrator


def _predict_pd(booster: lgb.Booster, calibrator: IsotonicRegression, x: pd.DataFrame) -> np.ndarray:
    raw = booster.predict(x, num_iteration=booster.best_iteration)
    return calibrator.predict(raw)


def _split(n: int, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    hold = int(n * valid_fraction)
    return order[2 * hold :], order[hold : 2 * hold], order[:hold]


def run(
    n: int = 12000,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    """Demonstrate the cash-flow-scoring inclusion thesis on synthetic data.

    A bureau-or-collateral lender only ever books the with-bureau subset; the
    thin-file (New-to-Credit) applicants are rejected unscored. We:

    1. train a known-good-bad (KGB) model on booked-only firms,
    2. apply fuzzy augmentation (parcelling): each rejected firm enters the
       training set twice, weighted by its KGB-inferred PD as a bad and
       ``1 - PD`` as a good,
    3. retrain an all-good-bad (AGB) model on booked + reweighted rejects,
    4. compare bureau approval rate versus PARAKH approval rate on a held-out
       through-the-door fold, and measure the realised default rate of the
       PARAKH-approved-but-bureau-rejected cohort.

    Both lenders apply the same calibrated-PD sanction cutoff (``PD_CUTOFF``).
    The bureau lender can only approve firms that both carry a bureau file and
    clear the cutoff on the KGB (booked-only) model; PARAKH scores every firm on
    alternate data, so it can additionally reach thin-file applicants. All figures
    are on synthetic data and demonstrate the mechanism only; they are not a
    real-world performance claim.
    """
    df = generate_through_the_door(n=n, seed=seed)
    y = df[TARGET].to_numpy()
    booked_mask = df[BUREAU_FEATURE].to_numpy() == 1

    train_idx, calib_idx, test_idx = _split(len(df), valid_fraction, seed)
    x = df[ALL_FEATURES]

    train_booked = train_idx[booked_mask[train_idx]]
    calib_booked = calib_idx[booked_mask[calib_idx]]
    train_rejects = train_idx[~booked_mask[train_idx]]

    kgb = HealthModel().fit(
        x.iloc[train_booked], y[train_booked], x.iloc[calib_booked], y[calib_booked]
    )

    reject_pd = kgb.predict_pd(x.iloc[train_rejects])
    x_aug = pd.concat(
        [x.iloc[train_booked], x.iloc[train_rejects], x.iloc[train_rejects]], ignore_index=True
    )
    y_aug = np.concatenate(
        [y[train_booked], np.ones(len(train_rejects), dtype=int), np.zeros(len(train_rejects), dtype=int)]
    )
    w_aug = np.concatenate(
        [np.ones(len(train_booked)), reject_pd, 1.0 - reject_pd]
    )
    agb_booster, agb_calibrator = _fit_weighted(
        x_aug, y_aug, w_aug, x.iloc[calib_booked], y[calib_booked]
    )

    x_test = x.iloc[test_idx]
    has_bureau_test = booked_mask[test_idx]

    kgb_pd_test = kgb.predict_pd(x_test)
    bureau_approved = has_bureau_test & (kgb_pd_test < PD_CUTOFF)

    parakh_pd = _predict_pd(agb_booster, agb_calibrator, x_test)
    parakh_approved = parakh_pd < PD_CUTOFF

    approval_rate_bureau = float(bureau_approved.mean())
    approval_rate_parakh = float(parakh_approved.mean())

    incremental_mask = parakh_approved & (~bureau_approved)
    y_test = y[test_idx]
    incremental_n = int(incremental_mask.sum())
    default_rate_incremental = (
        float(y_test[incremental_mask].mean()) if incremental_n else float("nan")
    )
    default_rate_bureau_booked = float(y_test[bureau_approved].mean())

    result: dict[str, object] = {
        "method": METHOD,
        "n_test": int(len(test_idx)),
        "n_booked_train": int(len(train_booked)),
        "n_rejects_train": int(len(train_rejects)),
        "approval_rate_bureau": round(approval_rate_bureau, 4),
        "approval_rate_parakh": round(approval_rate_parakh, 4),
        "incremental_approval_pp": round((approval_rate_parakh - approval_rate_bureau) * 100, 2),
        "incremental_cohort_n": incremental_n,
        "default_rate_incremental_cohort": (
            round(default_rate_incremental, 4) if incremental_n else None
        ),
        "default_rate_bureau_booked": round(default_rate_bureau_booked, 4),
        "pd_cutoff": PD_CUTOFF,
        "notes": (
            "SYNTHETIC DATA demonstrating the cash-flow-scoring inclusion mechanism. "
            "Bureau approval = firm has a bureau file AND clears the sanction cutoff on "
            "the booked-only (KGB) model; PARAKH approval = calibrated PD below the same "
            "cutoff, scored on alternate data for every firm. The incremental cohort is "
            "PARAKH-approved but bureau-rejected. Not a real-world performance claim."
        ),
    }

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "reject_inference.json").write_text(json.dumps(result, indent=2))
    logger.info(
        "reject inference (%s): bureau approval %.1f%% vs PARAKH %.1f%% (+%.1fpp), "
        "incremental cohort n=%d default rate %s",
        METHOD, approval_rate_bureau * 100, approval_rate_parakh * 100,
        result["incremental_approval_pp"], incremental_n,
        result["default_rate_incremental_cohort"],
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
