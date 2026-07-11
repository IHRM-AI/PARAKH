from __future__ import annotations

import json
import logging

import numpy as np

from parakh.config import settings
from parakh.eval.fairness import fairness_slices
from parakh.eval.stability import score_stability_by_cohort
from parakh.pipelines import reject_inference
from parakh.scoring.model import HealthModel, pd_to_score
from parakh.synth.persona import ALL_FEATURES, TARGET, generate

logger = logging.getLogger("parakh.rigor")


def _split(n: int, valid_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    hold = int(n * valid_fraction)
    return order[2 * hold :], order[hold : 2 * hold], order[:hold]


def run_stability_and_fairness(n: int = 8000) -> dict[str, dict[str, object]]:
    """Regenerate the population-stability and fairness-slicing artifacts.

    Both read off one held-out test fold scored by the deployed pipeline, so the
    numbers are the same the model card cites. Stability scores every cohort in
    the population against a baseline cohort; fairness slices the test fold.
    """
    df = generate(n=n, seed=settings.random_seed)
    y = df[TARGET].to_numpy()
    train_idx, calib_idx, test_idx = _split(len(df), settings.validation_fraction, settings.random_seed)
    x = df[ALL_FEATURES]

    model = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[calib_idx], y[calib_idx])

    all_pd = model.predict_pd(x)
    all_score = pd_to_score(all_pd)
    stability = score_stability_by_cohort(all_score, df["cohort"].to_numpy())
    logger.info(
        "stability: baseline cohort %d, max PSI %.4f, share within %.2f = %.2f",
        stability["baseline_cohort"], stability["max_psi"], stability["threshold"],
        stability["share_within_threshold"],
    )

    test_frame = df.iloc[test_idx].reset_index(drop=True)
    fairness = fairness_slices(test_frame, y[test_idx], all_pd[test_idx])
    for dim in ("sector", "state", "headcount_band"):
        d = fairness[dim]
        logger.info(
            "fairness %s: DI ratio %s, weakest AUC slice %s = %s",
            dim, d["disparate_impact_ratio"], d["weakest_auc_slice"], d["weakest_auc"],
        )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (settings.artifacts_dir / "stability.json").write_text(json.dumps(stability, indent=2))
    (settings.artifacts_dir / "fairness.json").write_text(json.dumps(fairness, indent=2))
    return {"stability": stability, "fairness": fairness}


def run() -> dict[str, object]:
    """Regenerate every ML-rigor artifact: reject inference, stability, fairness."""
    ri = reject_inference.run()
    sf = run_stability_and_fairness()
    return {"reject_inference": ri, **sf}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
