from __future__ import annotations

import json
import logging

import joblib
import numpy as np

from parakh.config import settings
from parakh.eval.ablation import source_ablation
from parakh.eval.metrics import evaluate
from parakh.scoring.model import HealthModel
from parakh.synth.persona import (
    BANK_FEATURES,
    EPFO_FEATURES,
    GST_FEATURES,
    TARGET,
    generate,
)

logger = logging.getLogger("parakh.train")

ALL_FEATURES = GST_FEATURES + BANK_FEATURES + EPFO_FEATURES


def run(n: int = 4000) -> dict[str, object]:
    df = generate(n=n, seed=settings.random_seed)
    logger.info("generated %d firms, default rate %.3f", len(df), df[TARGET].mean())

    ladder = source_ablation(df, valid_fraction=settings.validation_fraction, seed=settings.random_seed)
    logger.info("source-ablation AUC ladder: %s", ladder)

    rng = np.random.default_rng(settings.random_seed)
    order = rng.permutation(len(df))
    cutoff = int(len(df) * (1 - settings.validation_fraction))
    train_idx, valid_idx = order[:cutoff], order[cutoff:]
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()

    model = HealthModel().fit(x.iloc[train_idx], y[train_idx], x.iloc[valid_idx], y[valid_idx])
    report = evaluate(y[valid_idx], model.predict_pd(x.iloc[valid_idx]))
    logger.info(
        "full model AUC=%.3f Gini=%.3f KS=%.3f Brier=%.3f",
        report.auc, report.gini, report.ks, report.brier,
    )

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.artifacts_dir / "health_model.joblib")
    (settings.artifacts_dir / "ablation.json").write_text(json.dumps(ladder, indent=2))
    return {"ladder": ladder, "full": report.as_dict()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()


if __name__ == "__main__":
    main()
