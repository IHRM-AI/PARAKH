from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from parakh.scoring.model import HealthModel
from parakh.synth.persona import BANK_FEATURES, EPFO_FEATURES, GST_FEATURES, TARGET

STAGES: list[tuple[str, list[str]]] = [
    ("GST only", GST_FEATURES),
    ("+ AA bank statements", GST_FEATURES + BANK_FEATURES),
    ("+ EPFO", GST_FEATURES + BANK_FEATURES + EPFO_FEATURES),
]


def source_ablation(df: pd.DataFrame, valid_fraction: float = 0.2, seed: int = 42) -> dict[str, float]:
    """AUC as consented data sources are stacked. Each source should add signal."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    cutoff = int(len(df) * (1 - valid_fraction))
    train_idx, valid_idx = order[:cutoff], order[cutoff:]
    y = df[TARGET].to_numpy()

    results: dict[str, float] = {}
    for name, columns in STAGES:
        x = df[columns]
        model = HealthModel().fit(
            x.iloc[train_idx], y[train_idx], x.iloc[valid_idx], y[valid_idx]
        )
        auc = roc_auc_score(y[valid_idx], model.predict_pd(x.iloc[valid_idx]))
        results[name] = round(float(auc), 3)
    return results
