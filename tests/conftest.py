from __future__ import annotations

import pytest

from parakh.scoring.card import ALL_FEATURES
from parakh.scoring.model import HealthModel
from parakh.synth.persona import TARGET, generate


@pytest.fixture(scope="session")
def trained_model() -> HealthModel:
    df = generate(n=2000, seed=2)
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel(num_boost_round=150, early_stopping_rounds=30)
    model.fit(x.iloc[:1600], y[:1600], x.iloc[1600:], y[1600:])
    return model


@pytest.fixture
def valid_features() -> dict[str, float]:
    df = generate(n=20, seed=9)
    return {name: float(df.iloc[0][name]) for name in ALL_FEATURES}
