from __future__ import annotations

import math

import pytest

from parakh.adapters import (
    AdapterError,
    ConsentedFeedAdapter,
    PersonaFeedAdapter,
)
from parakh.adapters.base import RAIL_FEATURES, validate_feature_vector
from parakh.scoring.card import ALL_FEATURES, CardService
from parakh.scoring.model import HealthModel
from parakh.synth.persona import BANK_FEATURES, EPFO_FEATURES, GST_FEATURES, generate


@pytest.fixture
def persona_row():
    return generate(n=8, seed=11).iloc[0]


def test_persona_adapter_produces_complete_vector(persona_row):
    adapter = PersonaFeedAdapter()
    raw = adapter.raw_from_persona_row(persona_row)
    vector = adapter.to_feature_vector(raw)
    assert set(vector) == set(ALL_FEATURES)
    assert all(math.isfinite(value) for value in vector.values())


def test_persona_adapter_round_trips_values(persona_row):
    adapter = PersonaFeedAdapter()
    raw = adapter.raw_from_persona_row(persona_row)
    vector = adapter.to_feature_vector(raw)
    for name in ALL_FEATURES:
        assert vector[name] == pytest.approx(float(persona_row[name]))


def test_rail_features_partition_the_schema():
    covered = RAIL_FEATURES["gst"] + RAIL_FEATURES["aa"] + RAIL_FEATURES["epfo"]
    assert covered == GST_FEATURES + BANK_FEATURES + EPFO_FEATURES
    assert sorted(covered) == sorted(ALL_FEATURES)


def test_adapter_output_scores_through_the_model(persona_row, trained_model: HealthModel):
    adapter = PersonaFeedAdapter()
    vector = adapter.to_feature_vector(adapter.raw_from_persona_row(persona_row))
    card = CardService(trained_model).build(vector)
    assert 0 <= card.score <= 100
    assert card.reason_codes


def test_validate_rejects_missing_feature():
    with pytest.raises(AdapterError, match="missing features"):
        validate_feature_vector({"gst_avg_monthly_turnover": 1.0})


def test_validate_rejects_unknown_feature():
    vector = {name: 0.0 for name in ALL_FEATURES}
    vector["not_a_feature"] = 1.0
    with pytest.raises(AdapterError, match="unknown features"):
        validate_feature_vector(vector)


def test_validate_rejects_non_finite_value():
    vector = {name: 0.0 for name in ALL_FEATURES}
    vector["gst_avg_monthly_turnover"] = math.nan
    with pytest.raises(AdapterError, match="non-finite"):
        validate_feature_vector(vector)


def test_source_name_is_declared():
    assert PersonaFeedAdapter().source_name == "synthetic-persona"
    assert issubclass(PersonaFeedAdapter, ConsentedFeedAdapter)
