from parakh.scoring.card import (
    ALL_FEATURES,
    PD_REFER_HIGH,
    PD_REFER_LOW,
    CardService,
    decide,
)
from parakh.scoring.dimensions import grade, prequalified_limit, sub_scores
from parakh.scoring.model import MONOTONE_DIRECTIONS, HealthModel, monotone_vector
from parakh.synth.persona import TARGET, generate


def _trained_service() -> CardService:
    df = generate(n=2000, seed=2)
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel(num_boost_round=150, early_stopping_rounds=30)
    model.fit(x.iloc[:1600], y[:1600], x.iloc[1600:], y[1600:])
    return CardService(model)


def test_grade_bands_are_monotone():
    assert grade(85) == "A"
    assert grade(70) == "B+"
    assert grade(40) == "D"


def test_prequalified_limit_increases_with_score():
    assert prequalified_limit(80, 400000) > prequalified_limit(50, 400000)


def test_dimensions_are_bounded():
    df = generate(n=50, seed=3)
    scores = sub_scores(df.iloc[0])
    assert len(scores) == 6
    assert all(0 <= value <= 100 for value in scores.values())


def test_card_is_well_formed():
    service = _trained_service()
    df = generate(n=20, seed=9)
    features = {k: float(df.iloc[0][k]) for k in ALL_FEATURES}
    card = service.build(features)
    assert 0 <= card.score <= 100
    assert card.grade in {"A", "B+", "B", "C", "D"}
    assert card.reason_codes


def test_monotone_vector_matches_directions_and_covers_features():
    names = ["bank_bounce_count", "bank_cash_buffer_days", "unknown_feature"]
    assert monotone_vector(names) == [1, -1, 0]
    for feature in ALL_FEATURES:
        assert MONOTONE_DIRECTIONS.get(feature) in {-1, 1}


def test_monotone_constraint_reaches_the_booster():
    df = generate(n=1500, seed=11)
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel(num_boost_round=80, early_stopping_rounds=20)
    model.fit(x.iloc[:1200], y[:1200], x.iloc[1200:], y[1200:])
    reported = model.booster.params["monotone_constraints"]
    assert list(reported) == monotone_vector(ALL_FEATURES)


def test_decide_bands_are_ordered():
    assert decide(PD_REFER_LOW - 0.05)[0] == "auto-eligible"
    assert decide((PD_REFER_LOW + PD_REFER_HIGH) / 2)[0] == "refer to credit officer"
    assert decide(PD_REFER_HIGH + 0.05)[0] == "decline"


def test_card_carries_a_decision():
    service = _trained_service()
    df = generate(n=20, seed=9)
    features = {k: float(df.iloc[0][k]) for k in ALL_FEATURES}
    card = service.build(features)
    assert card.decision in {"auto-eligible", "refer to credit officer", "decline"}
    assert card.decision_reason
