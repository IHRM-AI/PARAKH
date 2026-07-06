from parakh.scoring.card import ALL_FEATURES, CardService
from parakh.scoring.dimensions import grade, prequalified_limit, sub_scores
from parakh.scoring.model import HealthModel
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
