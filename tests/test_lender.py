from parakh.genai.llm import GemmaClient
from parakh.genai.memo import LenderMemoService
from parakh.scoring.card import ALL_FEATURES, CardService
from parakh.scoring.model import HealthModel
from parakh.synth.persona import TARGET, generate


def _service() -> tuple[CardService, dict[str, float]]:
    df = generate(n=2000, seed=4)
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    model = HealthModel(num_boost_round=150, early_stopping_rounds=30)
    model.fit(x.iloc[:1600], y[:1600], x.iloc[1600:], y[1600:])
    features = {k: float(df.iloc[0][k]) for k in ALL_FEATURES}
    return CardService(model), features


def test_what_if_improves_score():
    service, features = _service()
    result = service.what_if(
        features,
        {"bank_cash_buffer_days": 60, "bank_bounce_count": 0, "bank_balance_dip_count": 0, "xf_gst_bank_gap": 0.02},
    )
    assert result["after"]["score"] >= result["before"]["score"]


def test_divergence_flag_fires():
    service, features = _service()
    card = service.build({**features, "xf_gst_bank_gap": 0.45})
    assert card.divergence_flag is True


def test_lender_memo_is_officer_gated():
    service, features = _service()
    card = service.build(features)
    memo = LenderMemoService(llm=GemmaClient(base_url="")).draft("Sharma Kirana Store", card)
    assert memo.status == "Awaiting officer approval"
    assert "Sharma Kirana Store" in memo.body
