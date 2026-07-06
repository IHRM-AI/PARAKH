from parakh.eval.ablation import source_ablation
from parakh.synth.persona import TARGET, generate


def test_population_has_realistic_default_rate():
    df = generate(n=2000, seed=1)
    rate = df[TARGET].mean()
    assert 0.10 < rate < 0.40


def test_each_source_adds_signal():
    df = generate(n=3000, seed=1)
    ladder = source_ablation(df, seed=1)
    stages = list(ladder.values())
    assert stages[0] < stages[1] <= stages[2]
    assert stages[0] > 0.65
