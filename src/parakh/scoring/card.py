from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from parakh.interpret.reason_codes import CardExplainer, ReasonCode
from parakh.scoring.dimensions import grade, prequalified_limit, sub_scores
from parakh.scoring.model import HealthModel, pd_to_score
from parakh.synth.persona import BANK_FEATURES, EPFO_FEATURES, GST_FEATURES

ALL_FEATURES = GST_FEATURES + BANK_FEATURES + EPFO_FEATURES


@dataclass(frozen=True)
class HealthCard:
    score: int
    grade: str
    pd: float
    dimensions: dict[str, int]
    prequalified_limit: float
    reason_codes: list[ReasonCode] = field(default_factory=list)


class CardService:
    def __init__(self, model: HealthModel):
        self._model = model
        self._explainer = CardExplainer(model)

    @classmethod
    def from_artifacts(cls, path: Path) -> "CardService":
        return cls(joblib.load(path))

    def build(self, features: dict[str, float]) -> HealthCard:
        row = pd.Series(features)
        frame = pd.DataFrame([features]).reindex(columns=self._model.feature_names)
        pd_value = float(self._model.predict_pd(frame)[0])
        score = int(pd_to_score(pd_value))
        return HealthCard(
            score=score,
            grade=grade(score),
            pd=round(pd_value, 4),
            dimensions=sub_scores(row),
            prequalified_limit=prequalified_limit(score, float(row["bank_avg_monthly_credits"])),
            reason_codes=self._explainer.explain(frame),
        )
