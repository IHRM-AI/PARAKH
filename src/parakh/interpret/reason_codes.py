from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from parakh.scoring.model import HealthModel


@dataclass(frozen=True)
class Phrase:
    english: str
    hindi: str


VERNACULAR: dict[str, Phrase] = {
    "gst_filing_punctuality": Phrase("GST returns filed on time", "GST रिटर्न समय पर भरे"),
    "gst_turnover_decline_3m": Phrase("Recent turnover decline", "हाल में टर्नओवर घटा"),
    "gst_turnover_growth": Phrase("Turnover growth", "बिक्री में बढ़ोतरी"),
    "gst_turnover_volatility": Phrase("Irregular monthly sales", "हर महीने बिक्री में उतार-चढ़ाव"),
    "bank_balance_dip_count": Phrase("Month-end balance running low", "महीने के अंत में बैलेंस कम"),
    "bank_bounce_count": Phrase("Cheque or mandate bounces", "चेक/मैंडेट बाउंस"),
    "bank_cash_buffer_days": Phrase("Healthy cash buffer", "अच्छा कैश बफर"),
    "bank_credit_debit_ratio": Phrase("Inflows exceed outflows", "आमदनी खर्च से ज़्यादा"),
    "xf_gst_bank_gap": Phrase("GST and bank turnover mismatch", "GST और बैंक टर्नओवर में अंतर"),
    "epfo_headcount_trend": Phrase("Workforce trend", "कर्मचारियों की संख्या का रुझान"),
    "epfo_headcount": Phrase("Registered workforce", "पंजीकृत कर्मचारी"),
}


@dataclass(frozen=True)
class ReasonCode:
    feature: str
    english: str
    hindi: str
    points: int
    supports_score: bool


class CardExplainer:
    def __init__(self, model: HealthModel):
        if model.booster is None:
            raise RuntimeError("Model is not trained.")
        self._explainer = shap.TreeExplainer(model.booster)

    def explain(self, x: pd.DataFrame, top_k: int = 4) -> list[ReasonCode]:
        values = self._explainer.shap_values(x.iloc[[0]])
        if isinstance(values, list):
            values = values[1]
        contributions = values[0]
        ranked = sorted(
            zip(x.columns, contributions), key=lambda kv: abs(kv[1]), reverse=True
        )
        codes: list[ReasonCode] = []
        for feature, value in ranked:
            phrase = VERNACULAR.get(feature)
            if phrase is None or abs(value) < 1e-6:
                continue
            codes.append(
                ReasonCode(
                    feature=feature,
                    english=phrase.english,
                    hindi=phrase.hindi,
                    points=int(round(abs(value) * 100)),
                    supports_score=bool(value < 0),
                )
            )
            if len(codes) >= top_k:
                break
        return codes
