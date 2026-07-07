from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from parakh.scoring.model import HealthModel


@dataclass(frozen=True)
class Phrase:
    supports_en: str
    supports_hi: str
    against_en: str
    against_hi: str


# Directional phrasing: the wording follows whether the factor supports the
# score or pulls it down, so a low-volatility firm reads "Steady sales" and a
# high-volatility firm reads "Irregular sales" from the same feature.
VERNACULAR: dict[str, Phrase] = {
    "gst_filing_punctuality": Phrase(
        "GST returns filed on time", "GST रिटर्न समय पर भरे",
        "GST returns filed late", "GST रिटर्न देर से भरे",
    ),
    "gst_turnover_decline_3m": Phrase(
        "Turnover holding up", "टर्नओवर स्थिर",
        "Recent turnover decline", "हाल में टर्नओवर घटा",
    ),
    "gst_turnover_growth": Phrase(
        "Turnover growing", "बिक्री बढ़ रही",
        "Turnover shrinking", "बिक्री घट रही",
    ),
    "gst_turnover_volatility": Phrase(
        "Steady monthly sales", "नियमित मासिक बिक्री",
        "Irregular monthly sales", "अस्थिर मासिक बिक्री",
    ),
    "bank_balance_dip_count": Phrase(
        "Balance stays healthy", "बैलेंस स्वस्थ रहता है",
        "Month-end balance running low", "महीने के अंत में बैलेंस कम",
    ),
    "bank_bounce_count": Phrase(
        "No cheque bounces", "कोई चेक बाउंस नहीं",
        "Cheque or mandate bounces", "चेक/मैंडेट बाउंस",
    ),
    "bank_cash_buffer_days": Phrase(
        "Healthy cash buffer", "अच्छा कैश बफर",
        "Thin cash buffer", "कम कैश बफर",
    ),
    "bank_credit_debit_ratio": Phrase(
        "Inflows exceed outflows", "आमदनी खर्च से ज़्यादा",
        "Outflows exceed inflows", "खर्च आमदनी से ज़्यादा",
    ),
    "xf_gst_bank_gap": Phrase(
        "GST and bank turnover reconcile", "GST और बैंक टर्नओवर मेल खाते हैं",
        "GST and bank turnover mismatch", "GST और बैंक टर्नओवर में अंतर",
    ),
    "epfo_headcount_trend": Phrase(
        "Workforce growing", "कर्मचारी बढ़ रहे",
        "Workforce shrinking", "कर्मचारी घट रहे",
    ),
    "epfo_headcount": Phrase(
        "Established workforce", "स्थापित कर्मचारी संख्या",
        "Small workforce", "कम कर्मचारी",
    ),
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
        ranked = [
            (feature, value)
            for feature, value in sorted(
                zip(x.columns, contributions), key=lambda kv: abs(kv[1]), reverse=True
            )
            if feature in VERNACULAR and abs(value) >= 1e-6
        ][:top_k]
        if not ranked:
            return []

        scale = max(abs(value) for _, value in ranked)
        codes: list[ReasonCode] = []
        for feature, value in ranked:
            phrase = VERNACULAR[feature]
            supports = bool(value < 0)
            codes.append(
                ReasonCode(
                    feature=feature,
                    english=phrase.supports_en if supports else phrase.against_en,
                    hindi=phrase.supports_hi if supports else phrase.against_hi,
                    points=max(1, int(round(abs(value) / scale * 12))),
                    supports_score=supports,
                )
            )
        return codes
