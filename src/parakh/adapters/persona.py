from __future__ import annotations

import pandas as pd

from parakh.adapters.base import ConsentedFeedAdapter, FeatureVector, RawFeed


class PersonaFeedAdapter(ConsentedFeedAdapter):
    """Reference adapter mapping synthetic persona rows to the feature schema.

    The synthetic engine (``parakh.synth.persona``) already emits columns named
    like the model features. A production adapter never gets that luxury: a real
    rail returns nested, differently-named fields. To keep this reference honest,
    :meth:`raw_from_persona_row` first re-shapes a persona row into a rail-shaped
    raw feed with the field names each rail actually uses, and the ``map_*``
    methods then translate that raw feed back to the model schema. The round trip
    exercises the same mapping seam a live AA/GST/EPFO adapter would.
    """

    source_name = "synthetic-persona"

    @staticmethod
    def raw_from_persona_row(row: "pd.Series[float]") -> RawFeed:
        """Re-shape a persona DataFrame row into a rail-partitioned raw feed.

        The nested keys deliberately do not match the model feature names; they
        stand in for the vendor field names a real rail would return, so the
        ``map_*`` methods below have a genuine translation to perform.
        """
        return {
            "gst": {
                "avgMonthlyTurnover": float(row["gst_avg_monthly_turnover"]),
                "turnoverGrowthYoY": float(row["gst_turnover_growth"]),
                "turnoverVolatility": float(row["gst_turnover_volatility"]),
                "filingsOnTimeLast12": float(row["gst_filing_punctuality"]),
                "turnoverDecline3m": float(row["gst_turnover_decline_3m"]),
            },
            "aa": {
                "avgMonthlyCredits": float(row["bank_avg_monthly_credits"]),
                "balanceDipCount": float(row["bank_balance_dip_count"]),
                "chequeBounceCount": float(row["bank_bounce_count"]),
                "cashBufferDays": float(row["bank_cash_buffer_days"]),
                "creditToDebitRatio": float(row["bank_credit_debit_ratio"]),
                "gstBankReconciliationGap": float(row["xf_gst_bank_gap"]),
            },
            "epfo": {
                "activeHeadcount": float(row["epfo_headcount"]),
                "headcountTrend": float(row["epfo_headcount_trend"]),
            },
        }

    def map_gst(self, raw: RawFeed) -> FeatureVector:
        return {
            "gst_avg_monthly_turnover": float(raw["avgMonthlyTurnover"]),
            "gst_turnover_growth": float(raw["turnoverGrowthYoY"]),
            "gst_turnover_volatility": float(raw["turnoverVolatility"]),
            "gst_filing_punctuality": float(raw["filingsOnTimeLast12"]),
            "gst_turnover_decline_3m": float(raw["turnoverDecline3m"]),
        }

    def map_account_aggregator(self, raw: RawFeed) -> FeatureVector:
        return {
            "bank_avg_monthly_credits": float(raw["avgMonthlyCredits"]),
            "bank_balance_dip_count": float(raw["balanceDipCount"]),
            "bank_bounce_count": float(raw["chequeBounceCount"]),
            "bank_cash_buffer_days": float(raw["cashBufferDays"]),
            "bank_credit_debit_ratio": float(raw["creditToDebitRatio"]),
            "xf_gst_bank_gap": float(raw["gstBankReconciliationGap"]),
        }

    def map_epfo(self, raw: RawFeed) -> FeatureVector:
        return {
            "epfo_headcount": float(raw["activeHeadcount"]),
            "epfo_headcount_trend": float(raw["headcountTrend"]),
        }
