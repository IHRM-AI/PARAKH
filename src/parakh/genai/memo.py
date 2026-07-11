from __future__ import annotations

from dataclasses import dataclass

from parakh.genai.llm import GemmaClient
from parakh.genai.text import plain_text
from parakh.scoring.card import HealthCard

SYSTEM_PROMPT = (
    "You are a credit officer at an Indian bank assessing a new-to-credit MSME for a "
    "working-capital facility on consented alternate data. Draft a formal credit "
    "assessment memo using only the supplied figures; never invent numbers. Produce "
    "these numbered sections: 1. Applicant; 2. Health score; 3. Score breakdown; "
    "4. GST and banking reconciliation; 5. Key factors; 6. Recommended facility; "
    "7. Conditions and monitoring. Two to four sentences per section, hyphen bullets "
    "for lists. Interpret the score and the divergence for the officer. Flag any "
    "GST-versus-bank divergence. Close by noting the memo is model-assisted and "
    "requires officer approval before sanction. Do not use Markdown symbols such as "
    "asterisks or hashes."
)


def _score_narrative(score: int) -> str:
    if score >= 68:
        return (
            "This is a healthy profile that supports the pre-qualified facility on "
            "consented data alone"
        )
    if score >= 45:
        return (
            "This is a moderate profile; the facility is supportable with conditions "
            "and a reduced opening limit"
        )
    return "This is a weak profile; the exposure needs mitigation before sanction"


@dataclass(frozen=True)
class LenderMemo:
    borrower: str
    body: str
    status: str
    generated_by: str


class LenderMemoService:
    def __init__(self, llm: GemmaClient | None = None):
        self._llm = llm or GemmaClient()

    def draft(self, borrower: str, card: HealthCard) -> LenderMemo:
        body = self._template(borrower, card)
        generated_by = "deterministic template (LLM offline)"
        if self._llm.available:
            try:
                body = plain_text(self._llm.complete(SYSTEM_PROMPT, self._prompt(borrower, card)))
                generated_by = f"Gemma-4 ({self._llm.model})"
            except Exception:
                generated_by = "deterministic template (LLM unreachable)"
        return LenderMemo(
            borrower=borrower,
            body=body,
            status="Awaiting officer approval",
            generated_by=generated_by,
        )

    @staticmethod
    def _reasons(card: HealthCard) -> list[str]:
        rows = []
        for code in card.reason_codes:
            sign = "supports" if code.supports_score else "weighs against"
            hindi = getattr(code, "hindi", "") or ""
            label = f"{code.english} / {hindi}" if hindi else code.english
            rows.append(f"- {label} ({sign} the score)")
        return rows

    def _facts(self, borrower: str, card: HealthCard) -> list[str]:
        facts = [
            f"Borrower: {borrower}",
            f"Health score: {card.score}/100 (grade {card.grade}); model-implied 12-month PD {card.pd:.1%}",
            f"Pre-qualified limit: {card.prequalified_limit:,.0f}",
            f"GST-vs-bank divergence: {card.divergence_gap:.0%} "
            f"({'ALERT' if card.divergence_flag else 'within tolerance'})",
            "Dimension sub-scores: "
            + ", ".join(f"{name} {value}/100" for name, value in card.dimensions.items()),
            "Key factors:",
            *self._reasons(card),
        ]
        return facts

    def _prompt(self, borrower: str, card: HealthCard) -> str:
        return "\n".join(self._facts(borrower, card))

    def _recommendation(self, card: HealthCard) -> str:
        if card.divergence_flag:
            return (
                "Refer to manual review before sanction: declared GST turnover materially "
                "exceeds observed bank credits, so the consented data cannot yet be relied on."
            )
        if card.score >= 68:
            return (
                f"Eligible for the pre-qualified working-capital limit of "
                f"{card.prequalified_limit:,.0f} on consented data."
            )
        if card.score >= 45:
            return (
                f"Supportable at a reduced opening limit against the indicative "
                f"{card.prequalified_limit:,.0f}, with additional comfort as below."
            )
        return "Seek additional collateral or decline at the current exposure level."

    def _template(self, borrower: str, card: HealthCard) -> str:
        parts = [
            "MSME CREDIT ASSESSMENT MEMO",
            "",
            "1. Applicant",
            f"Borrower: {borrower}",
            "Facility: working-capital limit for a new-to-credit MSME, scored on consented "
            "Account Aggregator, GST and EPFO data.",
            "",
            "2. Health score",
            f"Composite health score {card.score}/100 (grade {card.grade}); the model implies a "
            f"12-month probability of default of {card.pd:.1%}. {_score_narrative(card.score)}.",
            "",
            "3. Score breakdown",
            "Dimension sub-scores from the consented data:",
            *(f"- {name}: {value}/100" for name, value in card.dimensions.items()),
            "",
            "4. GST and banking reconciliation",
            f"GST-vs-bank divergence: {card.divergence_gap:.0%}. "
            + (
                "This exceeds tolerance and is a red flag; declared turnover is not fully "
                "corroborated by bank credits."
                if card.divergence_flag
                else "Declared GST turnover and observed bank credits reconcile within tolerance."
            ),
            "",
            "5. Key factors",
            *self._reasons(card),
            "",
            "6. Recommended facility",
            self._recommendation(card),
            "",
            "7. Conditions and monitoring",
            "- Every data fetch and decision is consent-gated and written to the DPDP audit ledger.",
            "- Place under 12-month score surveillance; a watch fires if the card falls below 55.",
            "- Re-confirm consent renewal and GST-bank reconciliation at each review.",
            "",
            "This memo is model-assisted and requires officer approval before sanction.",
        ]
        return "\n".join(parts)
