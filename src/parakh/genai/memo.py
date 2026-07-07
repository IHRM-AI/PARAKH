from __future__ import annotations

from dataclasses import dataclass

from parakh.genai.llm import GemmaClient
from parakh.genai.text import plain_text
from parakh.scoring.card import HealthCard

SYSTEM_PROMPT = (
    "You are a credit officer at an Indian bank assessing an MSME for a working "
    "capital facility. Write a short, factual sanction note from the supplied "
    "figures only. Do not invent numbers. Flag any GST-versus-bank divergence. "
    "End with a recommendation. The officer approves before sanction. Write in "
    "plain prose with simple headings and hyphen bullets; do not use Markdown "
    "symbols such as asterisks or hashes."
)


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
    def _facts(borrower: str, card: HealthCard) -> list[str]:
        top = ", ".join(f"{code.english} ({'+' if code.supports_score else '-'})" for code in card.reason_codes)
        facts = [
            f"Borrower: {borrower}",
            f"Health score: {card.score}/100 (grade {card.grade}, PD {card.pd:.1%})",
            f"Pre-qualified limit: {card.prequalified_limit:,.0f}",
            f"GST-vs-bank divergence: {card.divergence_gap:.0%}",
            f"Key factors: {top}",
        ]
        return facts

    def _prompt(self, borrower: str, card: HealthCard) -> str:
        return "\n".join(self._facts(borrower, card))

    def _template(self, borrower: str, card: HealthCard) -> str:
        lines = self._facts(borrower, card)
        if card.divergence_flag:
            lines.append(
                "Divergence alert: declared GST turnover materially exceeds observed "
                "bank credits; refer to manual review before sanction."
            )
            recommendation = "refer to manual review"
        elif card.score >= 68:
            recommendation = f"eligible for the pre-qualified limit of {card.prequalified_limit:,.0f}"
        else:
            recommendation = "seek additional collateral or a lower limit"
        lines.append(f"Recommendation: {recommendation}.")
        return "\n".join(lines)
