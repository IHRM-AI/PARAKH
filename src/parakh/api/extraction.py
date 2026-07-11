from __future__ import annotations

import re
from dataclasses import dataclass, field

GSTIN_PATTERN = re.compile(
    r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d]{2})\b"
)

_NUMBER = r"[-+]?[\d,]*\.?\d+"


@dataclass
class Extraction:
    """Result of parsing a document into NewMsmeForm fields.

    ``fields`` holds only the keys the parser actually found, so the officer
    fills the rest by hand. ``source`` labels the provenance for the UI.
    """

    fields: dict[str, float | str] = field(default_factory=dict)
    source: str = "ocr"


# Labelled-number rules keyed to the exact NewMsmeForm feature names. Each entry
# maps a form field to the label phrases that may introduce its value in an OCR
# transcript. Phrases are matched case-insensitively and the first numeric token
# after the label wins.
_NUMBER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("gst_avg_monthly_turnover", ("gst turnover", "monthly turnover", "turnover per month", "average monthly turnover")),
    ("gst_turnover_growth", ("turnover growth", "growth rate", "yoy growth")),
    ("gst_turnover_volatility", ("turnover volatility", "volatility")),
    ("gst_filing_punctuality", ("gst filings on time", "filings on time", "gst filing punctuality", "returns filed on time")),
    ("gst_turnover_decline_3m", ("3-month decline", "3 month decline", "three-month decline", "quarterly decline")),
    ("bank_avg_monthly_credits", ("bank credits", "monthly credits", "average monthly credits", "bank credits per month")),
    ("bank_balance_dip_count", ("balance dips", "month-end balance dips", "balance dip count")),
    ("bank_bounce_count", ("cheque bounces", "cheque bounce", "bounced cheques", "bounce count")),
    ("bank_cash_buffer_days", ("cash buffer", "cash buffer days", "buffer days")),
    ("bank_credit_debit_ratio", ("credit/debit ratio", "credit-debit ratio", "credit to debit ratio")),
    ("xf_gst_bank_gap", ("gst-vs-bank gap", "gst vs bank gap", "gst-bank gap", "gst bank gap")),
    ("epfo_headcount", ("epfo headcount", "headcount", "employees on epfo", "number of employees")),
    ("epfo_headcount_trend", ("headcount trend", "employee growth")),
]

_TEXT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("name", ("business name", "firm name", "trade name", "name of business", "legal name")),
    ("location", ("location", "address", "place of business", "city")),
]


def _first_number(segment: str) -> float | None:
    match = re.search(_NUMBER, segment)
    if match is None:
        return None
    token = match.group(0).replace(",", "")
    try:
        return float(token)
    except ValueError:  # pragma: no cover - defensive: regex only yields numeric tokens
        return None


def _match_labelled_number(text_lower: str, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        index = text_lower.find(label)
        if index == -1:
            continue
        tail = text_lower[index + len(label): index + len(label) + 40]
        value = _first_number(tail)
        if value is not None:
            return value
    return None


def _match_labelled_text(text: str, labels: tuple[str, ...]) -> str | None:
    for line in text.splitlines():
        lower = line.lower()
        for label in labels:
            index = lower.find(label)
            if index == -1:
                continue
            tail = line[index + len(label):]
            cleaned = tail.lstrip(" \t:-–—").strip()
            if cleaned:
                return cleaned
    return None


def parse_document_text(text: str) -> dict[str, float | str]:
    """Parse OCR text into a partial dict of NewMsmeForm fields.

    Only fields the parser is confident about are returned. The 15-character
    GSTIN is matched by regex; business name and location come from labelled
    lines; the remaining figures come from labelled numbers. Everything else is
    left for the officer to complete.
    """
    fields: dict[str, float | str] = {}

    gstin_match = GSTIN_PATTERN.search(text.upper())
    if gstin_match:
        fields["gstin"] = gstin_match.group(1)

    for key, labels in _TEXT_RULES:
        found = _match_labelled_text(text, labels)
        if found is not None:
            fields[key] = found

    text_lower = text.lower()
    for key, labels in _NUMBER_RULES:
        value = _match_labelled_number(text_lower, labels)
        if value is not None:
            fields[key] = value

    return fields


# Canned extraction for the offline demo path. Mirrors a representative firm so
# a judge can exercise the auto-fill flow without the GPU-hosted OCR service.
DEMO_FIXTURE: dict[str, float | str] = {
    "name": "Gupta Provisions",
    "location": "Kanpur, UP",
    "gstin": "09ABCDE1234F1Z5",
    "gst_avg_monthly_turnover": 620000,
    "gst_turnover_growth": 0.09,
    "gst_turnover_volatility": 0.16,
    "gst_filing_punctuality": 11,
    "gst_turnover_decline_3m": 0.0,
    "bank_avg_monthly_credits": 574000,
    "bank_balance_dip_count": 2,
    "bank_bounce_count": 0,
    "bank_cash_buffer_days": 26,
    "bank_credit_debit_ratio": 1.08,
    "xf_gst_bank_gap": 0.07,
    "epfo_headcount": 7,
    "epfo_headcount_trend": 0.04,
}

DEMO_SOURCE = "demo fixture — OCR offline"

_DEMO_FILENAME_HINTS = ("sample", "gupta")


def is_demo_request(filename: str | None, demo_flag: bool) -> bool:
    """Decide whether to serve the canned demo fixture.

    True when the caller passes ``demo=true`` or uploads a recognisable sample
    filename, so the auto-fill flow stays demonstrable while the OCR GPU is off.
    """
    if demo_flag:
        return True
    if not filename:
        return False
    lowered = filename.lower()
    return any(hint in lowered for hint in _DEMO_FILENAME_HINTS)


def demo_extraction() -> Extraction:
    return Extraction(fields=dict(DEMO_FIXTURE), source=DEMO_SOURCE)
