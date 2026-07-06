from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

IST = timezone(timedelta(hours=5, minutes=30))

# Account Aggregator consent purpose codes (ReBIT taxonomy).
PURPOSE = {
    "103": "One-time aggregated statement for underwriting",
    "104": "Periodic monitoring of financial position",
}


class ConsentArtefact(BaseModel):
    consent_start: str
    consent_expiry: str
    purpose_code: str
    purpose_text: str
    fi_types: list[str]
    data_life_unit: str
    data_life_value: int
    fetch_type: str
    frequency_unit: str
    frequency_value: int


def create_consent(purpose_code: str, issued_at: datetime | None = None) -> ConsentArtefact:
    if purpose_code not in PURPOSE:
        raise ValueError(f"Unknown purpose code {purpose_code!r}; expected one of {list(PURPOSE)}.")
    issued = issued_at or datetime.now(IST)
    periodic = purpose_code == "104"
    return ConsentArtefact(
        consent_start=issued.isoformat(timespec="seconds"),
        consent_expiry=(issued + timedelta(days=365 if periodic else 1)).isoformat(timespec="seconds"),
        purpose_code=purpose_code,
        purpose_text=PURPOSE[purpose_code],
        fi_types=["DEPOSIT"],
        data_life_unit="MONTH",
        data_life_value=12,
        fetch_type="PERIODIC" if periodic else "ONETIME",
        frequency_unit="MONTH",
        frequency_value=1 if periodic else 0,
    )
