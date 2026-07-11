from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_ID_PREFIX = "brw_"
_ID_HEX_LEN = 12
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class BorrowerRef:
    canonical_id: str
    gstin: str | None
    display_name: str | None


def _normalise(value: str | None) -> str:
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", value).strip()


def _normalise_gstin(gstin: str | None) -> str:
    return _normalise(gstin).upper().replace(" ", "")


def _normalise_pan(pan: str | None) -> str:
    return _normalise(pan).upper().replace(" ", "")


def _normalise_name_state(name: str | None, state: str | None) -> str:
    return f"{_normalise(name).upper()}|{_normalise(state).upper()}"


def _short_hash(basis: str) -> str:
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()
    return _ID_PREFIX + digest[:_ID_HEX_LEN]


def canonical_borrower_id(identity: dict[str, str | None]) -> str:
    """Derive a stable canonical borrower id from consented identity attributes.

    Resolution order is GSTIN (primary), then PAN, then normalised name+state.
    The same borrower yields the same id in both PARAKH and TRINETRA because the
    normalisation and hashing are identical and depend only on the inputs.
    """
    gstin = _normalise_gstin(identity.get("gstin"))
    if gstin:
        return _short_hash(f"gstin:{gstin}")

    pan = _normalise_pan(identity.get("pan"))
    if pan:
        return _short_hash(f"pan:{pan}")

    name_state = _normalise_name_state(identity.get("name"), identity.get("state"))
    if name_state.strip("|"):
        return _short_hash(f"name_state:{name_state}")

    raise ValueError("identity must include at least one of: gstin, pan, or name+state.")


def resolve(identity: dict[str, str | None]) -> BorrowerRef:
    canonical_id = canonical_borrower_id(identity)
    gstin = _normalise_gstin(identity.get("gstin")) or None
    display_name = _normalise(identity.get("name")) or None
    return BorrowerRef(canonical_id=canonical_id, gstin=gstin, display_name=display_name)
