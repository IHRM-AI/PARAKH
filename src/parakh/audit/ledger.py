from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from parakh.config import settings
from parakh.consent.artefact import ConsentArtefact

GENESIS_HASH = "0" * 64

# Ordered consent-to-cash lifecycle. Each borrower's chain walks these in order,
# from the consent that authorises the fetch through to the monitoring watch.
LIFECYCLE_EVENTS = (
    "CONSENT_GRANTED",
    "DATA_FETCHED",
    "SCORED",
    "DECISION",
    "MONITORING_WATCH",
)

_DEFAULT_PATH = settings.artifacts_dir / "audit_ledger.jsonl"


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    ts: str
    actor: str
    event_type: str
    subject: str
    payload_hash: str
    prev_hash: str
    entry_hash: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_hash(payload: object) -> str:
    return _sha256_hex(_canonical_json(payload))


def entry_hash(
    seq: int,
    ts: str,
    actor: str,
    event_type: str,
    subject: str,
    payload_hash_hex: str,
    prev_hash: str,
) -> str:
    basis = "|".join(
        [str(seq), ts, actor, event_type, subject, payload_hash_hex, prev_hash]
    )
    return _sha256_hex(basis)


@dataclass
class AuditLedger:
    """Append-only, hash-chained audit ledger.

    Each entry links to the previous by hash, so altering any earlier entry
    breaks the chain from that point on. This is a tamper-evidence mechanism, not
    a blockchain and not an immutable store: the JSONL file itself can be edited,
    but verify() will detect the edit. In production the same chain would be
    backed by a WORM store or an append-only, KMS-signed log (e.g. AWS QLDB); the
    demo proves the integrity mechanism.
    """

    path: Path = field(default_factory=lambda: _DEFAULT_PATH)
    entries: list[LedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.entries:
            self._append_genesis()

    def _last_hash(self) -> str:
        return self.entries[-1].entry_hash if self.entries else GENESIS_HASH

    def _append_genesis(self) -> None:
        ts = "1970-01-01T00:00:00+00:00"
        p_hash = payload_hash({"note": "genesis"})
        e_hash = entry_hash(0, ts, "system", "GENESIS", "", p_hash, GENESIS_HASH)
        self.entries.append(
            LedgerEntry(
                seq=0,
                ts=ts,
                actor="system",
                event_type="GENESIS",
                subject="",
                payload_hash=p_hash,
                prev_hash=GENESIS_HASH,
                entry_hash=e_hash,
            )
        )

    def append(
        self,
        *,
        actor: str,
        event_type: str,
        subject: str,
        payload: object,
        ts: str,
    ) -> LedgerEntry:
        seq = len(self.entries)
        prev_hash = self._last_hash()
        p_hash = payload_hash(payload)
        e_hash = entry_hash(seq, ts, actor, event_type, subject, p_hash, prev_hash)
        entry = LedgerEntry(
            seq=seq,
            ts=ts,
            actor=actor,
            event_type=event_type,
            subject=subject,
            payload_hash=p_hash,
            prev_hash=prev_hash,
            entry_hash=e_hash,
        )
        self.entries.append(entry)
        return entry

    def record_lifecycle(
        self,
        canonical_id: str,
        *,
        consent: ConsentArtefact,
        decision: str,
        ts: str,
        actor: str = "parakh",
    ) -> list[LedgerEntry]:
        """Append the ordered consent-to-cash sequence for one borrower.

        The chain links back to the ConsentArtefact: its hash is embedded in the
        CONSENT_GRANTED payload, so the granted consent is part of the tamper-
        evident record.
        """
        if decision not in {"sanction", "refer", "decline"}:
            raise ValueError(
                f"Unknown decision {decision!r}; expected sanction, refer or decline."
            )
        consent_payload = {"consent": consent.model_dump()}
        payloads: dict[str, object] = {
            "CONSENT_GRANTED": consent_payload,
            "DATA_FETCHED": {"purpose_code": consent.purpose_code},
            "SCORED": {"consent_hash": payload_hash(consent_payload)},
            "DECISION": {"decision": decision},
            "MONITORING_WATCH": {"purpose_code": consent.purpose_code},
        }
        return [
            self.append(
                actor=actor,
                event_type=event_type,
                subject=canonical_id,
                payload=payloads[event_type],
                ts=ts,
            )
            for event_type in LIFECYCLE_EVENTS
        ]

    def chain_for(self, canonical_id: str) -> list[LedgerEntry]:
        return [entry for entry in self.entries if entry.subject == canonical_id]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [_canonical_json(asdict(entry)) for entry in self.entries]
        self.path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> AuditLedger:
        target = path or _DEFAULT_PATH
        if not target.exists():
            return cls(path=target)
        entries: list[LedgerEntry] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(LedgerEntry(**json.loads(line)))
        return cls(path=target, entries=entries)


def verify(chain: list[LedgerEntry]) -> tuple[bool, int | None]:
    """Recompute every entry hash and the prev linkage.

    Returns (ok, first_broken_seq). first_broken_seq is the seq of the earliest
    entry whose recomputed hash or prev linkage does not match; None when the
    chain verifies. The linkage is checked against each entry's own recorded
    prev_hash, so this works on both the full ledger and a subject sub-chain
    (e.g. one borrower's entries), while still catching any mutated field: a
    changed field breaks that entry's recomputed hash, and the next entry's
    prev_hash no longer matches it.
    """
    expected_prev: str | None = None
    for entry in chain:
        recomputed = entry_hash(
            entry.seq,
            entry.ts,
            entry.actor,
            entry.event_type,
            entry.subject,
            entry.payload_hash,
            entry.prev_hash,
        )
        if recomputed != entry.entry_hash:
            return False, entry.seq
        if expected_prev is not None and entry.prev_hash != expected_prev:
            return False, entry.seq
        expected_prev = entry.entry_hash
    return True, None
