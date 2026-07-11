from __future__ import annotations

from dataclasses import replace

import pytest

from parakh.audit.ledger import (
    GENESIS_HASH,
    LIFECYCLE_EVENTS,
    AuditLedger,
    entry_hash,
    payload_hash,
    verify,
)
from parakh.consent.artefact import create_consent
from parakh.identity import canonical_borrower_id

TS = "2026-01-01T00:00:00+00:00"
SHARMA_ID = canonical_borrower_id({"gstin": "23ABCDE1234F1Z5"})


def _sharma_ledger() -> AuditLedger:
    ledger = AuditLedger(entries=[])
    consent = create_consent("104")
    ledger.record_lifecycle(SHARMA_ID, consent=consent, decision="sanction", ts=TS)
    return ledger


def test_genesis_entry_is_first():
    ledger = AuditLedger(entries=[])
    assert ledger.entries[0].seq == 0
    assert ledger.entries[0].event_type == "GENESIS"
    assert ledger.entries[0].prev_hash == GENESIS_HASH


def test_lifecycle_records_ordered_events():
    ledger = _sharma_ledger()
    chain = ledger.chain_for(SHARMA_ID)
    assert [entry.event_type for entry in chain] == list(LIFECYCLE_EVENTS)


def test_consent_to_cash_chain_verifies():
    ledger = _sharma_ledger()
    ok, first_broken_seq = verify(ledger.chain_for(SHARMA_ID))
    assert ok is True
    assert first_broken_seq is None
    # The whole ledger, including genesis, also verifies.
    assert verify(ledger.entries) == (True, None)


def test_consent_artefact_is_linked_into_the_chain():
    consent = create_consent("104")
    ledger = AuditLedger(entries=[])
    ledger.record_lifecycle(SHARMA_ID, consent=consent, decision="sanction", ts=TS)
    granted = ledger.chain_for(SHARMA_ID)[0]
    assert granted.event_type == "CONSENT_GRANTED"
    assert granted.payload_hash == payload_hash({"consent": consent.model_dump()})


def test_tamper_is_detected_at_the_right_seq():
    ledger = _sharma_ledger()
    # Mutate the DECISION entry's payload hash in place, as a tamper would.
    target_index = next(
        i for i, entry in enumerate(ledger.entries) if entry.event_type == "DECISION"
    )
    tampered_seq = ledger.entries[target_index].seq
    ledger.entries[target_index] = replace(
        ledger.entries[target_index], payload_hash="deadbeef" * 8
    )
    ok, first_broken_seq = verify(ledger.entries)
    assert ok is False
    assert first_broken_seq == tampered_seq


def test_broken_prev_linkage_is_detected():
    ledger = _sharma_ledger()
    # Re-point one entry's prev_hash and recompute its own entry_hash so the
    # entry is internally consistent but no longer links to its predecessor.
    target_index = next(
        i for i, entry in enumerate(ledger.entries) if entry.event_type == "SCORED"
    )
    original = ledger.entries[target_index]
    forged_prev = "a" * 64
    ledger.entries[target_index] = replace(
        original,
        prev_hash=forged_prev,
        entry_hash=entry_hash(
            original.seq,
            original.ts,
            original.actor,
            original.event_type,
            original.subject,
            original.payload_hash,
            forged_prev,
        ),
    )
    ok, first_broken_seq = verify(ledger.entries)
    assert ok is False
    assert first_broken_seq == original.seq


def test_unknown_decision_is_rejected():
    ledger = AuditLedger(entries=[])
    with pytest.raises(ValueError):
        ledger.record_lifecycle(
            SHARMA_ID, consent=create_consent("104"), decision="approve", ts=TS
        )


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = _sharma_ledger()
    ledger.path = path
    ledger.save()
    reloaded = AuditLedger.load(path)
    assert reloaded.entries == ledger.entries
    assert verify(reloaded.entries) == (True, None)


def test_load_missing_file_starts_fresh(tmp_path):
    ledger = AuditLedger.load(tmp_path / "absent.jsonl")
    assert ledger.entries[0].event_type == "GENESIS"
