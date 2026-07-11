# Consent-to-cash audit ledger — hash-chained, tamper-evident

Every lending decision in PARAKH starts with a consent and ends with a decision
that may put the borrower on a monitoring watch. The audit ledger records that
lifecycle as an append-only, **hash-chained** sequence so the record is
**tamper-evident**: altering any past entry breaks the chain from that point on,
and `verify()` reports exactly where.

This is a tamper-evidence mechanism, **not a blockchain and not an immutable
store**. The JSONL file itself can be edited on disk; what the chain guarantees
is that such an edit is detectable. In production the same chain would be backed
by a WORM store or an append-only, KMS-signed log (for example AWS QLDB); the
demo proves the integrity mechanism, not the storage substrate.

## Entry shape

Each entry (`src/parakh/audit/ledger.py`, `LedgerEntry`) is:

| field          | meaning                                                             |
| -------------- | ------------------------------------------------------------------- |
| `seq`          | position in the ledger, starting at 0 (genesis)                     |
| `ts`           | injected timestamp (no wall-clock in the hash, so hashes are stable)|
| `actor`        | who recorded the event                                              |
| `event_type`   | lifecycle event (see below)                                         |
| `subject`      | canonical borrower id                                               |
| `payload_hash` | `sha256` hex of the canonical-JSON event payload                    |
| `prev_hash`    | `entry_hash` of the previous entry                                  |
| `entry_hash`   | `sha256` hex over `seq\|ts\|actor\|event_type\|subject\|payload_hash\|prev_hash` |

The ledger starts with a genesis entry (`seq=0`, `prev_hash` all zeros). Payloads
are hashed as canonical JSON (sorted keys, no whitespace) so the hash is
reproducible across processes.

## Consent-to-cash lifecycle

`record_lifecycle(canonical_id, consent, decision, ts)` appends the ordered
sequence for one borrower:

```
CONSENT_GRANTED → DATA_FETCHED → SCORED → DECISION → MONITORING_WATCH
```

The `CONSENT_GRANTED` payload embeds the `ConsentArtefact` (ReBIT purpose code
103/104), and `SCORED` embeds the consent's payload hash, so the granting consent
is linked into the tamper-evident record rather than sitting beside it. `DECISION`
carries one of `sanction`, `refer`, `decline`.

## Endpoints

```
GET /audit/{canonical_id}          -> {canonical_id, entries: [...]}   # ordered chain
GET /audit/{canonical_id}/verify   -> {canonical_id, ok, first_broken_seq}
```

A sample verified chain for Sharma Kirana Store (`brw_99200b2d0b99`):

```json
{
  "canonical_id": "brw_99200b2d0b99",
  "ok": true,
  "first_broken_seq": null
}
```

with event types `["CONSENT_GRANTED", "DATA_FETCHED", "SCORED", "DECISION",
"MONITORING_WATCH"]`.

The write path is internal: a seeded ledger is built in the API lifespan so the
demo has a real chain to serve. There is no public `POST` — the ledger is
append-only from the application's side, not client-writable.

## Verification and tamper detection

`verify(chain)` recomputes every `entry_hash` and checks each entry's `prev_hash`
against the previous entry's hash, returning `(ok, first_broken_seq)`. It works on
the full ledger and on a subject sub-chain (one borrower's entries), because the
linkage is checked against each entry's own recorded `prev_hash`.

If any field of an entry is mutated, that entry's recomputed hash no longer
matches its stored `entry_hash`, and the following entry's `prev_hash` no longer
matches — so `verify()` returns `ok=False` with `first_broken_seq` set to the
mutated entry's `seq`. The test `test_tamper_is_detected_at_the_right_seq` mutates
the `DECISION` entry's payload hash and asserts `verify()` returns `ok=False` at
that seq.
