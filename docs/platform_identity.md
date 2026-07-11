# Platform identity — one borrower graph across originate → monitor

PARAKH (origination) and TRINETRA (monitoring) are separate services, but they
describe the same MSMEs. Before this change the correspondence was a hardcoded
per-borrower map in the frontend: PARAKH id `sharma` → TRINETRA id
`sharma-kirana`. That map is a manual join — it has to be maintained by hand and
it only knows about borrowers someone remembered to add.

The platform now resolves both sides to one **canonical borrower id** derived
from consented identity attributes. The same borrower yields the same id in both
services independently, so origination can hand off to monitoring by identity
rather than by lookup table.

## The canonical id

`canonical_borrower_id(identity)` (in `src/parakh/identity.py`, byte-identical to
`trinetra/src/trinetra/identity.py`) derives a stable id from consented identity
attributes:

1. **GSTIN** (primary) — normalised: whitespace collapsed, trimmed, upper-cased.
2. **PAN** (fallback) — same normalisation.
3. **Name + state** (last resort) — normalised and upper-cased.

The chosen basis is prefixed (`gstin:`, `pan:`, `name_state:`), hashed with
SHA-256, and the first 12 hex characters are taken with a `brw_` prefix:

```
canonical_id = "brw_" + sha256("gstin:" + normalised_gstin).hexdigest()[:12]
```

For Sharma Kirana Store, GSTIN `23ABCDE1234F1Z5`:

```
canonical_borrower_id({"gstin": "23ABCDE1234F1Z5"}) == "brw_99200b2d0b99"
```

`resolve(identity)` returns a `BorrowerRef{canonical_id, gstin, display_name}`
for callers that also want the human-readable fields.

Because the id is a deterministic function of the normalised inputs only, both
repos compute the same value with no shared state. The identity module carries no
package-specific imports, which is why the two copies are identical.

## How the live handoff uses it

- **TRINETRA** (`src/trinetra/api/portfolio.py`) pins the shared lifecycle
  account's `id` to `canonical_borrower_id({"gstin": SHARED_ACCOUNT_GSTIN})`,
  keeping `name`/`gstin` for display. The `/portfolio` response therefore
  identifies Sharma as `brw_99200b2d0b99`.
- **PARAKH** (`frontend/src/handoff.ts`) derives the same id from the borrower's
  consented GSTIN with a small synchronous SHA-256 port kept in lockstep with the
  Python module, and deep-links to `/trinetra/?focus=brw_99200b2d0b99`.
- **TRINETRA** (`frontend/src/App.tsx`) matches the `focus` param against the
  portfolio row id and opens the drill-down.

The result: one click in origination lands on the right monitored account, with
no manual join and no per-borrower map to maintain.

## `/resolve` endpoint

PARAKH exposes a dependency-free resolver:

```
GET /resolve?gstin=<gstin>[&pan=&name=&state=]
```

```json
{
  "canonical_id": "brw_99200b2d0b99",
  "gstin": "23ABCDE1234F1Z5",
  "display_name": "Sharma Kirana Store",
  "originated": true,
  "monitored": true
}
```

`originated` and `monitored` are presence flags. In this demo they are backed by
a small seeded registry; in production they would query the origination and
monitoring stores by canonical id.

## Scope and honesty

This is entity resolution by a single consented identifier (GSTIN first), not a
probabilistic matcher. It does not de-duplicate typo'd names, merge related
entities, or reconcile a borrower who consents under different identifiers across
services. It gives one stable, reproducible key for the same GSTIN so the two
services share one borrower graph instead of a hand-maintained join.
