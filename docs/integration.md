# Integration guide — plugging PARAKH into live rails

PARAKH scores a fixed feature vector. Everything upstream of that vector — where
the numbers come from — is isolated behind a single seam, the
`ConsentedFeedAdapter`. Moving from the synthetic demo to a bank's production
Account Aggregator, GST and EPFO feeds is therefore an adapter and configuration
change, not a model change. The trained model, calibration, scorecard mapping,
reason codes and API contract are untouched.

## The seam

`src/parakh/adapters/base.py` defines the contract:

```python
class ConsentedFeedAdapter(ABC):
    def map_gst(self, raw) -> FeatureVector: ...
    def map_account_aggregator(self, raw) -> FeatureVector: ...
    def map_epfo(self, raw) -> FeatureVector: ...
    def to_feature_vector(self, raw) -> FeatureVector: ...
```

Each rail is mapped independently, then `to_feature_vector` merges the subsets
and validates that the result covers exactly the model's feature schema with
finite values. A missing, unknown or non-finite feature raises `AdapterError`
rather than letting a silent `NaN` reach the model — the same contract the
`/score` endpoint enforces one layer higher.

The three rails partition the feature schema exactly (see `RAIL_FEATURES`), which
is what makes the source-ablation ladder meaningful: withholding a rail removes
precisely that rail's feature subset.

| Rail | Consent purpose | Feature subset |
|---|---|---|
| GST returns network | 103 (originate) / 104 (monitor) | `gst_*` |
| Account Aggregator (bank statements) | 103 / 104 | `bank_*`, `xf_gst_bank_gap` |
| EPFO (establishment) | 103 / 104 | `epfo_*` |

## Reference adapter

`src/parakh/adapters/persona.py` (`PersonaFeedAdapter`) is the worked example. It
re-shapes a synthetic persona row into a rail-partitioned raw feed whose field
names deliberately do **not** match the model schema (`avgMonthlyTurnover`,
`gstBankReconciliationGap`, `activeHeadcount`, …), then maps that raw feed back
to the feature vector. The round trip exercises the exact translation a live
adapter performs, so a production adapter is a copy of this file with the field
names and any unit conversions swapped for the rail's real payload shape.

## Writing a production adapter

1. Subclass `ConsentedFeedAdapter` (e.g. `SahamatiAAAdapter`).
2. Implement the three `map_*` methods against the rail's real response schema —
   the Sahamati / ReBIT AA FI data schema for bank statements, the GSTN returns
   payload for GST, the EPFO ECR / establishment response for workforce.
3. Do any unit and derived-feature work inside the adapter: `xf_gst_bank_gap` is
   `1 - bank_credits / gst_turnover`, cash-buffer days from balance history,
   filing punctuality from the returns calendar, and so on.
4. Return only that rail's subset; `to_feature_vector` handles merging and
   validation.
5. Register the adapter in the scoring service configuration and point it at the
   live consent artefact.

Because validation lives in the base class, an adapter cannot ship a partial or
`NaN` vector to the model — it fails loud at the seam.

## ULI / OCEN

The Unified Lending Interface and OCEN loan-application flows are transport, not
schema: they deliver the same consented AA / GST / EPFO artefacts over a
standardised rail. Integration is a new adapter (or a thin transport wrapper that
hands the decoded artefact to the existing rail adapters) plus endpoint
configuration — the feature schema and model are unchanged.

## What stays fixed

- Model artefact, calibration and 0–100 score mapping.
- Reason codes, dimensions and pre-qualified-limit logic.
- Consent artefacts (ReBIT purpose 103 / 104).
- The `/score`, `/whatif`, `/monitor` and `/memo` API contract.

Only the adapter and its configuration change per deployment.
