# PARAKH Real-Data Validation

Version 0.1.0 · IDBI Innovate 2026 · Team IHRM

This note answers one objection directly: *"your AUC is only synthetic."* It
runs PARAKH's actual modelling pipeline on a real loan-default dataset and
reports the honest, calibrated numbers. The synthetic engine
(`src/parakh/synth/persona.py`) exists only to simulate the consented
AA / GST / EPFO data surfaces, which are not published at MSME granularity; it
does not prop up the model.

## What was validated

The pipeline under test is identical to the production one:

- LightGBM gradient-boosted trees with per-feature monotone constraints
  (`src/parakh/scoring/model.py`),
- isotonic calibration fitted on a held-out fold with no leakage into the test
  fold,
- evaluated with the same `parakh.eval.metrics.evaluate` used everywhere else.

Only the input columns change: on the real dataset the model consumes the
dataset's own features instead of the synthetic AA / GST / EPFO columns.

## Dataset

SBA 7(a) small-business loans are the canonical real MSME-analog default set and
are the preferred anchor. That file could not be fetched on this network within
the time budget, so this run uses the fallback already on disk:

- **Home-Credit retail loans** — 307k real consumer loans with a real default
  label (`TARGET`). A stratified sample of 60k rows is used to keep training
  tractable.
- This is a **real-default methodology validation on a retail proxy**, not an
  MSME performance claim. It demonstrates that the pipeline produces a real,
  calibrated, out-of-sample AUC on genuine defaults; it does not claim these
  numbers transfer to MSME lending.

The raw file lives under `parakh/data/raw/` and is git-ignored; it is never
committed. Re-run with `python -m parakh.pipelines.real_validation`, which writes
`artifacts/real_validation.json`. If any SBA CSV is placed under `data/raw/sba/`,
the pipeline auto-detects its schema and uses it in preference to the retail proxy.

### Acquiring the SBA anchor

Drop either of these under `parakh/data/raw/sba/` (any filename ending `.csv`);
the loader reads the header and dispatches on the label column automatically
(`_sba_loader_for` in `src/parakh/pipelines/real_validation.py`):

- **Kaggle `SBAnational` dataset** — the widely used "Should This Loan be
  Approved or Denied?" export. Schema: label `MIS_Status` (`CHGOFF` = default = 1,
  `P I F` = paid in full = 0), with columns `Term`, `NoEmp`, `GrAppv`, `SBA_Appv`,
  `NAICS`, `ApprovalFY`. Search Kaggle for "SBA national loan dataset".
- **data.sba.gov 7(a) FOIA CSV** — the official 7(a) loan program FOIA export from
  <https://data.sba.gov> (Loan / Grant data). Schema: label `LoanStatus`
  (`CHGOFF` = 1, `PIF` = 0), with columns `GrossApproval`, `SBAGuaranteedApproval`,
  `TermInMonths`, `JobsSupported`, `BusinessType`, `NAICSCode`,
  `ApprovalFiscalYear`.

Both are placed on disk manually; neither is fetched or committed by the pipeline.
When present, the run reports SBA numbers with an out-of-time split on approval
fiscal year.

## Results

Held-out test fold (stratified random holdout, n_test = 12,000, real default
rate 8.0%):

| Metric                         | Value                 |
|--------------------------------|-----------------------|
| GBM AUC (calibrated)           | 0.735 [0.718, 0.750]  |
| KS                             | 0.372                 |
| Brier                          | 0.069                 |
| Logistic-regression AUC        | 0.732                 |
| GBM lift over logistic         | +0.003                |
| DeLong 95% CI on the lift      | [-0.006, 0.013]       |
| DeLong p-value                 | 0.50                  |

The 95% AUC confidence interval is a 500-sample case bootstrap. The GBM lift over
a plain logistic baseline is small and positive on this retail dataset, and the
DeLong test (paired ROC AUCs on the identical test fold,
`src/parakh/eval/delong.py`) puts a 95% CI of **[-0.006, 0.013]** around it with
p = 0.50 — the **CI includes 0, so the lift is not statistically distinguishable**.
Reported honestly: the deployed value of the GBM here is the calibrated,
monotone-constrained, glass-box score, not an AUC edge over logistic. The additive
value of gradient boosting is expected to be larger on the MSME feature set, which
carries the thresholds and interactions the synthetic generator encodes, but that
is not claimed here.

### Calibration

The isotonic-calibrated probabilities track the observed default rate closely
across risk bins (predicted vs observed, from `reliability_curve` in
`artifacts/real_validation.json`):

| Predicted band (mean) | Observed default rate | Count |
|-----------------------|-----------------------|-------|
| 0.05                  | 0.045                 | 8,901 |
| 0.13                  | 0.149                 | 2,310 |
| 0.24                  | 0.241                 | 402   |
| 0.34                  | 0.303                 | 330   |
| 0.40                  | 0.346                 | 52    |

Predicted probability and realised default rate line up bin by bin, which is the
point: the pipeline emits calibrated probabilities on real defaults, not just a
rank-ordering.

## Honest framing

- The pipeline is validated on **real defaults**: real labels, out-of-sample
  test fold, bootstrap CI, and a calibration curve that holds.
- The synthetic engine only **simulates the consented data surfaces** (AA bank
  statements, GST, EPFO) that no public dataset joins per firm; it is disclosed
  and calibrated to realistic base rates.
- These numbers are a **retail proxy** for methodology, not an MSME performance
  claim. Any production use retrains on the bank's own book in the sandbox before
  the numbers transfer.
