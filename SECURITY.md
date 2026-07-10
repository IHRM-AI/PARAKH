# Security Policy

## Scope

PARAKH is a prototype built for IDBI Innovate 2026. It runs on a disclosed
synthetic dataset and has no live connection to Account Aggregator, GST or EPFO
rails. There is no production personal data in this repository.

## Supported versions

The `main` branch is the only supported version during the hackathon.

## Reporting a vulnerability

Report suspected vulnerabilities privately to the maintainers rather than opening
a public issue. Please include:

- a description of the issue and its impact,
- steps to reproduce,
- affected files or endpoints.

We aim to acknowledge reports within a few working days.

## Handling secrets

- Never commit secrets. Configuration lives in `.env`, which is git-ignored;
  `.env.example` documents the available keys with empty values.
- The optional `API_KEY` gate and `CORS_ORIGINS` allowlist default to open,
  demo-friendly behaviour and must be set explicitly to tighten access. Set both
  before exposing the API beyond a trusted network.
- The optional self-hosted GenAI and OCR services (`VLLM_URL`, `OCR_SERVICE_URL`)
  are disabled unless configured.

## Deployment notes

- The API validates scoring inputs and rejects unknown, missing or non-finite
  features with HTTP 422; it does not echo raw payloads in logs.
- Before any deployment handling real, consented data, retrain on the sandboxed
  bank book, set `API_KEY` and a restrictive `CORS_ORIGINS`, and complete the
  fairness and model-risk review noted in `docs/model_card.md`.
