# Contributing to PARAKH

Thanks for your interest. This document covers the local workflow and the
conventions the project expects.

## Development setup

```bash
pip install -e ".[dev]"
cp .env.example .env
make train      # produces artifacts/health_model.joblib
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Before you open a pull request

Run the same checks CI runs:

```bash
make lint       # ruff check src tests
make test       # pytest with coverage (fails under 90%)
cd frontend && npx tsc -b --noEmit
```

- Tests live in `tests/` and use `pytest`. New behaviour needs a test; the
  coverage gate is enforced in `pyproject.toml`.
- Keep the coverage gate green. If you add a module, add tests for it rather than
  lowering the threshold.

## Conventions

- Code is professional and self-explanatory: no filler comments, no emoji.
  Comment only to explain non-obvious intent (why, not what).
- Formatting and linting are handled by `ruff` with a 100-character line length
  (see `pyproject.toml`). Run `ruff check --fix` before committing.
- Do not commit secrets or a populated `.env`. Add new configuration keys to
  `.env.example` with empty defaults.
- Do not change the API port (8092) or break the default open CORS / no-auth
  behaviour the live demo depends on; tighten only via env opt-in.
- Do not commit trained artifacts other than the tracked demo model, generated
  datasets, or `frontend/dist`.

## Changes to the model or scoring

- Do not silently change reported numbers. If a change moves the ablation ladder
  or headline metrics, regenerate `artifacts/ablation.json` via
  `src/parakh/eval/ablation.py` and update `docs/model_card.md` and the README
  in the same pull request.
- Preserve the documented score-mapping anchors in `src/parakh/scoring/model.py`
  unless the change is intentional and covered by an updated test.
