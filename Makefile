.PHONY: install train rigor real-validation test serve lint web demo

install:
	pip install -e ".[dev]"

train:
	python -m parakh.pipelines.train

rigor:
	python -m parakh.pipelines.rigor

real-validation:
	python -m parakh.pipelines.real_validation

test:
	pytest -q

serve:
	uvicorn parakh.api.app:app --reload --port 8092

web:
	cd frontend && npm install && npm run build

lint:
	ruff check src tests

# One-command judge demo: build the frontend, then serve it and the API from a
# single process on http://localhost:8092. Requires a trained model artifact
# (run `make train` first if artifacts/health_model.joblib is missing).
demo: web
	uvicorn parakh.api.app:app --host 0.0.0.0 --port 8092
