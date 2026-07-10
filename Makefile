.PHONY: install train test serve lint

install:
	pip install -e ".[dev]"

train:
	python -m parakh.pipelines.train

test:
	pytest -q

serve:
	uvicorn parakh.api.app:app --reload --port 8092

lint:
	ruff check src tests
