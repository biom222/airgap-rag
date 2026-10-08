.PHONY: install lint format typecheck test run migrate compose-up compose-down

install:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff check --fix .
	python -m ruff format .

typecheck:
	python -m mypy

test:
	python -m pytest

run:
	python -m uvicorn airgap_rag.main:app --reload

migrate:
	python -m alembic upgrade head

compose-up:
	docker compose up --build

compose-down:
	docker compose down

