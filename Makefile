SHELL=/bin/bash
PATH := .venv/bin:$(PATH)
EXTRA?=dev
export TEST?=./tests
VENV_ACTIVATE := source .venv/bin/activate

install:
	@set -e; \
	if [ ! -d .venv ]; then \
		python -m venv .venv; \
	fi; \
	$(VENV_ACTIVATE); \
	python -m pip install uv setuptools wheel; \
	python -m uv pip compile pyproject.toml -o requirements.txt --extra $(EXTRA); \
	python -m uv pip install -r requirements.txt;

lint:
	@set -e; \
	if [ ! -d .venv ]; then \
		python -m venv .venv; \
	fi; \
	$(VENV_ACTIVATE);  \
	python -m ruff format .; \
    python -m ruff check . --fix;

siamese:
	@set -e; \
	if [ ! -d .venv ]; then \
		python -m venv .venv; \
	fi; \
	$(VENV_ACTIVATE); \
	python experiments/siamese_experiment.py +architecture=siamese.yaml;
