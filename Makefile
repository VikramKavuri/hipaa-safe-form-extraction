# Developer entrypoints. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help setup lint format typecheck test check forms eval demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install the package with dev + eval + demo extras
	$(PY) -m pip install -e ".[dev,eval,demo]"

lint: ## Ruff lint
	ruff check src tests eval app

format: ## Ruff auto-format
	ruff format src tests eval app
	ruff check --fix src tests eval app

typecheck: ## mypy on the package
	mypy src

test: ## Run unit tests
	pytest -q

check: lint typecheck test ## Everything CI runs

forms: ## Generate synthetic sample forms + ground truth
	$(PY) -m eval.generate_synthetic_forms --n 12 --seed 7

eval: ## Run the evaluation harness (mock predictor; use PRED=real for the model)
	$(PY) -m eval.run_eval --predictor $(or $(PRED),mock)

demo: ## Launch the Streamlit demo
	streamlit run app/streamlit_app.py

clean: ## Remove caches and run artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache runs outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
