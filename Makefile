.PHONY: help install install-dev test lint format clean docs

help:
	@echo "Bilingual Package - Available Commands"
	@echo "======================================"
	@echo "install        - Install package"
	@echo "install-dev    - Install package with dev dependencies"
	@echo "test           - Run tests"
	@echo "test-cov       - Run tests with coverage"
	@echo "lint           - Run linters"
	@echo "format         - Format code"
	@echo "clean          - Clean build artifacts"
	@echo "docs           - Build documentation"
	@echo "example        - Run example usage script"
	@echo "collect-data   - Collect sample data"
	@echo "prepare-data   - Prepare and process data"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=bilingual --cov-report=html --cov-report=term

lint:
	flake8 bilingual/ tests/ scripts/
	mypy bilingual/

format:
	black bilingual/ tests/ scripts/
	isort bilingual/ tests/ scripts/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	@echo "Documentation available at:"
	@echo "  English: docs/en/README.md"
	@echo "  Bangla:  docs/bn/README.md"

example:
	python examples/basic_usage.py

collect-data:
	python scripts/collect_data.py --source sample --output data/raw/

prepare-data:
	python scripts/prepare_data.py --input data/raw/ --output datasets/processed/
