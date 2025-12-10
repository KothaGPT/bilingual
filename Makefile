.PHONY: help install install-dev test lint format clean docs data

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
	@echo ""
	@echo "Data Processing Commands:"
	@echo "======================================"
	@echo "collect-data   - Collect sample data"
	@echo "prepare-data   - Prepare and process data"
	@echo "remove-pii     - Remove PII from data"
	@echo "filter-quality - Filter data by quality"
	@echo "data-workflow  - Run complete data pipeline"
	@echo ""
	@echo "Model Training Commands:"
	@echo "======================================"
	@echo "train-tokenizer    - Train SentencePiece tokenizer"
	@echo "train-lm          - Train language model"
	@echo "train-translation - Train translation model"
	@echo "train-classifier  - Train classification model"
	@echo "evaluate-models   - Evaluate trained models"
	@echo "benchmark-models  - Benchmark model performance"

install:
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/bilingual --cov-report=html --cov-report=term

lint:
	flake8 src/bilingual/ tests/ scripts/
	mypy src/bilingual/

format:
	python3 -m black src/bilingual/ tests/ scripts/
	python3 -m isort src/bilingual/ tests/ scripts/

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
	python3 examples/basic_usage.py

collect-data:
	python3 scripts/collect_data.py --source sample --output data/raw/

prepare-data:
	python3 scripts/prepare_data.py --input data/raw/ --output datasets/processed/

remove-pii:
	python3 scripts/pii_detection.py --input data/raw/ --output data/cleaned/ --mode redact

filter-quality:
	python3 scripts/quality_filter.py --input data/cleaned/ --output data/filtered/ --min-quality 0.7

data-workflow:
	python3 scripts/data_workflow.py --source sample --output datasets/processed/ --dataset-name "Bilingual Corpus"

# Model training and evaluation commands
train-tokenizer:
	python3 scripts/train_tokenizer.py --input datasets/processed/final/train.jsonl datasets/processed/final/val.jsonl datasets/processed/final/test.jsonl --output models/tokenizer/ --vocab-size 1000

train-lm:
	python3 scripts/train_lm.py --train_data datasets/processed/final/train.jsonl --val_data datasets/processed/final/val.jsonl --output_dir models/bilingual-lm/ --model_name_or_path microsoft/DialoGPT-small --no_fp16

train-translation:
	python3 scripts/train_translation.py --data datasets/processed/final/ --output models/translation/

train-classifier:
	python3 scripts/train_classifier.py --task readability --data datasets/processed/final/ --output models/readability-classifier/ --synthetic-labels

evaluate-models:
	python3 scripts/evaluate_models.py --model models/bilingual-lm/ --test-data datasets/processed/final/test.jsonl --task generation --output results/evaluation.json

benchmark-models:
	python3 scripts/benchmark_models.py --models models/bilingual-lm/ --tasks generation --output results/benchmark.json

# PyPI Release Commands:
pypi-build:
	python3 -m build

pypi-test:
	python3 -m twine upload --repository testpypi dist/*

pypi-release:
	python3 -m twine upload dist/*

pypi-install:
	python3 -m pip install bilingual

check-install:
	python3 -c "import bilingual; print(bilingual.__version__)"

# Hugging Face Release Commands:
# ======================================
hf-login:
	@echo "Logging in to Hugging Face Hub..."
	huggingface-cli login

hf-upload-model:
	@echo "Uploading model to Hugging Face Hub..."
	python3 scripts/huggingface/upload_model.py \
		--model models/bilingual-lm/ \
		--repo "KothaGPT/bilingual-lm" \
		--message "Initial model release"

hf-release: hf-login hf-upload-model
	@echo "Hugging Face model upload completed successfully!"