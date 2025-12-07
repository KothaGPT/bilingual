# Production Grade Development Plan

## Overview
This plan outlines the steps to upgrade the current bilingual model codebase to a production-grade system. The focus is on addressing gaps in **Data Sources**, **Data Processing**, and **Modeling**.

## 1. Data Source & Collection (`datasource`)

### Current State
- Basic Wikipedia collection using `wikipedia` API wrapper (slow, rate-limited).
- Hardcoded web scraping logic in `collect_data.py`.
- Limited scalability and error handling.

### TODOs
- [ ] **Implement Wikipedia Dump Processor**
    - [ ] Create a script to download and stream process Wikipedia XML dumps.
    - [ ] Use `mwxml` or similar libraries for efficient parsing without loading everything into memory.
    - [ ] Extract parallel sentences from interlanguage links if possible.
- [ ] **Configurable Web Scraper**
    - [ ] Refactor `collect_web_data` to use a configuration file (YAML/JSON) for sources (URLs, CSS selectors).
    - [ ] Implement `retry` logic and `backoff` strategies for robust scraping.
    - [ ] Add `robots.txt` compliance checking.
- [ ] **Data Versioning & Lineage**
    - [ ] specific raw data with timestamps/versions (e.g., `data/raw/2023-10-27/`).
    - [ ] Log source metadata (URL, timestamp, success/fail status) for every crawled document.

## 2. Data Processing & Storage (`data`)

### Current State
- `BilingualDataset` loads all data into memory (`self.data = []`). **CRITICAL GAP**.
- Data storage format is simple JSONL/TXT.
- Preprocessing is basic regex-based cleaning.

### TODOs
- [ ] **Scalable Data Loading**
    - [ ] Refactor `BilingualDataset` to support **streaming/lazy loading** (e.g., via `IterableDataset`).
    - [ ] Migrate underlying storage from simple JSONL to **Apache Parquet** or **Hugging Face Datasets** (Arrow) for zero-copy memory mapping and compression.
- [ ] **Advanced Preprocessing Pipeline**
    - [ ] Implement a `Pipeline` class for chainable data transformations.
    - [ ] Add **Language Identification** (using `fasttext` or `langid`) to filter out noise (e.g., Chinese characters in a Bangla dataset).
    - [ ] Add **Deduplication** (MinHash LSH or Exact match) to remove duplicate sentences.
    - [ ] Add **Quality Filtering** (Length ratio, perplexity score) to remove bad translations.
- [ ] **Data Validation**
    - [ ] Implement schema validation for data entries (using `Pydantic`).
    - [ ] Add unit tests for tokenizer coverage (ensure vocabulary covers the dataset characters).

## 3. Modeling & Training (`model`)

### Current State
- Custom raw PyTorch `TransformerModel` implementation.
- Custom `BilingualTokenizer` wrapper around SentencePiece.
- Lack of standard experiment tracking and model registry interactions.

### TODOs
- [ ] **Hugging Face Integration**
    - [ ] Create a `HFTransformerModel` wrapper to utilize pre-trained models (e.g., `mBART`, `mT5`) or standard configurations.
    - [ ] Ensure model output is compatible with `AutoModelForSeq2SeqLM`.
- [ ] **Tokenizer Standardization**
    - [ ] Align `BilingualTokenizer` with `PreTrainedTokenizerFast` from `tokenizers` library for speed.
    - [ ] Ensure tokenizer can be saved/loaded via `tokenizer.save_pretrained()`.
- [ ] **Training Loop Enhancements**
    - [ ] Integrate **Experiment Tracking** (WandB, MLflow, or TensorBoard).
    - [ ] Implement **Checkpointing** based on validation metrics (save best model, not just last).
    - [ ] Add **Mixed Precision Training** (AMP) support for faster training.
- [ ] **Model Serving Preparation**
    - [ ] Add export scripts to **ONNX** or **TorchScript**.
    - [ ] Create a "Model Card" generator to automatically document model metrics and parameters.

## 4. Immediate Action Plan (Next Steps)

1.  **Refactor `BilingualDataset`** to stop loading entire files into Python lists.
2.  **Integrate `datasets` library** for efficient data handling.
3.  **Add `fasttext`** for language validation in the collection pipeline.
