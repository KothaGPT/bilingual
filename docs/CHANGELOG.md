# Changelog

All notable changes to the Bilingual project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Wikipedia LM Pipeline (PRs 1-8) ✅ COMPLETE

#### Infrastructure & Dependencies (PR-1)
- Added Wikipedia dump download script (`scripts/download_wiki.py`)
- Added extraction monitoring tools (`scripts/monitor_wiki_extraction.py`)
- Added dataset validation script (`scripts/validate_wiki_dataset.py`)
- Added `Makefile.wiki` with convenient targets for Wikipedia workflow
- Installed `wikiextractor` and `indic-nlp-library` dependencies

#### Post-Processing & Dataset Integration (PR-2)
- Added Wikipedia preprocessing script (`scripts/preprocess_wiki.py`)
  - WikiExtractor integration for article extraction
  - Text cleaning (markup, templates, citations removal)
  - Sentence-level tokenization
  - Unicode and Indic normalization
  - Length-based filtering
  - Train/val/test split (80/10/10)
- Added comprehensive preprocessing tests (`tests/wikipedia/test_preprocess.py`)
  - 15+ test cases covering all preprocessing steps
  - Edge case handling
  - Full pipeline integration tests
- Added HuggingFace dataset preparation script (`scripts/prepare_hf_dataset.py`)

#### Training Script (PR-3)
- Added Wikipedia LM training script (`scripts/train_wiki_lm.py`)
  - Support for both MLM (Masked Language Model) and CLM (Causal Language Model)
  - Configurable training parameters
  - Checkpoint saving and resumption
  - TensorBoard logging
  - Evaluation during training
- Added training tests (`tests/wikipedia/test_train.py`)
  - 12+ test cases for training pipeline
  - Mock training validation
  - Dataset loading and tokenization tests
- Added training configuration serialization

#### Evaluation Script (PR-4)
- Added Wikipedia LM evaluation script (`scripts/evaluate_wiki_lm.py`)
  - Perplexity computation
  - Fill-mask functionality (MLM)
  - Text generation (CLM)
  - Interactive evaluation mode
  - Metrics export to JSON
- Added evaluation tests (`tests/wikipedia/test_evaluate.py`)
  - 18+ test cases for evaluation pipeline
  - Inference testing
  - Metrics validation

#### Wikipedia LM Module Integration (PR-5)
- Added Wikipedia LM module (`src/bilingual/modules/wikipedia_lm.py`)
  - `WikipediaLanguageModel` class for easy model usage
  - Fill-mask API
  - Text generation API
  - Embeddings extraction
  - Sentence embeddings with multiple pooling strategies
  - Semantic similarity computation
  - Next word prediction
  - Convenience functions for common operations
- Added module tests (`tests/wikipedia/test_wikipedia_lm.py`)
  - 25+ test cases for module functionality
  - Bilingual support testing
  - Error handling validation
- Updated module exports in `src/bilingual/modules/__init__.py`

#### CI Integration (PR-6)
- Added Wikipedia test workflow (`.github/workflows/test_wikipedia.yml`)
  - Multi-Python version testing (3.9, 3.10, 3.11)
  - Automated test execution on push/PR
  - Coverage reporting with Codecov
  - Linting (flake8, black, isort, mypy)
  - Integration tests
  - Slow test job for main branch
- Added pytest configuration (`pytest.ini`)
  - Test markers (slow, integration, wikipedia)
  - Coverage configuration
- Added coverage configuration (`.coveragerc`)
  - Source and omit patterns
  - Report formatting

#### Documentation
- Added comprehensive Wikipedia workflow guide (`docs/WIKIPEDIA_WORKFLOW.md`)
- Added PR roadmap (`docs/WIKIPEDIA_PR_ROADMAP.md`)
- Added testing guide (`docs/TESTING_GUIDE.md`)
- Added progress tracker (`datasets/wikipedia/PROGRESS.md`)
- Added step-by-step checklist (`datasets/wikipedia/CHECKLIST.md`)
- Added dataset README (`datasets/wikipedia/README.md`)

#### Makefile Targets
- `make -f Makefile.wiki download-bn` - Download Bangla Wikipedia
- `make -f Makefile.wiki preprocess` - Preprocess Wikipedia dump
- `make -f Makefile.wiki monitor` - Monitor extraction progress
- `make -f Makefile.wiki monitor-watch` - Watch extraction in real-time
- `make -f Makefile.wiki validate` - Validate dataset quality
- `make -f Makefile.wiki prepare-hf` - Prepare HuggingFace dataset
- `make -f Makefile.wiki train` - Train language model
- `make -f Makefile.wiki train-test` - Quick test training
- `make -f Makefile.wiki evaluate` - Evaluate trained model
- `make -f Makefile.wiki interactive` - Interactive model testing

### Changed
- Updated project structure to support Wikipedia LM pipeline
- Enhanced module organization with Wikipedia LM integration

#### Bilingual Extension (PR-7) ✨ NEW
- Added bilingual article alignment script (`scripts/align_bilingual_wiki.py`)
  - Interwiki link extraction
  - Article alignment using interwiki links
  - Parallel corpus creation
  - Support for extracted files
- Added bilingual training support
  - Cross-lingual model training (XLM-R, mBERT)
  - Bilingual dataset loading
  - Multilingual evaluation
- Added bilingual tests (`tests/wikipedia/test_bilingual.py`)
  - 15+ test cases for bilingual functionality
  - Article alignment tests
  - Cross-lingual inference tests
  - Bilingual evaluation tests
- Added Makefile targets for bilingual workflow
  - `make align-bilingual` - Align articles
  - `make train-bilingual` - Train bilingual model
  - `make evaluate-bilingual` - Evaluate bilingual model

#### HuggingFace Publishing (PR-8) ✨ NEW
- Added model preparation script (`scripts/huggingface/prepare_model.py`)
  - Model file validation
  - Training artifact removal
  - Metadata generation
  - Git LFS configuration
- Added model card generation (`scripts/huggingface/generate_model_card.py`)
  - Base model template
  - Literary model template
  - Automatic metadata insertion
  - Usage examples
- Added Hub upload script (`scripts/huggingface/upload_model.py`)
  - Repository creation
  - Automated upload
  - Version management
- Added HuggingFace tests
  - `tests/huggingface/test_prepare_model.py` (15+ tests)
  - `tests/huggingface/test_model_card.py` (15+ tests)
- Added Makefile targets for HuggingFace workflow
  - `make prepare-hf-model` - Prepare model
  - `make generate-model-card` - Generate model card
  - `make upload-hf` - Upload to Hub
  - `make hf-pipeline` - Complete pipeline

### Testing
- Added 90+ comprehensive tests for Wikipedia pipeline
- Achieved 85%+ code coverage across all modules
- All CI checks passing
- Added bilingual and HuggingFace test suites

### Documentation
- Updated all documentation for PRs 7-8
- Added complete implementation guide
- Updated PR roadmap with completion status
- Added HuggingFace publishing guide

## [0.1.0] - Previous Release

### Added
- Initial project setup
- Literary analysis module
- Poetic meter analysis
- Style transfer with GAN
- Basic API structure
- Initial documentation

---

## Release Notes

### Wikipedia LM Pipeline (v0.2.0)

**Status:** All 8 PRs Complete ✅

The Wikipedia LM pipeline provides a complete end-to-end solution for:
1. Downloading Wikipedia dumps
2. Preprocessing and cleaning text
3. Training language models (MLM/CLM)
4. Evaluating model performance
5. Using trained models via Python API
6. Bilingual article alignment and training
7. Publishing models to HuggingFace Hub

**Key Features:**
- ✅ Automated Wikipedia data pipeline
- ✅ Support for both Masked and Causal language models
- ✅ Bilingual support (Bangla ↔ English)
- ✅ HuggingFace Hub integration
- ✅ Comprehensive test coverage (90+ tests)
- ✅ CI/CD integration
- ✅ Easy-to-use Python API
- ✅ Interactive evaluation mode
- ✅ Extensive documentation

**Quick Start:**
```bash
# Basic pipeline
make -f Makefile.wiki download-bn
make -f Makefile.wiki preprocess
make -f Makefile.wiki train
make -f Makefile.wiki evaluate

# Bilingual pipeline
make -f Makefile.wiki align-bilingual
make -f Makefile.wiki train-bilingual

# HuggingFace publishing
make -f Makefile.wiki hf-pipeline
make -f Makefile.wiki upload-hf

# Python API
from bilingual.modules.wikipedia_lm import load_model
model = load_model("models/wikipedia/base")
results = model.fill_mask("আমি [MASK] খাই")
```

**Implementation Complete:**
- ✅ PR-1: Infrastructure & Dependencies
- ✅ PR-2: Post-Processing & Dataset Integration
- ✅ PR-3: Training Script
- ✅ PR-4: Evaluation Script
- ✅ PR-5: Wikipedia LM Module Integration
- ✅ PR-6: CI Integration
- ✅ PR-7: Bilingual Extension
- ✅ PR-8: HuggingFace Publishing

---

[Unreleased]: https://github.com/KothaGPT/bilingual/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/KothaGPT/bilingual/releases/tag/v0.1.0
