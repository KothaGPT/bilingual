# Quick Start: Scaffold Implementation

This guide helps you get started with the newly implemented scaffold structure.

## 🚀 What Was Implemented

The complete scaffold structure for literary and semantic modules, including:

- **8 new module files** with stub implementations
- **5 training/evaluation scripts** ready for implementation
- **3 new test files** with comprehensive test cases
- **1 CI/CD workflow** for automated testing
- **Updated module exports** for easy imports

## 📁 Directory Structure

```
bilingual/
├── datasets/
│   ├── literary/          # Curated poetry, novels
│   ├── semantic/          # Wikipedia or bilingual corpora (NEW)
│   └── wikipedia/
├── src/bilingual/modules/
│   ├── literary_lm.py              # NEW: Literary LM
│   ├── style_transfer_gpt.py       # NEW: GPT-based style transfer
│   ├── metaphor_simile_detector.py # NEW: Figurative language
│   ├── sentiment_tone_classifier.py # NEW: Sentiment analysis
│   ├── cross_lingual_embed.py      # NEW: Cross-lingual embeddings
│   ├── named_entity_recognizer.py  # NEW: NER for Bangla
│   └── text_complexity_predictor.py # NEW: Readability analysis
├── scripts/
│   ├── train_literary_lm.py        # NEW
│   ├── evaluate_literary.py        # NEW
│   ├── train_style_transfer.py     # NEW
│   ├── evaluate_semantic.py        # NEW
│   └── preprocess_literary.py      # NEW
├── tests/
│   ├── literary/
│   │   └── test_style_transfer_gpt.py  # NEW
│   └── semantic/
│       ├── test_cross_lingual_embed.py # NEW
│       └── test_text_complexity.py     # NEW
└── .github/workflows/
    └── test_models.yml             # NEW: CI workflow
```

## 🔧 Setup

### 1. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install minimal dependencies for testing
pip install pytest pytest-cov
```

### 2. Verify Installation

```bash
# Check if tests can be discovered
pytest tests/literary tests/semantic --collect-only

# Run a simple import test (requires torch)
python3 -c "from bilingual.modules import LiteraryLM; print('Success!')"
```

## 📝 Usage Examples

### Literary LM

```python
from bilingual.modules import LiteraryLM

# Initialize and load model
lm = LiteraryLM("models/bilingual-lm")
lm.load_model()

# Generate literary text
text = lm.generate_text("বাংলা কবিতা", max_length=100)
print(text)
```

### Style Transfer

```python
from bilingual.modules import StyleTransferGPT

# Initialize model
model = StyleTransferGPT("models/style-transfer")
model.load_model()

# Convert to different styles
formal = model.convert("আমি ভাত খাই", "formal")
poetic = model.convert("চাঁদ উঠেছে", "poetic")
```

### Cross-lingual Embeddings

```python
from bilingual.modules import embed_text, compute_similarity

# Generate embeddings
emb_bn = embed_text("আমি ভাত খাই", lang="bn")
emb_en = embed_text("I eat rice", lang="en")

# Compute similarity
similarity = compute_similarity(
    "আমি ভাত খাই",
    "I eat rice",
    lang1="bn",
    lang2="en"
)
```

### Named Entity Recognition

```python
from bilingual.modules import extract_entities, extract_entities_by_type

# Extract all entities
entities = extract_entities("রবীন্দ্রনাথ কলকাতায় থাকতেন")

# Extract specific types
persons = extract_entities_by_type(text, "PERSON")
locations = extract_entities_by_type(text, "LOCATION")
```

### Text Complexity

```python
from bilingual.modules import predict_complexity, classify_difficulty

# Predict complexity score
score = predict_complexity("বাংলা সাহিত্যের ইতিহাস")

# Classify difficulty level
level = classify_difficulty(text)  # beginner/intermediate/advanced/expert
```

## 🧪 Running Tests

### Run All Tests

```bash
# Run all literary and semantic tests
pytest tests/literary tests/semantic -v

# With coverage report
pytest tests/literary tests/semantic -v --cov=src/bilingual --cov-report=term
```

### Run Specific Tests

```bash
# Test literary LM
pytest tests/literary/test_literary_lm.py -v

# Test style transfer
pytest tests/literary/test_style_transfer_gpt.py -v

# Test cross-lingual embeddings
pytest tests/semantic/test_cross_lingual_embed.py -v

# Test text complexity
pytest tests/semantic/test_text_complexity.py -v
```

## 🔨 Training Scripts

### Train Literary LM

```bash
python scripts/train_literary_lm.py \
    --dataset_path datasets/literary/corpus.txt \
    --model_path models/bilingual-lm \
    --epochs 3 \
    --batch_size 8
```

### Train Style Transfer

```bash
python scripts/train_style_transfer.py \
    --dataset_path datasets/literary/style_pairs.json \
    --model_path models/style-transfer-gpt \
    --source_style formal \
    --target_style informal
```

### Preprocess Data

```bash
python scripts/preprocess_literary.py \
    --input_path datasets/literary/raw/ \
    --output_path datasets/literary/processed/ \
    --clean
```

## 📊 Evaluation Scripts

### Evaluate Literary Models

```bash
python scripts/evaluate_literary.py \
    --model_path models/bilingual-lm \
    --test_dataset datasets/literary/test.json \
    --output_path results/literary_eval.json
```

### Evaluate Semantic Models

```bash
python scripts/evaluate_semantic.py \
    --test_dataset datasets/semantic/test.json \
    --output_path results/semantic_eval.json
```

## 🔄 CI/CD Workflow

The CI workflow (`.github/workflows/test_models.yml`) automatically:

1. Tests on Python 3.10 and 3.11
2. Runs literary tests
3. Runs semantic tests
4. Generates coverage reports
5. Uploads to Codecov
6. Archives HTML coverage reports

Triggered on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

## 📋 Next Steps

### Immediate (PR-2)
1. Implement `LiteraryLM.load_model()` with actual transformers model
2. Implement `LiteraryLM.generate_text()` with proper generation
3. Connect with training script
4. Add integration tests

### Short-term (PR-3 & PR-4)
1. Implement style transfer with GPT
2. Implement metaphor/simile detection
3. Implement sentiment classification
4. Train initial models

### Medium-term (PR-5 & PR-6)
1. Implement cross-lingual embeddings
2. Implement NER
3. Implement complexity prediction
4. Curate datasets

### Long-term (PR-7 & PR-8)
1. Complete training/evaluation scripts
2. Benchmark models
3. Prepare for Hugging Face Hub
4. Deploy demos

## 📚 Documentation

- **[Implementation Summary](SCAFFOLD_IMPLEMENTATION.md)** - Detailed implementation notes
- **[Module Reference](docs/MODULE_REFERENCE.md)** - API documentation
- **[Wikipedia Workflow](docs/WIKIPEDIA_WORKFLOW.md)** - Existing workflow docs

## ⚠️ Important Notes

1. **Stub Implementations**: All modules currently have placeholder implementations with TODO markers
2. **Dependencies**: Requires torch, transformers, and other ML libraries (see requirements.txt)
3. **Testing**: Tests verify structure and types, not actual functionality yet
4. **Incremental Development**: Follow the PR plan for systematic implementation

## 🤝 Contributing

When implementing actual functionality:

1. Keep the existing function signatures
2. Update TODO markers as you implement
3. Add proper error handling
4. Update tests with real assertions
5. Add integration tests
6. Update documentation

## 📞 Support

For questions or issues:
- Check the [Implementation Summary](SCAFFOLD_IMPLEMENTATION.md)
- Review the [Module Reference](docs/MODULE_REFERENCE.md)
- Look at existing implementations in `src/bilingual/modules/`

---

**Status**: ✅ Scaffold Complete - Ready for Implementation

**Last Updated**: October 23, 2025
