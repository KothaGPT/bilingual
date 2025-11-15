# Testing Guide

## Current Status

✅ **Base modules work** - Rule-based literary analysis functional  
✅ **ML modules added** - Code ready, needs trained models  
⚠️ **Tests need pytest** - Install with `pip install -e ".[dev]"`

---

## Quick Verification (No Dependencies)

```bash
# Verify imports work
python3 verify_imports.py

# Test basic functionality
python3 verify_pr1.py
```

**Expected Output**: All checks should pass ✓

---

## Running Tests

### Option 1: With pytest (Recommended)

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ --maxfail=1 --disable-warnings -q

# Run only literary tests
pytest tests/literary/ tests/style_transfer/ -v

# Run with coverage
pytest tests/ --cov=src/bilingual/modules --cov-report=term
```

### Option 2: Without pytest

```bash
# Use Python's unittest
python3 -m unittest discover tests/

# Or run verification scripts
python3 verify_imports.py
python3 verify_pr1.py
```

### Option 3: Use helper script

```bash
# Make executable
chmod +x run_tests.sh

# Run
./run_tests.sh
```

---

## Testing ML Features

The ML classes (`PoeticMeterDetector`, `StyleTransferGPT`) require:
1. torch and transformers installed
2. Trained model weights

### Test ML Import

```python
# Test if ML features available
python3 -c "
from bilingual.modules import _ML_AVAILABLE
print(f'ML Features Available: {_ML_AVAILABLE}')

if _ML_AVAILABLE:
    from bilingual.modules import PoeticMeterDetector, StyleTransferGPT
    print('✓ ML classes can be imported')
else:
    print('⚠ Install torch/transformers for ML features')
"
```

### Test ML Classes (Requires Models)

```python
from bilingual.modules import PoeticMeterDetector

# This will raise ImportError if torch not installed
try:
    detector = PoeticMeterDetector(model_path='path/to/model')
    result = detector.detect("আমার সোনার বাংলা")
    print(result)
except ImportError as e:
    print(f"ML features not available: {e}")
except Exception as e:
    print(f"Model not found or other error: {e}")
```

---

## Hugging Face Publishing Tests

The HF publishing scripts exist but need actual models to test.

### Dry Run (Will Show Errors)

```bash
# This will fail because no model exists at the path
bash scripts/huggingface/publish_all.sh models/test test-model username
```

### Create Test Model First

```python
# Create a dummy model for testing
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = Path("models/test")
model_path.mkdir(parents=True, exist_ok=True)

# Save a small model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

print(f"✓ Test model created at {model_path}")
```

Then run:
```bash
bash scripts/huggingface/publish_all.sh models/test test-model username
```

---

## Common Issues

### 1. "pytest not found"

**Solution**:
```bash
pip install -e ".[dev]"
# or
python3 -m pip install pytest pytest-cov
```

### 2. "torch not found" when importing ML classes

**Solution**:
```bash
pip install -e ".[torch]"
# or
pip install torch transformers
```

### 3. NumPy compatibility warning

**Warning Message**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2
```

**Solution** (if it causes issues):
```bash
pip install "numpy<2"
```

### 4. Import errors

**Solution**:
```bash
# Reinstall in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 5. Tests fail with "module not found"

**Solution**:
```bash
# Install package first
pip install -e .

# Then run tests
pytest tests/
```

---

## Test Coverage Goals

- **Rule-based modules**: 90%+ coverage ✅
- **ML modules**: Integration tests needed 🚧
- **Overall**: 85%+ coverage target

### Check Coverage

```bash
pytest tests/ --cov=src/bilingual/modules --cov-report=html
open htmlcov/index.html
```

---

## CI/CD Testing

### GitHub Actions Workflows

1. **`.github/workflows/ci.yml`** - Main CI (existing)
   - Runs on all Python versions
   - Tests all modules
   - Uploads coverage

2. **`.github/workflows/test_literary.yml`** - Literary CI (new)
   - Runs on Python 3.10/3.11
   - Tests only literary modules
   - Path-triggered

3. **`.github/workflows/huggingface_publish.yml`** - HF Publishing (new)
   - Manual trigger or on model changes
   - Publishes to Hugging Face Hub

### Local CI Simulation

```bash
# Simulate CI environment
docker run -it --rm -v $(pwd):/app -w /app python:3.10 bash

# Inside container
pip install -e ".[dev]"
pytest tests/ -v
```

---

## What Works Now

✅ **Literary Analysis**
```python
from bilingual.modules import metaphor_detector, simile_detector, tone_classifier

metaphors = metaphor_detector("Life is a journey")
similes = simile_detector("She runs like the wind")
tone = tone_classifier("This is wonderful!")
```

✅ **Poetic Meter**
```python
from bilingual.modules import detect_meter

result = detect_meter("আমার সোনার বাংলা", language='bengali')
print(result['pattern'])  # 'payar' or similar
```

✅ **Style Transfer (Rule-Based)**
```python
from bilingual.modules import StyleTransferModel

model = StyleTransferModel()
model.load()
formal = model.convert("I can't do this", target_style='formal')
print(formal)  # "I cannot do this"
```

---

## What Needs Models

🚧 **ML-Based Meter Detection**
```python
# Needs trained model
from bilingual.modules import PoeticMeterDetector
detector = PoeticMeterDetector(model_path='models/meter-detector')
result = detector.detect("poem text")
```

🚧 **GPT-Based Style Transfer**
```python
# Needs trained GPT model
from bilingual.modules import StyleTransferGPT
gpt = StyleTransferGPT(model_path='models/style-gpt')
result = gpt.transfer("text", "formal", "poetic")
```

---

## Next Steps

1. **Immediate**: Run `python3 verify_imports.py` to confirm setup
2. **Short-term**: Install pytest and run full test suite
3. **Medium-term**: Train or download models for ML features
4. **Long-term**: Publish models to Hugging Face Hub

---

## Summary Commands

```bash
# Quick check (no dependencies)
python3 verify_imports.py

# Full test suite (needs pytest)
pip install -e ".[dev]"
pytest tests/ -v

# Check ML availability
python3 -c "from bilingual.modules import _ML_AVAILABLE; print(_ML_AVAILABLE)"

# Run specific test file
pytest tests/literary/test_literary_analysis.py -v
```

---

**Last Updated**: 2024-10-24  
**Status**: Base features ✅ | Tests ready ✅ | ML models needed 🚧
