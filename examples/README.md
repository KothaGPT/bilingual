# Bilingual Package Examples

This directory contains examples demonstrating the capabilities of the bilingual NLP toolkit.

## Available Examples

### 1. Basic Usage Script (`basic_usage.py`)

A comprehensive Python script demonstrating core features:
- Text normalization for Bangla and English
- Language detection
- Readability analysis
- Safety checking
- Dataset operations
- Text classification

**Run it:**
```bash
python examples/basic_usage.py
```

### 2. Interactive Tutorial Notebook (`bilingual_tutorial.ipynb`)

A Jupyter notebook providing an interactive walkthrough of all features:

#### Features Covered:
- **Text Normalization**: Clean and standardize text
- **Language Detection**: Automatically identify Bangla vs English
- **Readability Analysis**: Assess text complexity and reading level
- **Literary Analysis**: 
  - Metaphor detection
  - Simile detection
  - Tone classification (positive/neutral/negative)
- **Poetic Meter Detection**: Analyze syllable patterns and rhythm
- **Style Transfer**: Convert between formal, informal, and poetic styles
- **Dataset Operations**: Filter, transform, and process bilingual datasets
- **Advanced Pipelines**: Combine multiple features for comprehensive analysis

**Run it:**
```bash
# Install Jupyter if needed
pip install jupyter

# Launch the notebook
jupyter notebook examples/bilingual_tutorial.ipynb
```

### 3. Training Examples

For model training examples, see:
- `train_language_model.py` - Train custom language models
- `../scripts/train_tokenizer.py` - Train bilingual tokenizers
- `../scripts/prepare_data.py` - Prepare datasets for training

## Quick Start

### Installation

```bash
# Install the package
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Minimal Example

```python
from bilingual import bilingual_api as bb

# Normalize text
text = "আমি   স্কুলে যাচ্ছি।"
normalized = bb.normalize_text(text, lang="bn")
print(normalized)  # "আমি স্কুলে যাচ্ছি।"

# Check readability
result = bb.readability_check(text, lang="bn")
print(result['level'])  # "elementary"
```

## Module-Specific Examples

### Literary Analysis

```python
from bilingual.modules.literary_analysis import metaphor_detector, tone_classifier

# Detect metaphors
text = "Life is a journey"
metaphors = metaphor_detector(text)

# Classify tone
tone = tone_classifier("This is wonderful!")
print(tone)  # {'positive': 0.8, 'neutral': 0.1, 'negative': 0.1}
```

### Poetic Meter

```python
from bilingual.modules.poetic_meter import detect_meter

poem = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate."""

result = detect_meter(poem, language='english')
print(result['pattern'])  # 'iambic'
```

### Style Transfer

```python
from bilingual.modules.style_transfer_gan import StyleTransferModel

model = StyleTransferModel()
model.load()

# Convert to formal style
text = "I can't do this"
formal = model.convert(text, target_style='formal')
print(formal)  # "I cannot do this"
```

## Dataset Examples

### Working with Bilingual Datasets

```python
from bilingual.data_utils import BilingualDataset

# Create dataset
data = [
    {"text": "আমি স্কুলে যাই।", "lang": "bn"},
    {"text": "I go to school.", "lang": "en"},
]

dataset = BilingualDataset(data=data)

# Filter by language
bn_data = dataset.filter(lambda x: x["lang"] == "bn")

# Transform with normalization
normalized = dataset.map(
    lambda x: {**x, "normalized": normalize_text(x["text"])}
)
```

## Advanced Usage

### Complete Text Analysis Pipeline

```python
from bilingual import bilingual_api as bb
from bilingual.normalize import detect_language, normalize_text
from bilingual.modules.literary_analysis import metaphor_detector, tone_classifier

def analyze_text(text: str):
    # Detect language
    lang = detect_language(text)
    
    # Normalize
    normalized = normalize_text(text, lang=lang)
    
    # Analyze readability
    readability = bb.readability_check(text, lang=lang)
    
    # Detect literary devices
    metaphors = metaphor_detector(text)
    
    # Classify tone
    tone = tone_classifier(text)
    
    return {
        'language': lang,
        'normalized': normalized,
        'readability': readability,
        'metaphors': metaphors,
        'tone': tone
    }
```

## Contributing

To add new examples:

1. Create a new Python script or notebook
2. Follow the existing code style
3. Add documentation and comments
4. Update this README with your example
5. Submit a pull request

## Resources

- **Documentation**: [docs/en/README.md](../docs/en/README.md)
- **API Reference**: [docs/api/index.md](../docs/api/index.md)
- **Bangla Docs**: [docs/bn/README.md](../docs/bn/README.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/kothagpt/bilingual/issues
- Documentation: https://bilingual.readthedocs.io
