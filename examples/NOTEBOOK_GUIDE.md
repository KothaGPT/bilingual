# Bilingual Tutorial Notebook Guide

## Overview

The `bilingual_tutorial.ipynb` notebook provides a comprehensive, interactive tutorial for the bilingual NLP toolkit. It covers all major features with executable code examples.

## Prerequisites

```bash
# Install the package
pip install -e ..

# Install Jupyter (if not already installed)
pip install jupyter notebook

# Optional: Install additional dependencies
pip install -e "..[torch,onnx]"
```

## Running the Notebook

### Option 1: Jupyter Notebook (Classic)
```bash
cd examples
jupyter notebook bilingual_tutorial.ipynb
```

### Option 2: JupyterLab
```bash
cd examples
jupyter lab bilingual_tutorial.ipynb
```

### Option 3: VS Code
1. Open VS Code
2. Install the "Jupyter" extension
3. Open `bilingual_tutorial.ipynb`
4. Select Python kernel
5. Run cells interactively

## Notebook Structure

### Section 1: Installation & Setup
- Import all necessary modules
- Verify installation
- Configure environment

**Key Imports:**
```python
from bilingual import bilingual_api as bb
from bilingual.normalize import normalize_text, detect_language
from bilingual.modules.literary_analysis import metaphor_detector, tone_classifier
from bilingual.modules.poetic_meter import detect_meter
from bilingual.modules.style_transfer_gan import StyleTransferModel
```

### Section 2: Text Normalization
Learn how to clean and standardize text in both Bangla and English.

**Example:**
```python
text = "আমি   স্কুলে যাচ্ছি।  "
normalized = bb.normalize_text(text, lang="bn")
```

### Section 3: Language Detection
Automatically identify whether text is Bangla, English, or mixed.

**Use Cases:**
- Preprocessing multilingual datasets
- Routing text to language-specific models
- Content classification

### Section 4: Readability Analysis
Assess text complexity and determine appropriate reading levels.

**Metrics Provided:**
- Reading level (elementary, intermediate, advanced)
- Numerical score
- Age range recommendations

### Section 5: Literary Analysis
Detect literary devices and analyze tone.

**Features:**
- **Metaphor Detection**: Find metaphorical expressions
- **Simile Detection**: Identify comparisons using "like" or "as"
- **Tone Classification**: Determine positive/neutral/negative sentiment

**Example:**
```python
text = "Life is a journey"
metaphors = metaphor_detector(text)
tone = tone_classifier(text)
```

### Section 6: Poetic Meter Detection
Analyze syllable patterns and rhythm in poetry.

**Supports:**
- English syllable counting
- Bengali মাত্রা (matra) counting
- Pattern detection (iambic, payar, etc.)

**Example:**
```python
poem = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate."""
result = detect_meter(poem, language='english')
```

### Section 7: Style Transfer
Convert text between different stylistic registers.

**Available Styles:**
- **Formal**: Professional, academic tone
- **Informal**: Casual, conversational tone
- **Poetic**: Artistic, embellished language

**Example:**
```python
model = StyleTransferModel()
model.load()
formal = model.convert("I can't do this", target_style='formal')
# Output: "I cannot do this"
```

### Section 8: Dataset Operations
Work with bilingual datasets efficiently.

**Operations:**
- Create datasets from lists
- Filter by language or category
- Transform with mapping functions
- Batch processing

**Example:**
```python
dataset = BilingualDataset(data=data)
bn_data = dataset.filter(lambda x: x["lang"] == "bn")
```

### Section 9: Advanced Examples
Combine multiple features for comprehensive text analysis.

**Complete Pipeline:**
1. Language detection
2. Text normalization
3. Readability assessment
4. Literary device detection
5. Tone classification

## Tips for Using the Notebook

### 1. Run Cells in Order
The notebook is designed to be executed sequentially. Each section builds on previous imports and definitions.

### 2. Experiment with Your Own Text
Replace example texts with your own Bangla or English content:

```python
# Try your own text
my_text = "আপনার বাংলা টেক্সট এখানে লিখুন"
result = bb.readability_check(my_text, lang="bn")
```

### 3. Modify Parameters
Experiment with different parameters:

```python
# Try different styles
for style in ['formal', 'informal', 'poetic']:
    result = style_model.convert(text, target_style=style)
    print(f"{style}: {result}")
```

### 4. Combine Features
Create custom analysis pipelines:

```python
def my_analysis(text):
    lang = detect_language(text)
    readability = bb.readability_check(text, lang=lang)
    tone = tone_classifier(text)
    return {'lang': lang, 'readability': readability, 'tone': tone}
```

### 5. Save Results
Export analysis results for later use:

```python
import json

results = analyze_text_complete(text)
with open('analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

## Common Issues & Solutions

### Issue: Module Not Found
**Solution:**
```bash
# Make sure package is installed
pip install -e ..

# Or from the root directory
cd /Users/neopilot/bilingual
pip install -e .
```

### Issue: Kernel Dies
**Solution:**
- Restart kernel: `Kernel > Restart`
- Clear outputs: `Cell > All Output > Clear`
- Check memory usage for large datasets

### Issue: Unicode Display Problems
**Solution:**
```python
# Ensure proper encoding
import sys
print(sys.stdout.encoding)  # Should show 'utf-8'

# Force UTF-8 if needed
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

### Issue: Import Errors for Optional Dependencies
**Solution:**
```bash
# Install torch dependencies
pip install torch transformers

# Install ONNX dependencies
pip install onnx onnxruntime
```

## Extending the Notebook

### Add New Cells
Feel free to add your own cells to:
- Test additional examples
- Create custom functions
- Visualize results
- Compare different approaches

### Create Visualizations
```python
import matplotlib.pyplot as plt

# Example: Visualize tone scores
tones = ['positive', 'neutral', 'negative']
scores = [tone['positive'], tone['neutral'], tone['negative']]

plt.bar(tones, scores)
plt.title('Tone Analysis')
plt.ylabel('Score')
plt.show()
```

### Export as Script
Convert the notebook to a Python script:
```bash
jupyter nbconvert --to script bilingual_tutorial.ipynb
```

## Performance Tips

### 1. Use Batch Processing
For multiple texts, use batch operations:
```python
texts = ["text1", "text2", "text3"]
results = style_model.batch_convert(texts, target_style='formal')
```

### 2. Cache Results
Store intermediate results to avoid recomputation:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_analysis(text):
    return analyze_text_complete(text)
```

### 3. Process Large Datasets
For large datasets, use generators:
```python
def process_large_dataset(dataset):
    for item in dataset:
        yield analyze_text_complete(item['text'])
```

## Next Steps

After completing the notebook:

1. **Explore Documentation**: Read [docs/en/README.md](../docs/en/README.md)
2. **Try Training**: Run `train_language_model.py`
3. **Build Applications**: Use the API in your projects
4. **Contribute**: Add new features or examples

## Resources

- **Main Documentation**: [../docs/en/README.md](../docs/en/README.md)
- **API Reference**: [../docs/api/index.md](../docs/api/index.md)
- **Bengali Guide**: [../docs/bn/README.md](../docs/bn/README.md)
- **GitHub Repository**: https://github.com/kothagpt/bilingual

## Support

Need help?
- Open an issue: https://github.com/kothagpt/bilingual/issues
- Check documentation: https://bilingual.readthedocs.io
- Email: info@khulnasoft.com
