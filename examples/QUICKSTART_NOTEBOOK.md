# Quick Start: Bilingual Tutorial Notebook

Get started with the bilingual NLP toolkit in 5 minutes!

## 🚀 Quick Setup

```bash
# 1. Navigate to the bilingual directory
cd /Users/neopilot/bilingual

# 2. Install the package
pip install -e .

# 3. Install Jupyter
pip install jupyter

# 4. Launch the notebook
jupyter notebook examples/bilingual_tutorial.ipynb
```

## 📚 What's Inside

The notebook covers **10 major topics**:

1. ✅ **Setup** - Import modules and verify installation
2. 📝 **Text Normalization** - Clean Bangla & English text
3. 🌐 **Language Detection** - Auto-detect Bangla vs English
4. 📊 **Readability** - Assess text complexity
5. 🎭 **Literary Analysis** - Find metaphors, similes, analyze tone
6. 🎵 **Poetic Meter** - Analyze syllables and rhythm
7. 🎨 **Style Transfer** - Convert formal ↔ informal ↔ poetic
8. 📦 **Datasets** - Filter, transform, process data
9. 🔬 **Advanced** - Complete analysis pipelines

## 🎯 Key Features Demonstrated

### Text Normalization
```python
text = "আমি   স্কুলে যাচ্ছি।  "
normalized = bb.normalize_text(text, lang="bn")
# Output: "আমি স্কুলে যাচ্ছি।"
```

### Literary Analysis
```python
# Detect metaphors
metaphors = metaphor_detector("Life is a journey")

# Analyze tone
tone = tone_classifier("This is wonderful!")
# Output: {'positive': 0.8, 'neutral': 0.1, 'negative': 0.1}
```

### Style Transfer
```python
model = StyleTransferModel()
model.load()
formal = model.convert("I can't do this", target_style='formal')
# Output: "I cannot do this"
```

### Poetic Meter
```python
poem = "Shall I compare thee to a summer's day?"
result = detect_meter(poem, language='english')
# Detects: iambic pentameter
```

## 💡 Usage Tips

### Run All Cells
Click `Cell > Run All` to execute the entire notebook at once.

### Try Your Own Text
Replace examples with your own Bangla or English text:
```python
my_text = "আপনার টেক্সট এখানে"
result = bb.readability_check(my_text, lang="bn")
```

### Export Results
Save analysis to file:
```python
import json
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

## 🔧 Troubleshooting

### Module Not Found?
```bash
pip install -e .
```

### Kernel Issues?
Restart: `Kernel > Restart & Clear Output`

### Unicode Problems?
```python
import sys
print(sys.stdout.encoding)  # Should be 'utf-8'
```

## 📖 Learn More

- **Full Guide**: [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)
- **Examples README**: [README.md](README.md)
- **Documentation**: [../docs/en/README.md](../docs/en/README.md)

## 🎓 Next Steps

1. ✅ Complete the tutorial notebook
2. 📝 Run `basic_usage.py` for more examples
3. 🔨 Try `train_language_model.py` for training
4. 🚀 Build your own NLP application!

## 📞 Support

- **Issues**: https://github.com/kothagpt/bilingual/issues
- **Docs**: https://bilingual.readthedocs.io
- **Email**: info@khulnasoft.com

---

**Ready to start?** Open the notebook and run the first cell! 🎉
