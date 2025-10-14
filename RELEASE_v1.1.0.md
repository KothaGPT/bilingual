# 🚀 Bilingual NLP Toolkit v1.1.0 - Major Release

**Released on October 14, 2025**

Breaking changes and new features

## 🎯 **Release Highlights**

### 🏭 **Production Infrastructure**
- ✅ **FastAPI Server** with monitoring and async processing
- ✅ **Docker Containerization** with multi-stage builds
- ✅ **GitHub Actions CI/CD** pipeline automation
- ✅ **ONNX Model Optimization** for production deployment

### 🎨 **Developer Experience**
- ✅ **Auto-Generated Commit Messages** with emojis
- ✅ **25+ Professional GitHub Labels** for issue management
- ✅ **Rich CLI Interface** built with Typer + Rich
- ✅ **Interactive Documentation** with MkDocs Material

### 📚 **Documentation & Community**
- ✅ **Bilingual Documentation** (English + Bengali)
- ✅ **API Documentation** with live examples
- ✅ **Production Deployment Guide**
- ✅ **Contributing Guidelines** for open source

## 📋 **What's Changed**

### 🚀 **Features**


## 📦 **Installation**

### PyPI Installation
```bash
pip install bilingual==1.1.0
```

### Docker Deployment
```bash
docker run -p 8000:8000 ghcr.io/kothagpt/bilingual:v1.1.0
```

### Development Setup
```bash
git clone https://github.com/kothagpt/bilingual.git
cd bilingual
pip install -e ".[dev]"
```

## 🚀 **Quick Start**

```python
import bilingual as bb

# Language detection
result = bb.detect_language("আমি স্কুলে যাই।")
print(f"Language: {result['language']}")  # Language: bn

# Translation
translation = bb.translate_text("t5-small", "Hello world", "en", "bn")
print(f"Translation: {translation}")

# Text generation
story = bb.generate_text("t5-small", "Once upon a time...")
print(f"Story: {story}")
```

## 📚 **Documentation**

- 🌐 **[Interactive API Docs](https://kothagpt.github.io/bilingual/api/)**
- 📖 **[Full Documentation](https://bilingual.readthedocs.io/)**
- 🐛 **[Issues & Support](https://github.com/kothagpt/bilingual/issues)**
- 💬 **[Discussions](https://github.com/kothagpt/bilingual/discussions)**

## 🤝 **Contributing**

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas where help is needed:
- 📊 **Dataset Collection** - Quality Bangla-English parallel corpora
- 🤖 **Model Training** - Fine-tuning for specific domains
- 📝 **Documentation** - Translation and improvements
- 🧪 **Testing** - Comprehensive test coverage
- 🐛 **Bug Fixes** - Issue resolution and improvements

## 🙏 **Acknowledgments**

Thanks to all contributors who made this release possible!

---

**Built with ❤️ for the Bengali language community worldwide**

*For questions or support: [GitHub Issues](https://github.com/kothagpt/bilingual/issues)*
