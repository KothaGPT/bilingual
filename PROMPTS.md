# 🚀 KothaGPT — Bilingual AI System Prompt Library

A comprehensive, production-grade prompt collection for developing, evaluating, and enhancing the **KothaGPT bilingual (Bangla + English) AI ecosystem**.

## 📋 Table of Contents
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Core Audit Prompts](#-core-audit-prompts)
- [Module-Specific Reviews](#-module-specific-reviews)
- [Benchmarking & Evaluation](#-benchmarking--evaluation)
- [Contributing](#-contributing)
- [Versioning](#-versioning)

---

## 🛠 System Requirements
- Python 3.9+
- CUDA 11.7+ (for GPU acceleration)
- Minimum 16GB RAM (32GB recommended)
- 100GB+ free disk space (for models and datasets)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/khulnasoft-bot/bilingual.git
cd bilingual

# Install dependencies
pip install -r requirements.txt

# Run the benchmark suite
python scripts/run_benchmarks.py --language bn+en
```

## 🔍 Core Audit Prompts

### Master System Prompt

```
You are a **Senior AI/ML Researcher** and **Multilingual NLP Specialist** with expertise in:
- Bangla/English NLP
- Low-resource language modeling
- End-to-end ML system design

TASK: Perform a comprehensive audit of the KothaGPT bilingual AI ecosystem.

REVIEW SCOPE:
1. Project Architecture
2. Models (LLM/ASR/TTS)
3. Datasets & Data Pipeline
4. Training & Optimization
5. Tokenizer & Text Processing
6. Evaluation Framework
7. Deployment Infrastructure
8. Documentation & MLOps

OUTPUT FORMAT:
- Executive Summary (1-2 paragraphs)
- Critical Gaps & Risks (prioritized)
- Model Improvement Plan
- Dataset Enhancement Strategy
- Tokenizer Optimization
- Training Pipeline Improvements
- Evaluation Framework Updates
- 30/60/90-Day Action Plan
```

---

## 🧩 Module-Specific Reviews

### 1. Project Architecture Review

```
ROLE: Senior AI Systems Architect

TASK: Analyze the KothaGPT project structure and architecture

REVIEW:
1. Code organization and modularity
2. Dependency management
3. Configuration system
4. Testing infrastructure
5. CI/CD pipelines
6. Documentation coverage

OUTPUT:
- Architecture diagram (Mermaid/PlantUML)
- Dependency analysis
- Technical debt assessment
- Optimization recommendations
```

### 2. Dataset Review & Gap Analysis

```
ROLE: Multilingual Dataset Engineer

TASK: Evaluate current datasets and identify improvements

ANALYZE:
- Bangla/English parallel corpus quality
- Domain coverage and diversity
- Dialect representation
- Data preprocessing pipelines
- Data versioning and provenance

RECOMMEND:
- Specific datasets to include
- Data augmentation strategies
- Quality improvement steps
- Licensing considerations
```

### 3. Tokenizer Optimization

```
ROLE: NLP Engineer

TASK: Optimize tokenizer for Bangla+English

EVALUATE:
- Subword tokenization efficiency
- Unicode/script handling
- Special token coverage
- Vocabulary distribution

SUGGEST:
- Vocabulary size adjustments
- Custom token additions
- Normalization rules
- Performance optimizations
```

### 4. Model Architecture

```
ROLE: ML Research Scientist

TASK: Review and improve model architectures

COMPARE AGAINST:
- LLaMA 3
- Qwen2
- Gemma 2
- Whisper v3

FOCUS AREAS:
- Attention mechanisms
- Embedding strategies
- Model scaling
- Quantization support
- Inference optimization
```

## 📊 Benchmarking & Evaluation

### Evaluation Framework

```
TASKS TO COVER:
- BLEU, ROUGE, METEOR for translation
- WER, CER for ASR
- MOS for TTS
- Latency/throughput metrics
- Memory usage
- Energy efficiency

REQUIREMENTS:
- Standardized test sets
- Automated scoring
- Regular regression testing
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Include tests and documentation

## 🔄 Versioning

- Follows Semantic Versioning (SemVer)
- Main branch tracks latest stable release
- Development happens in feature branches

## 📄 License

[Your License Here] - See [LICENSE](LICENSE) for details
