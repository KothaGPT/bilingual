---
language:
- bn
- en
license: apache-2.0
tags:
- bilingual
- bengali
- bangla
- wikipedia
- education
- parallel-corpus
task_categories:
- text-generation
- translation
- fill-mask
size_categories:
- 10K<n<100K
---

# KothaGPT Bilingual Corpus

## Dataset Description

**Dataset Name**: KothaGPT Bilingual Corpus  
**Version**: 1.0.0  
**Languages**: Bangla (Bengali) and English  
**Size**: ~50K sentences, ~1M tokens  
**Format**: JSONL (JSON Lines)  
**License**: Apache 2.0

### Dataset Summary

The KothaGPT Bilingual Corpus is a comprehensive collection of Bangla and English text designed for training bilingual language models, tokenizers, and NLP systems. The corpus includes Wikipedia articles, educational content, literary texts, and parallel translations.

### Supported Tasks

- **Language Modeling**: Train causal or masked language models
- **Machine Translation**: Bangla ↔ English translation
- **Text Classification**: Sentiment, readability, topic classification
- **Named Entity Recognition**: Identify entities in bilingual text
- **Cross-lingual Understanding**: Multilingual embeddings and representations

## Dataset Structure

### Data Instances

Each instance in the dataset contains:

```json
{
  "text": "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।",
  "language": "bn",
  "source": "wikipedia",
  "domain": "geography"
}
```

### Data Fields

- `text` (string): The text content
- `language` (string): Language code ("bn" for Bangla, "en" for English)
- `source` (string): Source of the text (wikipedia, literary, educational)
- `domain` (string): Content domain (optional)

### Data Splits

| Split | Samples | Tokens (approx) |
|-------|---------|----------------|
| Train | 40,000 | 800K |
| Validation | 5,000 | 100K |
| Test | 5,000 | 100K |
| **Total** | **50,000** | **1M** |

## Dataset Creation

### Source Data

#### Wikipedia
- Bangla Wikipedia articles (30,000 sentences)
- English Wikipedia articles (20,000 sentences)
- Topics: History, geography, science, culture

#### Literary Corpus
- Bengali literature and poetry (5,000 sentences)
- Classic and modern works
- Public domain texts

#### Educational Content
- Textbooks and learning materials (10,000 sentences)
- Grades 1-12 content
- Science, mathematics, social studies

#### Parallel Translations
- Bangla-English sentence pairs (5,000 pairs)
- Manual translations and alignments
- Quality-checked by native speakers

### Collection Process

1. **Data Extraction**: Wikipedia dumps, literary archives, educational databases
2. **Cleaning**: Unicode normalization, deduplication, quality filtering
3. **Annotation**: Language tagging, domain classification
4. **Validation**: Manual review by native speakers
5. **Splitting**: Stratified split maintaining domain distribution

### Annotations

**Language Tags**: Automatic detection + manual verification  
**Domain Labels**: Semi-automatic classification  
**Quality Scores**: Human-rated on 1-5 scale  
**Parallel Alignments**: Manual alignment for translation pairs

## Considerations for Using the Data

### Social Impact

This dataset enables:
- Development of Bangla NLP tools and applications
- Improved machine translation for low-resource language
- Educational technology for Bengali-speaking students
- Preservation and digitization of Bengali literature

### Discussion of Biases

**Geographic Bias**: Content primarily from Bangladesh and India  
**Formal Language Bias**: Wikipedia and educational content is formal  
**Domain Bias**: Limited representation of colloquial speech  
**Temporal Bias**: Reflects language use up to 2024

### Other Known Limitations

- **Size**: Relatively small compared to English-only corpora
- **Dialects**: Standard Bangla, limited dialectal variation
- **Code-Switching**: Limited mixed Bangla-English text
- **Domains**: Underrepresentation of technical and scientific text

## Additional Information

### Dataset Curators

KothaGPT Team - A collaborative effort to build open-source Bangla NLP resources.

### Licensing Information

This dataset is licensed under the Apache License 2.0. You are free to:
- Use the dataset for any purpose
- Modify and distribute the dataset
- Use in commercial applications

With the following conditions:
- Provide attribution to KothaGPT
- Include a copy of the license
- State any significant changes made

### Citation Information

```bibtex
@dataset{kothagpt_bilingual_corpus,
  title={KothaGPT Bilingual Corpus: A Bangla-English Dataset for NLP},
  author={KothaGPT Team},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/KothaGPT/bilingual-corpus}}
}
```

### Contributions

Thanks to the following contributors:
- Wikipedia contributors for source content
- Bengali literature community for literary texts
- Educational content creators
- KothaGPT team members for curation and quality control

## How to Use

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("KothaGPT/bilingual-corpus")

# Load specific split
train_data = load_dataset("KothaGPT/bilingual-corpus", split="train")

# Load only Bangla text
bangla_data = dataset.filter(lambda x: x["language"] == "bn")

# Iterate through examples
for example in dataset["train"]:
    print(example["text"])
    print(example["language"])
```

### Dataset Statistics

```python
from datasets import load_dataset

dataset = load_dataset("KothaGPT/bilingual-corpus")

print(f"Total samples: {len(dataset['train'])}")
print(f"Languages: {set(dataset['train']['language'])}")
print(f"Domains: {set(dataset['train']['domain'])}")
```

## Contact

For questions, issues, or contributions:
- **GitHub**: https://github.com/KothaGPT/bilingual
- **Issues**: https://github.com/KothaGPT/bilingual/issues
- **Discussions**: https://huggingface.co/datasets/KothaGPT/bilingual-corpus/discussions
