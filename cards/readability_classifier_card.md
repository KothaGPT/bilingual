---
language:
- bn
- en
license: apache-2.0
tags:
- text-classification
- readability
- bengali
- bangla
- education
- text-complexity
metrics:
- accuracy
- f1
widget:
- text: "এটি একটি সহজ বাক্য।"
  example_title: "Simple Bangla"
- text: "This is a simple sentence."
  example_title: "Simple English"
- text: "আধুনিক কম্পিউটার বিজ্ঞানের জটিল অ্যালগরিদম এবং ডেটা স্ট্রাকচার বিশ্লেষণ।"
  example_title: "Complex Bangla"
---

# Readability Classifier (Bangla-English)

## Model Description

A text classification model that predicts the readability level of Bangla and English text. The model classifies text into difficulty levels suitable for different educational grades.

**Model Type:** Text Classification  
**Languages:** Bangla (bn), English (en)  
**Task:** Readability Assessment  
**License:** Apache 2.0

## Readability Levels

The model classifies text into the following levels:

| Level | Grade Range | Description |
|-------|-------------|-------------|
| **Easy** | 1-4 | Simple vocabulary, short sentences, basic concepts |
| **Medium** | 5-8 | Moderate vocabulary, compound sentences, intermediate concepts |
| **Hard** | 9-12 | Advanced vocabulary, complex sentences, abstract concepts |
| **Very Hard** | College+ | Academic vocabulary, technical terms, sophisticated ideas |

## Intended Uses

### Primary Use Cases
- **Educational Content Assessment**: Evaluate text difficulty for students
- **Content Adaptation**: Adjust text complexity for target audiences
- **Curriculum Development**: Match reading materials to grade levels
- **Accessibility**: Identify texts needing simplification
- **Publishing**: Grade content for appropriate age groups

### Example Applications
- Automated textbook leveling
- Reading comprehension test generation
- Content recommendation systems
- Language learning platforms
- Educational app development

## How to Use

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "KothaGPT/readability-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Classify Bangla text
text = "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get readability level
labels = ["Easy", "Medium", "Hard", "Very Hard"]
predicted_level = labels[predictions.argmax().item()]
confidence = predictions.max().item()

print(f"Readability Level: {predicted_level}")
print(f"Confidence: {confidence:.2%}")
```

### Using Pipeline

```python
from transformers import pipeline

# Create classifier pipeline
classifier = pipeline("text-classification", model=model_name)

# Classify multiple texts
texts = [
    "এটি একটি সহজ বাক্য।",
    "আধুনিক প্রযুক্তির জটিল ব্যবহার সমাজে ব্যাপক পরিবর্তন এনেছে।",
    "This is a simple sentence.",
    "The epistemological implications of quantum mechanics remain contentious."
]

for text in texts:
    result = classifier(text)[0]
    print(f"Text: {text[:50]}...")
    print(f"Level: {result['label']} ({result['score']:.2%})\n")
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "বাংলা ভাষা সুন্দর।",
    "The cat sat on the mat.",
    "জটিল গাণিতিক সমীকরণ সমাধান করা কঠিন।"
]

results = classifier(texts, batch_size=8)
for text, result in zip(texts, results):
    print(f"{text}: {result['label']}")
```

## Training Details

### Training Data

**Sources:**
- Textbooks (grades 1-12)
- Wikipedia articles
- News articles
- Literary texts
- Educational websites

**Data Distribution:**
- Easy: 25% (10,000 samples)
- Medium: 35% (14,000 samples)
- Hard: 30% (12,000 samples)
- Very Hard: 10% (4,000 samples)

**Total Samples:** 40,000 labeled texts

### Training Procedure

**Base Model:** `bert-base-multilingual-cased`  
**Fine-tuning Steps:** 10,000  
**Batch Size:** 16  
**Learning Rate:** 2e-5  
**Warmup Steps:** 500  
**Max Sequence Length:** 512 tokens

### Data Preprocessing
- Text normalization
- Sentence segmentation
- Tokenization with multilingual BERT tokenizer
- Balanced sampling across difficulty levels

## Evaluation

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.3% |
| **Macro F1** | 85.8% |
| **Weighted F1** | 87.1% |

### Per-Class Performance

| Level | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Easy | 0.91 | 0.89 | 0.90 | 500 |
| Medium | 0.85 | 0.88 | 0.86 | 700 |
| Hard | 0.86 | 0.84 | 0.85 | 600 |
| Very Hard | 0.88 | 0.87 | 0.87 | 200 |

### Language-Specific Performance
- **Bangla Accuracy**: 86.5%
- **English Accuracy**: 88.1%

### Confusion Matrix Analysis
- Most confusion between adjacent levels (Medium ↔ Hard)
- High accuracy for extreme levels (Easy, Very Hard)

## Features Used for Classification

The model considers multiple linguistic features:

### Lexical Features
- Vocabulary complexity
- Word length distribution
- Rare word frequency
- Technical term density

### Syntactic Features
- Sentence length
- Clause complexity
- Dependency depth
- Parse tree structure

### Semantic Features
- Concept abstractness
- Domain specificity
- Contextual complexity

## Limitations

### Known Issues
- **Domain Sensitivity**: Performance varies by text domain
- **Length Bias**: Very short texts (<50 words) may be misclassified
- **Code-Switching**: Mixed language texts not well-supported
- **Regional Variations**: Optimized for standard Bangla/English

### Edge Cases
- Poetry and creative writing may be misclassified
- Technical documentation requires domain context
- Conversational text differs from written text norms

### Not Suitable For
- Real-time speech transcripts
- Social media posts with heavy slang
- Highly specialized academic papers without context
- Texts with significant OCR errors

## Ethical Considerations

### Bias and Fairness
- Model trained on formal educational materials
- May not reflect all cultural contexts
- Potential bias towards urban/standard language varieties

### Educational Impact
- Should supplement, not replace, human assessment
- Consider cultural and linguistic diversity in application
- Avoid using as sole criterion for student placement

### Recommended Practices
- Combine with human expert review
- Calibrate for specific educational contexts
- Regular evaluation on diverse text samples
- Transparent communication of limitations to users

## Use Cases by Sector

### Education
- Textbook leveling
- Assignment difficulty assessment
- Reading comprehension test design
- Differentiated instruction planning

### Publishing
- Children's book classification
- Content grading for age appropriateness
- Editorial guidelines enforcement

### Technology
- Content recommendation engines
- Accessibility tools
- Language learning apps
- Automated content adaptation

## Citation

```bibtex
@misc{kothagpt-readability-classifier,
  title={Readability Classifier for Bangla and English Text},
  author={KothaGPT Team},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/KothaGPT/readability-classifier}}
}
```

## Model Card Authors

KothaGPT Team

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/KothaGPT/bilingual).

## Additional Resources

- **GitHub Repository**: https://github.com/KothaGPT/bilingual
- **Documentation**: https://github.com/KothaGPT/bilingual/tree/main/docs
- **Evaluation Results**: https://github.com/KothaGPT/bilingual/tree/main/results
- **Demo**: https://huggingface.co/spaces/KothaGPT/readability-classifier-demo
