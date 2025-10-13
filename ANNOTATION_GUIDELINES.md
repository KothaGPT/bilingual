# Annotation Guidelines for Bilingual Dataset

## Overview

These guidelines provide standards for annotating and curating bilingual (Bangla-English) text data for NLP tasks, with a focus on child-friendly, educational content.

## Data Collection Principles

### Content Selection
- **Child-Friendly**: All content must be appropriate for children aged 6-12
- **Educational Value**: Prioritize content that teaches language, concepts, or cultural knowledge
- **Cultural Sensitivity**: Respect both Bangla and English cultural contexts
- **Language Balance**: Maintain reasonable balance between Bangla and English content

### Quality Standards
- **Accuracy**: Ensure translations are accurate and natural
- **Clarity**: Text should be clear and unambiguous
- **Completeness**: Include complete sentences and coherent passages
- **Diversity**: Include various topics, sentence structures, and vocabulary levels

## Annotation Process

### 1. Language Identification
```json
{
  "text": "আমি স্কুলে যাই।",
  "lang": "bn",
  "confidence": 0.95
}
```

**Guidelines:**
- `bn`: Pure Bangla text
- `en`: Pure English text
- `mixed`: Code-switched or parallel text
- `confidence`: Model confidence score (0.0-1.0)

### 2. Text Normalization
Apply consistent normalization rules:
- Remove extra whitespace
- Standardize punctuation
- Normalize Unicode characters
- Fix common OCR errors

### 3. Content Classification
Categorize content into appropriate classes:
- `story`: Narrative fiction, fables, tales
- `educational`: Learning content, explanations, facts
- `conversational`: Dialogue, daily conversations
- `descriptive`: Descriptions of objects, places, processes

### 4. Readability Assessment
Assess text complexity:
- **Elementary** (6-8 years): Simple vocabulary, short sentences
- **Intermediate** (9-12 years): Moderate vocabulary, compound sentences
- **Advanced** (13+ years): Complex vocabulary, sophisticated structures

### 5. Safety Annotation
Flag content for safety concerns:
- **Safe**: No concerning content
- **Review**: Potentially inappropriate content requiring review
- **Unsafe**: Content that should be excluded

## Quality Control

### Annotation Review Process
1. **Automated Checks**: Run scripts to validate format and consistency
2. **Peer Review**: Cross-validate annotations between annotators
3. **Expert Review**: Subject matter experts review specialized content
4. **Final Validation**: Automated and manual final checks

### Inter-Annotator Agreement
- Target >90% agreement for language identification
- Target >85% agreement for content classification
- Target >80% agreement for readability assessment

## Tools and Scripts

### Required Tools
- Language detection models (Bangla/English)
- Text normalization utilities
- Readability assessment algorithms
- Content classification models

### Validation Scripts
```bash
# Validate dataset format
python scripts/validate_dataset.py --input datasets/processed/

# Check annotation consistency
python scripts/check_agreement.py --annotations annotations/

# Generate quality report
python scripts/quality_report.py --dataset datasets/processed/
```

## Ethical Considerations

### Privacy Protection
- Remove all personally identifiable information (PII)
- Avoid sensitive personal data
- Use synthetic data where appropriate

### Bias Mitigation
- Ensure diverse representation across regions, ages, and backgrounds
- Monitor for cultural, gender, or socioeconomic biases
- Include content from various sources and authors

### Child Safety
- Strict content filtering for age-appropriate material
- No exposure to harmful, violent, or inappropriate themes
- Parental guidance considerations included

## Data Formats

### JSONL Format for Processed Data
```json
{
  "text": "আমি স্কুলে যাই।",
  "lang": "bn",
  "category": "educational",
  "readability_level": "elementary",
  "safety_status": "safe",
  "source": "manual_curation",
  "split": "train",
  "metadata": {
    "word_count": 4,
    "sentence_count": 1,
    "character_count": 24
  }
}
```

### Parallel Corpus Format
```
Bangla Sentence || English Translation || Category || Readability
```

## Maintenance and Updates

### Version Control
- Maintain version history of datasets
- Document all changes and rationale
- Provide migration guides for major changes

### Continuous Improvement
- Regular quality audits
- Incorporation of user feedback
- Updates based on new linguistic research

## Contact and Support

For questions about these guidelines or the annotation process, please contact the Bilingual Project team through the repository issue tracker.
