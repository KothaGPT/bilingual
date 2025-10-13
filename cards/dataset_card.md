# Bilingual Training Dataset Card

## Dataset Details

**Dataset Name**: Bilingual Educational Corpus
**Version**: 1.0.0
**Data Type**: Text corpus (parallel and monolingual)
**Languages**: Bangla (Bengali) and English
**Size**: ~416 words, ~50 sentences
**Format**: Plain text files

## Description

This dataset contains a small curated collection of bilingual text designed for training tokenizers and language models. It includes children's stories, educational content, and simple conversational text in both Bangla and English.

## Data Collection

**Collection Method**: Manually curated from public domain sources and created content
**Collection Period**: 2024
**Geographic Focus**: Bangladesh and English-speaking regions
**Content Types**:
- Children's stories and fables
- Educational descriptions
- Simple conversations
- Parallel sentences for translation

## Data Structure

### Files
- `bilingual_corpus.txt`: Mixed Bangla and English text for tokenizer training
- `parallel_test.txt`: Parallel sentences for translation evaluation

### Format
- Plain text files with UTF-8 encoding
- One sentence per line
- Parallel format: "Bangla Sentence || English Translation"

## Use Cases

This dataset is intended for:
- Training bilingual tokenizers
- Developing small language models
- Translation system prototyping
- Educational NLP research
- Child-friendly content generation

## Data Quality

**Cleaning Process**:
- Unicode normalization applied
- Basic text cleaning and formatting
- Manual review for appropriateness
- Child-friendly content filtering

**Quality Metrics**:
- All text manually reviewed
- No PII or sensitive information
- Culturally appropriate content
- Educational value prioritized

## Limitations

- **Small Size**: Limited vocabulary coverage
- **Domain Specific**: Primarily educational/child-friendly content
- **Geographic Bias**: Bangladesh/English focus
- **Temporal Limitation**: 2024 collection period only

## Ethical Considerations

**Privacy**: No personal information included
**Content Safety**: Child-appropriate content only
**Cultural Sensitivity**: Reviewed for cultural appropriateness
**Bias Mitigation**: Balanced representation attempted within scope

**Child Safety**: All content designed to be appropriate for children aged 6-12

## Usage Rights

**License**: Apache 2.0 (inherited from project license)
**Attribution**: Credit to Bilingual Project contributors
**Restrictions**: Educational and research use encouraged

## Access and Citation

The dataset is available in the project repository under `data/raw/`.

**Citation**:
```
Bilingual Project Contributors. (2024). Bilingual Educational Corpus (Version 1.0.0) [Dataset].
Available from: https://github.com/YOUR_ORG/bilingual
```

## Maintenance

**Update Schedule**: As needed for model improvements
**Quality Monitoring**: Ongoing manual review
**Feedback Process**: Issues and PRs welcome

## Contact

For dataset-related questions or issues, please contact the Bilingual Project maintainers through the repository issue tracker.

## Version History

- **v1.0.0**: Initial release with basic corpus
