# Dataset Card: [Dataset Name]

## Dataset Description

### Dataset Summary

[Provide a 2-3 sentence summary of the dataset, its purpose, and key characteristics.]

### Supported Tasks and Leaderboards

- **Tasks**: [List primary tasks this dataset supports, e.g., language modeling, translation, classification]
- **Leaderboards**: [Link to any leaderboards using this dataset, or "None" if not applicable]

### Languages

- **Primary Languages**: Bangla (bn), English (en)
- **Language Codes**: `bn`, `en`, `mixed` (code-switched)
- **Scripts**: Bengali script (U+0980-U+09FF), Latin script

---

## Dataset Structure

### Data Instances

A typical example from the dataset:

```json
{
  "text": "একবার একটি খরগোশ ছিল যে খুব দ্রুত দৌড়াতে পারত।",
  "language": "bn",
  "domain": "story",
  "age_range": "6-8",
  "source": "Traditional Bangla Folk Tale",
  "license": "Public Domain",
  "quality_score": 1.0,
  "word_count": 10,
  "char_count": 52
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The main text content |
| `language` | string | Language code: `bn`, `en`, or `mixed` |
| `domain` | string | Content category (story, education, dialogue, etc.) |
| `age_range` | string | Target age range: `6-8`, `9-10`, `11-12`, `general` |
| `source` | string | Source description and attribution |
| `license` | string | Content license (CC0, CC-BY, Public Domain, etc.) |
| `quality_score` | float | Quality assessment score (0.0-1.0) |
| `word_count` | int | Number of words in text |
| `char_count` | int | Number of characters in text |

**For Parallel Corpora**, additional fields include:

| Field | Type | Description |
|-------|------|-------------|
| `en_text` | string | English text |
| `bn_text` | string | Bangla text |
| `alignment_quality` | float | Translation alignment quality (0.0-1.0) |

### Data Splits

| Split | Size | Percentage |
|-------|------|------------|
| Train | [NUMBER] | [XX]% |
| Validation | [NUMBER] | [XX]% |
| Test | [NUMBER] | [XX]% |
| **Total** | **[NUMBER]** | **100%** |

**Split Criteria**:
- Stratified by domain and language
- No overlap between splits
- Temporal split where applicable (earlier data in train, later in test)

---

## Dataset Creation

### Curation Rationale

[Explain why this dataset was created, what gap it fills, and what makes it valuable.]

### Source Data

#### Initial Data Collection and Normalization

**Sources**:
1. **[Source Type 1]**: [Description, URL/Reference, License]
2. **[Source Type 2]**: [Description, URL/Reference, License]
3. **[Source Type 3]**: [Description, URL/Reference, License]

**Collection Methods**:
- [Method 1: e.g., Web scraping from public domain sources]
- [Method 2: e.g., Manual transcription]
- [Method 3: e.g., API access to public datasets]

**Normalization Process**:
1. Unicode normalization (NFC)
2. Whitespace standardization
3. Punctuation normalization
4. Digit format standardization
5. HTML/markup removal
6. Duplicate removal

#### Who are the source language producers?

[Describe the original authors/creators of the text. E.g., "Traditional folk tale authors", "Wikipedia contributors", "Educational content creators"]

### Annotations

#### Annotation process

[Describe how annotations were added, who added them, and what tools were used.]

**Annotation Team**:
- Size: [NUMBER] annotators
- Background: [Description of annotator qualifications]
- Training: [Description of training process]
- Quality Control: [Description of QA process]

**Annotation Tool**: [Name and description of tool used]

**Guidelines**: See [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md)

#### Who are the annotators?

[Describe the annotators' backgrounds, languages spoken, and relevant expertise.]

### Personal and Sensitive Information

**PII Removal Process**:
1. Automatic detection using regex patterns
2. Named Entity Recognition (NER) for person/location/organization
3. Manual review of flagged content
4. Redaction or generalization of identified PII

**Categories Removed**:
- ❌ Full names (kept common first names only)
- ❌ Email addresses
- ❌ Phone numbers
- ❌ Physical addresses
- ❌ National ID numbers
- ❌ Financial information
- ❌ Medical information
- ❌ URLs with personal information

**Verification**: All samples manually reviewed for PII before inclusion.

---

## Considerations for Using the Data

### Social Impact of Dataset

**Intended Use**:
- Training bilingual language models (Bangla-English)
- Educational technology development
- Child-appropriate content generation
- Cross-lingual NLP research

**Potential Positive Impacts**:
- Improved NLP tools for Bangla language
- Better educational resources for bilingual children
- Increased representation of Bangla in AI systems
- Support for code-switching research

**Potential Risks**:
- Possible biases in training data
- Cultural representation imbalances
- Over-generalization of child-appropriate content
- Limited domain coverage

### Discussion of Biases

**Known Biases**:
1. **Geographic Bias**: [Description, e.g., "Primarily content from Bangladesh/India, limited global Bangla perspective"]
2. **Domain Bias**: [Description, e.g., "Overrepresentation of educational content vs. creative writing"]
3. **Temporal Bias**: [Description, e.g., "Mostly contemporary content, limited historical texts"]
4. **Socioeconomic Bias**: [Description, e.g., "Content may reflect urban, educated perspectives"]

**Mitigation Strategies**:
- Diverse source collection
- Balanced representation across domains
- Cultural sensitivity review
- Continuous monitoring and updates

### Other Known Limitations

1. **Size**: [Description of dataset size limitations]
2. **Domain Coverage**: [Description of domains not well represented]
3. **Language Variants**: [Description of dialects or variations not included]
4. **Quality Variance**: [Description of quality inconsistencies]
5. **Temporal Coverage**: [Description of time period limitations]

---

## Additional Information

### Dataset Curators

**Organization**: [Organization Name]  
**Team Members**: [Names or "See CONTRIBUTORS.md"]  
**Contact**: [Email address]

### Licensing Information

**Dataset License**: [License Name, e.g., CC-BY-4.0, CC0-1.0, etc.]

**License Details**:
- Commercial use: [Permitted/Not Permitted]
- Modification: [Permitted/Not Permitted]
- Attribution required: [Yes/No]
- Share-alike required: [Yes/No]

**Component Licenses**:
- Some content may have different licenses - see `license` field per sample
- All content verified for redistribution rights

### Citation Information

```bibtex
@dataset{dataset_name_year,
  author = {[Author Names]},
  title = {[Dataset Name]},
  year = {[Year]},
  publisher = {[Publisher]},
  version = {[Version]},
  url = {[URL]},
  doi = {[DOI if available]}
}
```

### Contributions

Thanks to the following contributors:
- [Contributor names or reference to CONTRIBUTORS.md]

**How to Contribute**:
See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to this dataset.

---

## Changelog

### Version [X.Y.Z] - [Date]
- [Change description]
- [Change description]

### Version [X.Y.Z] - [Date]
- Initial release
- [NUMBER] samples
- [Description of initial content]

---

## Ethical Considerations

### Child Safety

- All content reviewed for age-appropriateness
- Conservative filtering for potentially sensitive content
- Regular audits for safety compliance
- Reporting mechanism for inappropriate content

### Cultural Sensitivity

- Content reviewed by native speakers of both languages
- Cultural appropriateness verification
- Avoidance of stereotypes and biases
- Respectful representation of both cultures

### Privacy & Consent

- No personal data from social media or private sources
- All content from public domain or properly licensed sources
- PII removal verified before inclusion
- Opt-out mechanism for content removal requests

---

## Usage Example

```python
from bilingual.data_utils import BilingualDataset

# Load dataset
dataset = BilingualDataset(file_path="path/to/dataset.jsonl")

# Access samples
for sample in dataset:
    print(f"Text: {sample['text']}")
    print(f"Language: {sample['language']}")
    print(f"Domain: {sample['domain']}")
    print()

# Filter by language
bn_samples = dataset.filter(lambda x: x["language"] == "bn")

# Split into train/val/test
train, val, test = dataset.train_val_test_split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

---

**Last Updated**: [Date]  
**Version**: [X.Y.Z]  
**Status**: [Production/Beta/Alpha]
