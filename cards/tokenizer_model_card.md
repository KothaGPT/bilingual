# Bilingual Tokenizer Model Card

## Model Details

**Model Name**: Bilingual SentencePiece Tokenizer
**Version**: 1.0.0
**Model Type**: SentencePiece BPE Tokenizer
**Languages**: Bangla (Bengali) and English
**Vocabulary Size**: 500
**Framework**: SentencePiece

## Description

This is a bilingual tokenizer trained on a small corpus of Bangla and English text using Byte Pair Encoding (BPE). It provides unified tokenization for both languages, enabling efficient processing of mixed-language text.

## Intended Use

This tokenizer is designed for:
- Tokenizing bilingual (Bangla-English) text
- Preprocessing text for language models
- Supporting downstream NLP tasks in both languages
- Educational and research purposes

## Training Data

The tokenizer was trained on a small curated corpus of:
- Children's stories in Bangla and English
- Educational content
- Simple conversational text
- Parallel sentences for translation tasks

**Training Corpus Size**: ~416 words
**Language Distribution**: Mixed Bangla and English text

## Performance Characteristics

- **Vocabulary Size**: 500 tokens (optimized for small corpus)
- **Character Coverage**: 99% for both scripts
- **Model Type**: BPE (Byte Pair Encoding)
- **Special Tokens**: `<pad>`, `<unk>`, `<s>`, `</s>`

## Limitations

- Trained on a small corpus, may not handle rare words well
- Limited domain coverage (primarily educational/child-friendly content)
- Not optimized for large-scale production use
- May produce suboptimal tokenization for very long texts

## Ethical Considerations

- Trained on child-friendly, educational content
- No known biases in training data
- Designed for positive, educational use cases
- Includes basic safety considerations for child-appropriate content

## Usage

```python
from bilingual.tokenizer import load_tokenizer

tokenizer = load_tokenizer("bilingual-tokenizer")
tokens = tokenizer.encode("আমি স্কুলে যাই।")
decoded = tokenizer.decode(tokens)
```

## Contact

For questions or issues, please contact the Bilingual Project maintainers.

## License

Apache 2.0 License
