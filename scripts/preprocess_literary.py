#!/usr/bin/env python3
"""
Preprocessing script for Literary and general text datasets.

This script handles preprocessing of text files (e.g., from literary works,
crawled data, etc.). It recursively finds all .txt files in a given
input directory, cleans them, tokenizes them into sentences, normalizes
the text, and saves the result to a single output file.

It uses the same robust normalization from `preprocess_wiki.py` for consistency.

Usage:
    python scripts/preprocess_literary.py \
        --input-dir data/raw/literary/ \
        --output-file data/processed/literary_corpus.txt \
        --lang bn
"""

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import List

try:
    from indic_nlp.normalize import indic_normalize
    from indic_nlp.tokenize import sentence_tokenize
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False
    logging.warning("indic-nlp-library not available. Using basic tokenization for Bangla.")

# Add parent directory to path to allow project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and normalize text from general corpora."""

    def __init__(self, language: str = "bn"):
        self.language = language
        self.normalizer = None

        if INDIC_NLP_AVAILABLE and self.language == "bn":
            try:
                self.normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer(language)
                logger.info("Initialized IndicNormalizer for Bangla.")
            except Exception as e:
                logger.error(f"Failed to initialize IndicNormalizer: {e}")
                self.normalizer = None
        else:
            logger.warning("Indic NLP library not used for normalization.")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and other artifacts.
        """
        # Replace multiple whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace from each line
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return text.strip()

    def normalize_text(self, text: str) -> str:
        """
        Normalize text (Unicode normalization and Indic normalization).
        """
        text = unicodedata.normalize("NFC", text)
        if self.normalizer:
            text = self.normalizer.normalize(text)
        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        """
        if INDIC_NLP_AVAILABLE and self.language == "bn":
            sentences = sentence_tokenize.sentence_split(text, lang=self.language)
        else:
            # Fallback for English or if Indic NLP is not available
            text = re.sub(r'([.!?])\s*', r'\1\n', text)
            sentences = text.splitlines()

        return [s.strip() for s in sentences if s.strip()]

    def filter_sentence(self, sentence: str, min_len: int, max_len: int) -> bool:
        """
        Filter out sentences that are too short, too long, or contain invalid characters.
        """
        return min_len <= len(sentence) <= max_len


def preprocess_literary_dataset(
    input_dir: Path,
    output_file: Path,
    lang: str,
    min_length: int,
    max_length: int
):
    """
    Preprocess all .txt files in a directory and save to a single output file.
    """
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)

    cleaner = TextCleaner(language=lang)
    all_sentences = []
    
    logger.info(f"Scanning for .txt files in: {input_dir}")
    text_files = list(input_dir.rglob("*.txt"))
    if not text_files:
        logger.warning(f"No .txt files found in {input_dir}. Nothing to process.")
        return

    logger.info(f"Found {len(text_files)} text file(s) to process.")

    for file_path in text_files:
        logger.info(f"Processing {file_path}...")
        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()

        text = cleaner.clean_text(text)
        text = cleaner.normalize_text(text)
        sentences = cleaner.tokenize_sentences(text)
        
        filtered_sentences = [
            s for s in sentences if cleaner.filter_sentence(s, min_length, max_length)
        ]
        all_sentences.extend(filtered_sentences)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(sentence + "\n")

    logger.info(f"Total sentences processed: {len(all_sentences)}")
    logger.info(f"Saved preprocessed data to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess Literary and General Text Datasets")
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the directory containing raw .txt files."
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="Path to save the single preprocessed corpus file."
    )
    parser.add_argument(
        "--lang", type=str, default="bn", choices=["bn", "en"], help="Language of the text ('bn' or 'en')."
    )
    parser.add_argument("--min-length", type=int, default=10, help="Minimum sentence length to keep.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sentence length to keep.")
    
    args = parser.parse_args()

    preprocess_literary_dataset(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output_file),
        lang=args.lang,
        min_length=args.min_length,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
