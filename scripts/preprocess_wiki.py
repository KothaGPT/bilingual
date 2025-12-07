#!/usr/bin/env python3
"""
Preprocess Wikipedia dumps: extract, clean, normalize, and tokenize.

Usage:
    python scripts/preprocess_wiki.py --input datasets/wikipedia/raw/bnwiki-latest-pages-articles.xml.bz2 --output datasets/wikipedia/processed
    python scripts/preprocess_wiki.py --bilingual --align --input datasets/wikipedia/raw --output datasets/wikipedia/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from indic_nlp.normalize import indic_normalize
    from indic_nlp.tokenize import sentence_tokenize

    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False
    logging.warning("indic-nlp-library not available. Using basic tokenization.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WikiTextCleaner:
    """Clean and normalize Wikipedia text."""

    def __init__(self, language: str = "bn"):
        self.language = language
        self.normalizer = None

        if INDIC_NLP_AVAILABLE and language == "bn":
            self.normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer(language)

    def clean_text(self, text: str) -> str:
        """
        Clean Wikipedia text by removing markup, citations, etc.

        Args:
            text: Raw Wikipedia text

        Returns:
            Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove citations like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Iteratively remove nested templates and links
        # Process from inside out to handle nesting
        for _ in range(5):  # Limit iterations to prevent infinite loops
            # Remove templates {{...}}
            text = re.sub(r"\{\{[^\{\}]*\}\}", "", text)
            # Clean wiki links [[link|text]] -> text
            text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]\[]+)\]\]", r"\1", text)

        # Remove file/image references
        text = re.sub(r"\[\[File:.*?\]\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\[Image:.*?\]\]", "", text, flags=re.IGNORECASE)

        # Remove category links
        text = re.sub(r"\[\[Category:.*?\]\]", "", text, flags=re.IGNORECASE)

        # Remove external links
        text = re.sub(r"\[http[^\]]+\]", "", text)

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def normalize_text(self, text: str) -> str:
        """
        Normalize text (Unicode normalization, Indic normalization).

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Indic normalization if available
        if self.normalizer:
            text = self.normalizer.normalize(text)

        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.

        Args:
            text: Text to tokenize

        Returns:
            List of sentences
        """
        if INDIC_NLP_AVAILABLE and self.language == "bn":
            sentences = sentence_tokenize.sentence_split(text, lang=self.language)
        else:
            # Basic sentence tokenization
            sentences = re.split(r"[।\.\!\?]+", text)

        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def filter_sentence(self, sentence: str, min_length: int = 10, max_length: int = 500) -> bool:
        """
        Filter out sentences that are too short or too long.

        Args:
            sentence: Sentence to filter
            min_length: Minimum character length
            max_length: Maximum character length

        Returns:
            True if sentence should be kept
        """
        length = len(sentence)
        return min_length <= length <= max_length


def extract_wikipedia_dump(input_path: Path, output_dir: Path, language: str = "bn") -> Path:
    """
    Extract Wikipedia dump using WikiExtractor.

    Args:
        input_path: Path to compressed Wikipedia dump
        output_dir: Directory to save extracted text
        language: Language code

    Returns:
        Path to extracted text directory
    """
    logger.info(f"Extracting Wikipedia dump: {input_path}")

    extract_dir = output_dir / f"{language}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try using wikiextractor
        cmd = [
            "python3",
            "-m",
            "wikiextractor.WikiExtractor",
            str(input_path),
            "-o",
            str(extract_dir),
            "--json",
            "--processes",
            "4",
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("WikiExtractor not available. Please install: pip install wikiextractor")
        logger.info("Alternative: Use manual extraction or install WikiExtractor")
        raise

    logger.info(f"Extraction complete: {extract_dir}")
    return extract_dir


def process_extracted_files(
    extract_dir: Path,
    output_dir: Path,
    language: str = "bn",
    min_sentence_length: int = 10,
    max_sentence_length: int = 500,
) -> Tuple[Path, Path, Path]:
    """
    Process extracted Wikipedia files: clean, normalize, tokenize.

    Args:
        extract_dir: Directory with extracted Wikipedia text
        output_dir: Directory to save processed text
        language: Language code
        min_sentence_length: Minimum sentence length
        max_sentence_length: Maximum sentence length

    Returns:
        Paths to train, val, test files
    """
    logger.info(f"Processing extracted files from {extract_dir}")

    cleaner = WikiTextCleaner(language)

    # Collect all sentences
    all_sentences = []

    # Process all extracted files
    for file_path in extract_dir.rglob("wiki_*"):
        if not file_path.is_file():
            continue

        logger.info(f"Processing {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # WikiExtractor outputs JSON lines
                    data = json.loads(line)
                    text = data.get("text", "")

                    # Clean and normalize
                    text = cleaner.clean_text(text)
                    text = cleaner.normalize_text(text)

                    # Tokenize into sentences
                    sentences = cleaner.tokenize_sentences(text)

                    # Filter sentences
                    sentences = [
                        s
                        for s in sentences
                        if cleaner.filter_sentence(s, min_sentence_length, max_sentence_length)
                    ]

                    all_sentences.extend(sentences)

                except json.JSONDecodeError:
                    continue

    logger.info(f"Total sentences collected: {len(all_sentences)}")

    # Split into train/val/test (80/10/10)
    total = len(all_sentences)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_sentences = all_sentences[:train_size]
    val_sentences = all_sentences[train_size : train_size + val_size]
    test_sentences = all_sentences[train_size + val_size :]

    logger.info(
        f"Train: {len(train_sentences)}, Val: {len(val_sentences)}, Test: {len(test_sentences)}"
    )

    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train" / f"{language}_train.txt"
    val_path = output_dir / "val" / f"{language}_val.txt"
    test_path = output_dir / "test" / f"{language}_test.txt"

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_sentences))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_sentences))

    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test_sentences))

    logger.info(f"Saved train: {train_path}")
    logger.info(f"Saved val: {val_path}")
    logger.info(f"Saved test: {test_path}")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess Wikipedia dumps")
    parser.add_argument(
        "--input", type=str, required=True, help="Input Wikipedia dump file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/wikipedia/processed",
        help="Output directory for processed text",
    )
    parser.add_argument("--lang", type=str, default="bn", help="Language code")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum sentence length")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum sentence length")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction step (use already extracted files)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        # Extract Wikipedia dump
        if not args.skip_extraction:
            extract_dir = extract_wikipedia_dump(input_path, output_dir, args.lang)
        else:
            extract_dir = input_path

        # Process extracted files
        train_path, val_path, test_path = process_extracted_files(
            extract_dir, output_dir, args.lang, args.min_length, args.max_length
        )

        logger.info("Preprocessing complete!")
        logger.info(f"Train: {train_path}")
        logger.info(f"Val: {val_path}")
        logger.info(f"Test: {test_path}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
