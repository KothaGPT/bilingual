#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer on bilingual corpus.

Usage:
    python scripts/train_tokenizer.py \
        --input corpus_bn.txt corpus_en.txt \
        --model-prefix bilingual_sp \
        --vocab-size 32000
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer for bilingual corpus"
    )

    parser.add_argument(
        "--input", nargs="+", required=True, help="Input text files (can specify multiple files)"
    )

    parser.add_argument(
        "--model-prefix", default="bilingual_sp", help="Prefix for output model files"
    )

    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")

    parser.add_argument(
        "--model-type", choices=["bpe", "unigram", "char", "word"], default="bpe", help="Model type"
    )

    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage (0.9995 recommended for Bangla+English)",
    )

    parser.add_argument(
        "--output-dir", default="models/tokenizer", help="Output directory for trained model"
    )

    args = parser.parse_args()

    # Validate input files
    for input_file in args.input:
        if not Path(input_file).exists():
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set model prefix with output directory
    model_prefix = str(output_dir / args.model_prefix)

    print(f"Training tokenizer with:")
    print(f"  Input files: {', '.join(args.input)}")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Model type: {args.model_type}")
    print(f"  Character coverage: {args.character_coverage}")
    print(f"  Output: {model_prefix}.model")
    print()

    # Train tokenizer
    train_tokenizer(
        input_files=args.input,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )

    print(f"\nTokenizer trained successfully!")
    print(f"Model saved to: {model_prefix}.model")
    print(f"Vocabulary saved to: {model_prefix}.vocab")


if __name__ == "__main__":
    main()
