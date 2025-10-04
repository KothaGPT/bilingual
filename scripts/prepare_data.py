#!/usr/bin/env python3
"""
Prepare and clean bilingual corpus data.

This script:
1. Loads raw text data
2. Normalizes and cleans text
3. Filters by language and quality
4. Splits into train/val/test sets
5. Saves processed data

Usage:
    python scripts/prepare_data.py \
        --input raw_data/ \
        --output datasets/processed/ \
        --split 0.8 0.1 0.1
"""

import argparse
from pathlib import Path
import sys
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual.normalize import normalize_text, detect_language, contains_bangla
from bilingual.data_utils import BilingualDataset


def clean_and_filter_text(text: str, min_length: int = 10, max_length: int = 1000) -> bool:
    """
    Check if text passes quality filters.
    
    Args:
        text: Input text
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        True if text passes filters
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    if len(text) > max_length:
        return False
    
    # Check for reasonable character distribution
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < len(text) * 0.5:  # At least 50% alphabetic
        return False
    
    return True


def process_file(
    input_file: Path,
    lang: str = None,
    normalize: bool = True,
) -> List[dict]:
    """
    Process a single input file.
    
    Args:
        input_file: Path to input file
        lang: Language code (auto-detect if None)
        normalize: Whether to normalize text
        
    Returns:
        List of processed samples
    """
    print(f"Processing {input_file}...")
    
    samples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line:
                continue
            
            # Detect language if not specified
            detected_lang = lang or detect_language(line)
            
            # Normalize if requested
            if normalize:
                line = normalize_text(line, lang=detected_lang)
            
            # Filter by quality
            if not clean_and_filter_text(line):
                continue
            
            samples.append({
                "text": line,
                "lang": detected_lang,
                "source_file": str(input_file.name),
                "line_num": line_num,
            })
    
    print(f"  Processed {len(samples)} samples from {input_file.name}")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and clean bilingual corpus data"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory or file"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory"
    )
    
    parser.add_argument(
        "--lang",
        choices=["bn", "en", "auto"],
        default="auto",
        help="Language of input data (auto-detect if 'auto')"
    )
    
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios"
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum text length"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 1e-6:
        print("Error: Split ratios must sum to 1.0")
        sys.exit(1)
    
    # Get input files
    input_path = Path(args.input)
    
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = list(input_path.glob("*.txt"))
        if not input_files:
            print(f"Error: No .txt files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input file(s)")
    print()
    
    # Process all files
    all_samples = []
    
    for input_file in input_files:
        lang = None if args.lang == "auto" else args.lang
        samples = process_file(
            input_file,
            lang=lang,
            normalize=not args.no_normalize,
        )
        all_samples.extend(samples)
    
    print()
    print(f"Total samples: {len(all_samples)}")
    
    # Create dataset
    dataset = BilingualDataset(data=all_samples)
    
    # Count by language
    bn_count = sum(1 for s in all_samples if s["lang"] == "bn")
    en_count = sum(1 for s in all_samples if s["lang"] == "en")
    mixed_count = sum(1 for s in all_samples if s["lang"] == "mixed")
    
    print(f"  Bangla: {bn_count}")
    print(f"  English: {en_count}")
    print(f"  Mixed: {mixed_count}")
    print()
    
    # Split dataset
    train_ratio, val_ratio, test_ratio = args.split
    train_data, val_data, test_data = dataset.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    print()
    
    # Save datasets
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_data.save(output_dir / "train.jsonl")
    val_data.save(output_dir / "validation.jsonl")
    test_data.save(output_dir / "test.jsonl")
    
    print(f"Saved datasets to {output_dir}/")
    print("  - train.jsonl")
    print("  - validation.jsonl")
    print("  - test.jsonl")


if __name__ == "__main__":
    main()
