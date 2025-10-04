#!/usr/bin/env python3
"""
Data collection script for bilingual corpus.

This script provides utilities to collect data from various sources:
- Public domain texts
- Wikipedia dumps
- Parallel corpora
- Web scraping (with proper licensing)

Usage:
    python scripts/collect_data.py --source wikipedia --lang bn --output data/raw/
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def collect_wikipedia(lang: str, output_dir: Path, limit: int = None):
    """
    Collect data from Wikipedia.

    Args:
        lang: Language code ('bn' or 'en')
        output_dir: Output directory
        limit: Maximum number of articles (None for all)
    """
    print(f"Collecting Wikipedia data for language: {lang}")
    print("Note: This requires the 'wikipedia' package.")
    print("Install with: pip install wikipedia")
    print()

    try:
        import wikipedia
    except ImportError:
        print("Error: wikipedia package not installed")
        print("Install with: pip install wikipedia")
        sys.exit(1)

    # Set language
    wikipedia.set_lang(lang)

    output_file = output_dir / f"wikipedia_{lang}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # For demo purposes, collect a few sample articles
    # In production, you'd use Wikipedia dumps
    sample_queries = {
        "bn": ["বাংলাদেশ", "ঢাকা", "রবীন্দ্রনাথ ঠাকুর", "বাংলা ভাষা"],
        "en": ["Bangladesh", "Dhaka", "Rabindranath Tagore", "Bengali language"],
    }

    queries = sample_queries.get(lang, [])

    with open(output_file, "w", encoding="utf-8") as f:
        for query in queries:
            try:
                print(f"Fetching: {query}")
                page = wikipedia.page(query)
                f.write(page.content + "\n\n")
            except Exception as e:
                print(f"  Error: {e}")

    print(f"Saved to: {output_file}")


def collect_sample_data(output_dir: Path):
    """
    Create sample bilingual data for testing.

    Args:
        output_dir: Output directory
    """
    print("Creating sample bilingual data...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample Bangla sentences
    bangla_samples = [
        "আমি স্কুলে যাই।",
        "আমি বই পড়তে ভালোবাসি।",
        "আজ আবহাওয়া খুব সুন্দর।",
        "আমার নাম রহিম।",
        "আমি বাংলাদেশে থাকি।",
        "আমি প্রতিদিন সকালে ব্যায়াম করি।",
        "আমার একটি ছোট বোন আছে।",
        "আমি ফুটবল খেলতে পছন্দ করি।",
        "আমি গান শুনতে ভালোবাসি।",
        "আমি প্রতিদিন স্কুলে যাই।",
    ]

    # Sample English sentences
    english_samples = [
        "I go to school.",
        "I love to read books.",
        "The weather is very nice today.",
        "My name is Rahim.",
        "I live in Bangladesh.",
        "I exercise every morning.",
        "I have a younger sister.",
        "I like to play football.",
        "I love to listen to music.",
        "I go to school every day.",
    ]

    # Sample parallel corpus
    parallel_samples = [
        {"bn": "আমি স্কুলে যাই।", "en": "I go to school."},
        {"bn": "আমি বই পড়তে ভালোবাসি।", "en": "I love to read books."},
        {"bn": "আজ আবহাওয়া খুব সুন্দর।", "en": "The weather is very nice today."},
        {"bn": "আমার নাম রহিম।", "en": "My name is Rahim."},
        {"bn": "আমি বাংলাদেশে থাকি।", "en": "I live in Bangladesh."},
    ]

    # Save Bangla samples
    with open(output_dir / "sample_bn.txt", "w", encoding="utf-8") as f:
        for sentence in bangla_samples:
            f.write(sentence + "\n")

    # Save English samples
    with open(output_dir / "sample_en.txt", "w", encoding="utf-8") as f:
        for sentence in english_samples:
            f.write(sentence + "\n")

    # Save parallel corpus
    with open(output_dir / "parallel_corpus.jsonl", "w", encoding="utf-8") as f:
        for pair in parallel_samples:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved sample data to {output_dir}/")
    print("  - sample_bn.txt")
    print("  - sample_en.txt")
    print("  - parallel_corpus.jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Collect bilingual corpus data from various sources"
    )

    parser.add_argument(
        "--source", choices=["wikipedia", "sample"], default="sample", help="Data source"
    )

    parser.add_argument(
        "--lang", choices=["bn", "en", "both"], default="both", help="Language to collect"
    )

    parser.add_argument("--output", default="data/raw", help="Output directory")

    parser.add_argument("--limit", type=int, help="Maximum number of items to collect")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "sample":
        collect_sample_data(output_dir)

    elif args.source == "wikipedia":
        if args.lang == "both":
            collect_wikipedia("bn", output_dir, args.limit)
            collect_wikipedia("en", output_dir, args.limit)
        else:
            collect_wikipedia(args.lang, output_dir, args.limit)

    print("\nData collection complete!")


if __name__ == "__main__":
    main()
