#!/usr/bin/env python3
"""
Analyze Wikipedia dataset: compute statistics, check quality.

Usage:
    python scripts/analyze_wiki_dataset.py --input datasets/wikipedia/processed/train/bn_train.txt
    python scripts/analyze_wiki_dataset.py --input datasets/wikipedia/processed --all-splits
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze Wikipedia dataset quality and statistics."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.sentences = []
        self.stats = {}

    def load_data(self) -> None:
        """Load dataset from file."""
        logger.info(f"Loading dataset: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(self.sentences)} sentences")

    def compute_basic_stats(self) -> Dict:
        """Compute basic statistics about the dataset."""
        logger.info("Computing basic statistics...")

        total_sentences = len(self.sentences)
        total_chars = sum(len(s) for s in self.sentences)
        total_words = sum(len(s.split()) for s in self.sentences)

        sentence_lengths = [len(s) for s in self.sentences]

        stats = {
            "total_sentences": total_sentences,
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_sentence_length_chars": (
                total_chars / total_sentences if total_sentences > 0 else 0
            ),
            "avg_sentence_length_words": (
                total_words / total_sentences if total_sentences > 0 else 0
            ),
            "min_sentence_length": min(sentence_lengths) if sentence_lengths else 0,
            "max_sentence_length": max(sentence_lengths) if sentence_lengths else 0,
            "median_sentence_length": np.median(sentence_lengths) if sentence_lengths else 0,
            "std_sentence_length": np.std(sentence_lengths) if sentence_lengths else 0,
        }

        self.stats = stats
        return stats

    def compute_vocabulary_stats(self) -> Dict:
        """Compute vocabulary statistics."""
        logger.info("Computing vocabulary statistics...")

        # Collect all words
        all_words = []
        for sentence in self.sentences:
            all_words.extend(sentence.split())

        # Count word frequencies
        word_freq = Counter(all_words)

        vocab_stats = {
            "vocabulary_size": len(word_freq),
            "total_tokens": len(all_words),
            "unique_tokens": len(word_freq),
            "most_common_words": word_freq.most_common(20),
            "hapax_legomena": sum(1 for count in word_freq.values() if count == 1),
        }

        self.stats.update(vocab_stats)
        return vocab_stats

    def check_encoding_issues(self) -> List[str]:
        """Check for encoding issues in the dataset."""
        logger.info("Checking for encoding issues...")

        issues = []

        for i, sentence in enumerate(self.sentences[:1000]):  # Check first 1000
            try:
                # Try to encode/decode
                sentence.encode("utf-8").decode("utf-8")
            except UnicodeError:
                issues.append(f"Line {i}: Encoding issue")

        if issues:
            logger.warning(f"Found {len(issues)} encoding issues")
        else:
            logger.info("No encoding issues found")

        return issues

    def check_anomalies(self) -> Dict:
        """Check for anomalies in the dataset."""
        logger.info("Checking for anomalies...")

        anomalies = {
            "very_short": [],
            "very_long": [],
            "repeated": [],
            "suspicious_chars": [],
        }

        seen_sentences = set()

        for i, sentence in enumerate(self.sentences):
            # Very short sentences
            if len(sentence) < 10:
                anomalies["very_short"].append((i, sentence))

            # Very long sentences
            if len(sentence) > 1000:
                anomalies["very_long"].append((i, sentence[:100] + "..."))

            # Repeated sentences
            if sentence in seen_sentences:
                anomalies["repeated"].append((i, sentence))
            seen_sentences.add(sentence)

            # Suspicious characters (e.g., excessive punctuation)
            if sentence.count("।") > 10 or sentence.count(".") > 10:
                anomalies["suspicious_chars"].append((i, sentence[:100]))

        # Log anomaly counts
        for anomaly_type, items in anomalies.items():
            if items:
                logger.warning(f"Found {len(items)} {anomaly_type} anomalies")

        return anomalies

    def plot_distributions(self, output_dir: Path) -> None:
        """Plot sentence length distributions."""
        logger.info("Plotting distributions...")

        sentence_lengths = [len(s) for s in self.sentences]
        word_counts = [len(s.split()) for s in self.sentences]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Sentence length histogram (characters)
        axes[0, 0].hist(sentence_lengths, bins=50, edgecolor="black")
        axes[0, 0].set_title("Sentence Length Distribution (Characters)")
        axes[0, 0].set_xlabel("Length (chars)")
        axes[0, 0].set_ylabel("Frequency")

        # Word count histogram
        axes[0, 1].hist(word_counts, bins=50, edgecolor="black", color="green")
        axes[0, 1].set_title("Sentence Length Distribution (Words)")
        axes[0, 1].set_xlabel("Length (words)")
        axes[0, 1].set_ylabel("Frequency")

        # Box plot for sentence lengths
        axes[1, 0].boxplot(sentence_lengths)
        axes[1, 0].set_title("Sentence Length Box Plot")
        axes[1, 0].set_ylabel("Length (chars)")

        # Cumulative distribution
        sorted_lengths = np.sort(sentence_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        axes[1, 1].plot(sorted_lengths, cumulative)
        axes[1, 1].set_title("Cumulative Distribution")
        axes[1, 1].set_xlabel("Sentence Length (chars)")
        axes[1, 1].set_ylabel("Cumulative Probability")
        axes[1, 1].grid(True)

        plt.tight_layout()

        output_path = output_dir / f"{self.file_path.stem}_analysis.png"
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved plot: {output_path}")
        plt.close()

    def generate_report(self, output_dir: Path) -> None:
        """Generate comprehensive analysis report."""
        logger.info("Generating analysis report...")

        report = {
            "file": str(self.file_path),
            "basic_stats": {k: v for k, v in self.stats.items() if k not in ["most_common_words"]},
            "most_common_words": self.stats.get("most_common_words", []),
        }

        # Save report as JSON
        report_path = output_dir / f"{self.file_path.stem}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved report: {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print(f"DATASET ANALYSIS: {self.file_path.name}")
        print("=" * 60)
        print(f"Total Sentences: {report['basic_stats']['total_sentences']:,}")
        print(f"Total Words: {report['basic_stats']['total_words']:,}")
        print(f"Vocabulary Size: {report['basic_stats'].get('vocabulary_size', 'N/A'):,}")
        print(
            f"Avg Sentence Length: {report['basic_stats']['avg_sentence_length_words']:.2f} words"
        )
        print(
            f"Avg Sentence Length: {report['basic_stats']['avg_sentence_length_chars']:.2f} chars"
        )
        print("=" * 60 + "\n")


def analyze_dataset(file_path: Path, output_dir: Path) -> None:
    """
    Run complete analysis on a dataset file.

    Args:
        file_path: Path to dataset file
        output_dir: Directory to save analysis results
    """
    analyzer = DatasetAnalyzer(file_path)
    analyzer.load_data()
    analyzer.compute_basic_stats()
    analyzer.compute_vocabulary_stats()
    analyzer.check_encoding_issues()
    analyzer.check_anomalies()

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.plot_distributions(output_dir)
    analyzer.generate_report(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze Wikipedia dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset file or directory")
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/wikipedia/analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--all-splits", action="store_true", help="Analyze all splits (train, val, test)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        if args.all_splits and input_path.is_dir():
            # Analyze all splits
            for split in ["train", "val", "test"]:
                split_dir = input_path / split
                if split_dir.exists():
                    for file_path in split_dir.glob("*.txt"):
                        logger.info(f"Analyzing {file_path}")
                        analyze_dataset(file_path, output_dir / split)
        else:
            # Analyze single file
            analyze_dataset(input_path, output_dir)

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
