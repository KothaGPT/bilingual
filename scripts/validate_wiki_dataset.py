#!/usr/bin/env python3
"""
Validate Wikipedia dataset quality and integrity.

Usage:
    python scripts/validate_wiki_dataset.py --data datasets/wikipedia/processed
    python scripts/validate_wiki_dataset.py --data datasets/wikipedia/processed --sample 100
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate Wikipedia dataset quality."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.stats = {
            "total_sentences": 0,
            "splits": {},
            "length_distribution": Counter(),
            "issues": [],
        }

    def validate_split(self, split_name: str) -> Dict:
        """Validate a single data split."""
        split_dir = self.data_dir / split_name

        if not split_dir.exists():
            logger.warning(f"Split '{split_name}' not found")
            return {"exists": False}

        stats = {
            "exists": True,
            "files": [],
            "total_lines": 0,
            "total_chars": 0,
            "min_length": float("inf"),
            "max_length": 0,
            "empty_lines": 0,
            "samples": [],
        }

        for txt_file in split_dir.glob("*.txt"):
            file_stats = self._validate_file(txt_file)
            stats["files"].append(file_stats)
            stats["total_lines"] += file_stats["lines"]
            stats["total_chars"] += file_stats["chars"]
            stats["min_length"] = min(stats["min_length"], file_stats["min_length"])
            stats["max_length"] = max(stats["max_length"], file_stats["max_length"])
            stats["empty_lines"] += file_stats["empty_lines"]

        if stats["min_length"] == float("inf"):
            stats["min_length"] = 0

        return stats

    def _validate_file(self, file_path: Path) -> Dict:
        """Validate a single text file."""
        stats = {
            "name": file_path.name,
            "lines": 0,
            "chars": 0,
            "min_length": float("inf"),
            "max_length": 0,
            "empty_lines": 0,
        }

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    stats["empty_lines"] += 1
                    continue

                stats["lines"] += 1
                length = len(line)
                stats["chars"] += length
                stats["min_length"] = min(stats["min_length"], length)
                stats["max_length"] = max(stats["max_length"], length)

        if stats["min_length"] == float("inf"):
            stats["min_length"] = 0

        return stats

    def sample_sentences(self, split_name: str, n: int = 10) -> List[str]:
        """Sample random sentences from a split."""
        split_dir = self.data_dir / split_name

        if not split_dir.exists():
            return []

        all_sentences = []
        for txt_file in split_dir.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                all_sentences.extend([line.strip() for line in f if line.strip()])

        if not all_sentences:
            return []

        return random.sample(all_sentences, min(n, len(all_sentences)))

    def check_data_quality(self) -> Dict:
        """Perform comprehensive quality checks."""
        issues = []

        # Check if splits exist
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                issues.append(f"Missing split: {split}")

        # Validate each split
        split_stats = {}
        for split in ["train", "val", "test"]:
            stats = self.validate_split(split)
            split_stats[split] = stats

            if stats.get("exists"):
                # Check for empty files
                if stats["total_lines"] == 0:
                    issues.append(f"{split}: No sentences found")

                # Check for too many empty lines
                if stats["empty_lines"] > stats["total_lines"] * 0.1:
                    issues.append(f"{split}: High empty line ratio ({stats['empty_lines']} empty)")

                # Check sentence length distribution
                if stats["min_length"] < 5:
                    issues.append(
                        f"{split}: Very short sentences detected (min={stats['min_length']})"
                    )

                if stats["max_length"] > 1000:
                    issues.append(
                        f"{split}: Very long sentences detected (max={stats['max_length']})"
                    )

        # Check split ratios
        total_lines = sum(s.get("total_lines", 0) for s in split_stats.values())
        if total_lines > 0:
            train_ratio = split_stats.get("train", {}).get("total_lines", 0) / total_lines
            val_ratio = split_stats.get("val", {}).get("total_lines", 0) / total_lines
            test_ratio = split_stats.get("test", {}).get("total_lines", 0) / total_lines

            # Expected ratios: 80/10/10
            if train_ratio < 0.7 or train_ratio > 0.9:
                issues.append(f"Train split ratio unusual: {train_ratio:.2%}")

            if val_ratio < 0.05 or val_ratio > 0.15:
                issues.append(f"Val split ratio unusual: {val_ratio:.2%}")

            if test_ratio < 0.05 or test_ratio > 0.15:
                issues.append(f"Test split ratio unusual: {test_ratio:.2%}")

        return {"splits": split_stats, "total_sentences": total_lines, "issues": issues}

    def print_validation_report(self, quality_check: Dict, samples: Dict[str, List[str]] = None):
        """Print comprehensive validation report."""
        print("\n" + "=" * 70)
        print("Wikipedia Dataset Validation Report")
        print("=" * 70)

        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Sentences: {quality_check['total_sentences']:,}")

        # Split statistics
        for split, stats in quality_check["splits"].items():
            if not stats.get("exists"):
                print(f"\n⚠️  {split.upper()}: NOT FOUND")
                continue

            print(f"\n✓ {split.upper()}:")
            print(f"  Lines: {stats['total_lines']:,}")
            print(f"  Characters: {stats['total_chars']:,}")
            print(f"  Avg Length: {stats['total_chars'] / max(stats['total_lines'], 1):.1f} chars")
            print(f"  Min Length: {stats['min_length']} chars")
            print(f"  Max Length: {stats['max_length']} chars")
            print(f"  Empty Lines: {stats['empty_lines']}")

            if stats["total_lines"] > 0:
                ratio = stats["total_lines"] / quality_check["total_sentences"]
                print(f"  Split Ratio: {ratio:.2%}")

        # Issues
        if quality_check["issues"]:
            print(f"\n⚠️  Issues Found ({len(quality_check['issues'])}):")
            for issue in quality_check["issues"]:
                print(f"  - {issue}")
        else:
            print("\n✅ No issues found!")

        # Samples
        if samples:
            print("\n📝 Sample Sentences:")
            for split, sentences in samples.items():
                if sentences:
                    print(f"\n  {split.upper()} (showing {len(sentences)}):")
                    for i, sentence in enumerate(sentences[:5], 1):
                        preview = sentence[:100] + "..." if len(sentence) > 100 else sentence
                        print(f"    {i}. {preview}")

        print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate Wikipedia dataset quality")
    parser.add_argument(
        "--data", type=str, default="datasets/wikipedia/processed", help="Processed data directory"
    )
    parser.add_argument(
        "--sample", type=int, default=10, help="Number of sample sentences to show per split"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    data_dir = Path(args.data)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Validate dataset
    validator = DatasetValidator(data_dir)
    quality_check = validator.check_data_quality()

    # Sample sentences
    samples = {}
    if args.sample > 0:
        for split in ["train", "val", "test"]:
            samples[split] = validator.sample_sentences(split, args.sample)

    # Output results
    if args.json:
        output = {"quality_check": quality_check, "samples": samples}
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        validator.print_validation_report(quality_check, samples)

    # Exit with error if issues found
    if quality_check["issues"]:
        logger.warning(f"Validation completed with {len(quality_check['issues'])} issues")
        exit(1)
    else:
        logger.info("Validation completed successfully!")


if __name__ == "__main__":
    main()
