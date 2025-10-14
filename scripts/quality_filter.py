#!/usr/bin/env python3
"""
Advanced Quality Filtering for Bilingual Corpus.

This script implements comprehensive quality checks for text data:
- Length filtering
- Language identification
- Character set validation
- Content appropriateness
- Duplication detection
- Readability assessment

Usage:
    python scripts/quality_filter.py \
        --input data/raw/ \
        --output data/filtered/ \
        --min-quality 0.7
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual.normalize import detect_language, is_bangla_char  # noqa: E402


class QualityFilter:
    """Filter bilingual corpus by quality criteria."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 5000,
        min_quality_score: float = 0.7,
        check_duplicates: bool = True,
    ):
        """
        Initialize quality filter.

        Args:
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters)
            min_quality_score: Minimum quality score (0.0-1.0)
            check_duplicates: Whether to check for duplicates
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_quality_score = min_quality_score
        self.check_duplicates = check_duplicates
        self.seen_hashes: Set[str] = set()

        # Inappropriate content keywords (basic list)
        self.inappropriate_keywords = {
            "en": [
                "violence",
                "weapon",
                "gun",
                "knife",
                "kill",
                "murder",
                "death",
                "blood",
                "scary",
                "horror",
                "monster",
                "ghost",
                "demon",
                "hate",
                "racism",
                "discrimination",
                "drug",
                "alcohol",
                "beer",
                "wine",
                "drunk",
            ],
            "bn": [
                "সহিংসতা",
                "অস্ত্র",
                "বন্দুক",
                "ছুরি",
                "হত্যা",
                "মারা",
                "মৃত্যু",
                "রক্ত",
                "ভয়ানক",
                "ভূত",
                "প্রেত",
                "দানব",
                "ঘৃণা",
                "বৈষম্য",
                "মাদক",
                "মদ",
                "মাতাল",
            ],
        }

    def check_length(self, text: str) -> Tuple[bool, float, str]:
        """
        Check if text length is within acceptable range.

        Args:
            text: Input text

        Returns:
            Tuple of (pass, score, reason)
        """
        length = len(text.strip())

        if length < self.min_length:
            return False, 0.0, f"Too short ({length} < {self.min_length})"
        elif length > self.max_length:
            return False, 0.0, f"Too long ({length} > {self.max_length})"
        else:
            # Score based on ideal length (500-2000 chars)
            if 500 <= length <= 2000:
                score = 1.0
            elif length < 500:
                score = length / 500
            else:
                score = max(0.5, 1.0 - (length - 2000) / (self.max_length - 2000) * 0.5)
            return True, score, "Length OK"

    def check_character_distribution(self, text: str, lang: str) -> Tuple[bool, float, str]:
        """
        Check character distribution and validity.

        Args:
            text: Input text
            lang: Expected language ('bn', 'en', 'mixed')

        Returns:
            Tuple of (pass, score, reason)
        """
        if not text:
            return False, 0.0, "Empty text"

        # Count character types
        alpha_chars = sum(c.isalpha() for c in text)
        digit_chars = sum(c.isdigit() for c in text)
        space_chars = sum(c.isspace() for c in text)
        punct_chars = sum(c in ".,!?;:।" for c in text)
        total_chars = len(text)

        # Check minimum alphabetic content
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        if alpha_ratio < 0.4:  # At least 40% alphabetic
            return False, 0.0, f"Too few alphabetic characters ({alpha_ratio:.1%})"

        # Check for excessive repetition
        if self._has_excessive_repetition(text):
            return False, 0.0, "Excessive character repetition"

        # Language-specific checks
        if lang == "bn":
            bn_chars = sum(1 for c in text if is_bangla_char(c))
            bn_ratio = bn_chars / alpha_chars if alpha_chars > 0 else 0
            if bn_ratio < 0.5:  # At least 50% Bangla for 'bn' labeled text
                return False, 0.5, f"Insufficient Bangla characters ({bn_ratio:.1%})"

        elif lang == "en":
            latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
            latin_ratio = latin_chars / alpha_chars if alpha_chars > 0 else 0
            if latin_ratio < 0.7:  # At least 70% Latin for 'en' labeled text
                return False, 0.5, f"Insufficient English characters ({latin_ratio:.1%})"

        # Score based on character balance
        score = min(1.0, alpha_ratio + 0.3)  # Bonus for more alphabetic content

        return True, score, "Character distribution OK"

    def check_language_consistency(self, text: str, expected_lang: str) -> Tuple[bool, float, str]:
        """
        Check if detected language matches expected language.

        Args:
            text: Input text
            expected_lang: Expected language code

        Returns:
            Tuple of (pass, score, reason)
        """
        detected_lang = detect_language(text)

        if expected_lang == "mixed":
            # For mixed content, any detection is acceptable
            return True, 1.0, f"Mixed content (detected: {detected_lang})"

        if detected_lang == expected_lang:
            return True, 1.0, "Language matches"
        elif detected_lang == "mixed" and expected_lang in ["bn", "en"]:
            return True, 0.8, f"Mixed content with {expected_lang}"
        else:
            return False, 0.3, f"Language mismatch (expected {expected_lang}, got {detected_lang})"

    def check_content_appropriateness(self, text: str, lang: str) -> Tuple[bool, float, str]:
        """
        Check if content is appropriate for children.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Tuple of (pass, score, reason)
        """
        text_lower = text.lower()

        # Check for inappropriate keywords
        keywords = self.inappropriate_keywords.get(lang, [])
        found_keywords = []

        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)

        if found_keywords:
            return False, 0.0, f"Inappropriate content: {', '.join(found_keywords[:3])}"

        # Check for excessive capitalization (shouting)
        if lang == "en":
            words = text.split()
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            if len(words) > 0 and caps_words / len(words) > 0.3:
                return False, 0.5, "Excessive capitalization"

        return True, 1.0, "Content appropriate"

    def check_sentence_structure(self, text: str, lang: str) -> Tuple[bool, float, str]:
        """
        Check basic sentence structure quality.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Tuple of (pass, score, reason)
        """
        # Split into sentences
        if lang == "bn":
            sentences = re.split(r"[।!?]+", text)
        else:
            sentences = re.split(r"[.!?]+", text)

        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return False, 0.0, "No sentences found"

        # Check average sentence length
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_words < 3:
            return False, 0.3, "Sentences too short"
        elif avg_words > 50:
            return False, 0.5, "Sentences too long"

        # Score based on sentence count and length
        if 3 <= len(sentences) <= 20 and 5 <= avg_words <= 20:
            score = 1.0
        else:
            score = 0.7

        return True, score, "Sentence structure OK"

    def check_duplication(self, text: str) -> Tuple[bool, float, str]:
        """
        Check if text is a duplicate.

        Args:
            text: Input text

        Returns:
            Tuple of (pass, score, reason)
        """
        if not self.check_duplicates:
            return True, 1.0, "Duplicate check disabled"

        # Create hash of normalized text
        normalized = " ".join(text.lower().split())
        text_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

        if text_hash in self.seen_hashes:
            return False, 0.0, "Duplicate text"

        self.seen_hashes.add(text_hash)
        return True, 1.0, "Unique text"

    def check_readability(self, text: str, lang: str) -> Tuple[bool, float, str]:
        """
        Check readability level (simplified).

        Args:
            text: Input text
            lang: Language code

        Returns:
            Tuple of (pass, score, reason)
        """
        # Simple readability heuristics
        words = text.split()
        if not words:
            return False, 0.0, "No words"

        avg_word_length = sum(len(w) for w in words) / len(words)

        # Too simple or too complex
        if avg_word_length < 3:
            score = 0.6  # Very simple, maybe too simple
        elif avg_word_length > 12:
            score = 0.7  # Complex vocabulary
        else:
            score = 1.0

        return True, score, f"Readability OK (avg word len: {avg_word_length:.1f})"

    def calculate_overall_quality(
        self, text: str, lang: str
    ) -> Tuple[float, Dict[str, Tuple[bool, float, str]]]:
        """
        Calculate overall quality score.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Tuple of (overall_score, detailed_results)
        """
        checks = {
            "length": self.check_length(text),
            "characters": self.check_character_distribution(text, lang),
            "language": self.check_language_consistency(text, lang),
            "content": self.check_content_appropriateness(text, lang),
            "sentences": self.check_sentence_structure(text, lang),
            "duplication": self.check_duplication(text),
            "readability": self.check_readability(text, lang),
        }

        # Calculate weighted average
        weights = {
            "length": 0.1,
            "characters": 0.2,
            "language": 0.15,
            "content": 0.25,  # Most important
            "sentences": 0.15,
            "duplication": 0.1,
            "readability": 0.05,
        }

        total_score = 0.0
        total_weight = 0.0

        for check_name, (passed, score, reason) in checks.items():
            if not passed and check_name in ["content", "duplication"]:
                # Critical failures
                return 0.0, checks

            weight = weights.get(check_name, 0.1)
            total_score += score * weight
            total_weight += weight

        overall_score = total_score / total_weight if total_weight > 0 else 0.0

        return overall_score, checks

    def _has_excessive_repetition(self, text: str) -> bool:
        """
        Check for excessive character or word repetition.

        Args:
            text: Input text

        Returns:
            True if excessive repetition detected
        """
        # Check character repetition
        for i in range(len(text) - 4):
            if text[i] == text[i + 1] == text[i + 2] == text[i + 3] == text[i + 4]:
                return True

        # Check word repetition
        words = text.split()
        if len(words) >= 4:
            for i in range(len(words) - 3):
                if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                    return True

        return False


def process_file(
    input_file: Path, output_file: Path, filter: QualityFilter, report_file: Optional[Path] = None
) -> Dict:
    """
    Process a single file with quality filtering.

    Args:
        input_file: Input file path
        output_file: Output file path
        filter: QualityFilter instance
        report_file: Optional report file path

    Returns:
        Statistics dictionary
    """
    print(f"Processing: {input_file}")

    stats = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "failure_reasons": Counter(),
    }

    passed_samples = []
    failed_samples = []

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        if input_file.suffix == ".jsonl":
            samples = [json.loads(line) for line in f if line.strip()]
        elif input_file.suffix == ".json":
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]
        else:
            samples = [{"text": line.strip()} for line in f if line.strip()]

    # Process each sample
    for sample in samples:
        stats["total"] += 1

        text = sample.get("text", "")
        lang = sample.get("language", "mixed")

        if not text:
            stats["failed"] += 1
            stats["failure_reasons"]["empty_text"] += 1
            continue

        # Calculate quality score
        quality_score, check_results = filter.calculate_overall_quality(text, lang)

        # Add quality score to sample
        sample["quality_score"] = quality_score
        sample["quality_checks"] = {
            name: {"passed": result[0], "score": result[1], "reason": result[2]}
            for name, result in check_results.items()
        }

        # Check if passes threshold
        if quality_score >= filter.min_quality_score:
            stats["passed"] += 1
            passed_samples.append(sample)
        else:
            stats["failed"] += 1
            # Find primary failure reason
            for name, (passed, score, reason) in check_results.items():
                if not passed or score < 0.5:
                    stats["failure_reasons"][f"{name}: {reason}"] += 1
                    break
            failed_samples.append(sample)

    # Write passed samples
    if passed_samples:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            if output_file.suffix == ".jsonl":
                for sample in passed_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                json.dump(passed_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(passed_samples)} passed samples to: {output_file}")

    # Write report if requested
    if report_file and failed_samples:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "statistics": dict(stats),
                    "failed_samples": failed_samples[:100],  # First 100 failures
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved quality report to: {report_file}")

    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Filter bilingual corpus by quality criteria")

    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output file or directory")
    parser.add_argument(
        "--min-length", type=int, default=50, help="Minimum text length (default: 50)"
    )
    parser.add_argument(
        "--max-length", type=int, default=5000, help="Maximum text length (default: 5000)"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.7,
        help="Minimum quality score 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--no-duplicate-check", action="store_true", help="Disable duplicate detection"
    )
    parser.add_argument("--report", type=str, help="Output file for quality report")

    args = parser.parse_args()

    # Initialize filter
    filter = QualityFilter(
        min_length=args.min_length,
        max_length=args.max_length,
        min_quality_score=args.min_quality,
        check_duplicates=not args.no_duplicate_check,
    )

    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else None

    if input_path.is_file():
        # Single file
        stats = process_file(input_path, output_path, filter, report_path)

        # Print statistics
        print("\n" + "=" * 60)
        print("QUALITY FILTERING STATISTICS")
        print("=" * 60)
        print(f"Total samples: {stats['total']}")
        print(f"Passed: {stats['passed']} ({stats['passed']/max(stats['total'],1)*100:.1f}%)")
        print(f"Failed: {stats['failed']} ({stats['failed']/max(stats['total'],1)*100:.1f}%)")
        print("\nTop failure reasons:")
        for reason, count in stats["failure_reasons"].most_common(10):
            print(f"  {reason}: {count}")

    elif input_path.is_dir():
        # Directory processing
        print(f"Processing directory: {input_path}")

        combined_stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "failure_reasons": Counter(),
        }

        for file in input_path.rglob("*.json*"):
            if file.suffix in [".json", ".jsonl"]:
                rel_path = file.relative_to(input_path)
                output_file = output_path / rel_path
                report_file = report_path / rel_path if report_path else None

                stats = process_file(file, output_file, filter, report_file)

                # Aggregate stats
                combined_stats["total"] += stats["total"]
                combined_stats["passed"] += stats["passed"]
                combined_stats["failed"] += stats["failed"]
                combined_stats["failure_reasons"].update(stats["failure_reasons"])

        # Print combined statistics
        print("\n" + "=" * 60)
        print("COMBINED QUALITY FILTERING STATISTICS")
        print("=" * 60)
        print(f"Total samples: {combined_stats['total']}")
        print(
            f"Passed: {combined_stats['passed']} ({combined_stats['passed']/max(combined_stats['total'],1)*100:.1f}%)"
        )
        print(
            f"Failed: {combined_stats['failed']} ({combined_stats['failed']/max(combined_stats['total'],1)*100:.1f}%)"
        )
        print("\nTop failure reasons:")
        for reason, count in combined_stats["failure_reasons"].most_common(10):
            print(f"  {reason}: {count}")

    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
