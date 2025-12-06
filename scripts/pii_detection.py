#!/usr/bin/env python3
"""
PII (Personally Identifiable Information) Detection and Removal Script.

This script detects and removes/redacts personal information from text data
to ensure privacy protection in the bilingual corpus.

Usage:
    python scripts/pii_detection.py \
        --input data/raw/ \
        --output data/cleaned/ \
        --mode redact
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PIIDetector:
    """Detect and remove PII from text in Bangla and English."""

    def __init__(self, language: str = "mixed"):
        """
        Initialize PII detector.

        Args:
            language: Target language ('bn', 'en', or 'mixed')
        """
        self.language = language
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for PII detection."""

        # Email patterns
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

        # Phone patterns (various formats)
        self.phone_patterns = [
            # Bangladesh: +880, 880, 01...
            re.compile(r"\+?880[-\s]?1\d{9}"),
            re.compile(r"\b01\d{9}\b"),
            # Generic international
            re.compile(r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"),
            # US/Generic
            re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ]

        # URL patterns (may contain personal info)
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # Credit card patterns
        self.credit_card_pattern = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

        # National ID patterns (Bangladesh)
        self.nid_pattern = re.compile(r"\b\d{10}(?:\d{3}|\d{7})?\b")  # 10, 13, or 17 digit NID

        # IP addresses
        self.ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

        # Common Bangla/English name patterns (requires NER for better detection)
        # These are simple heuristics
        self.name_indicators = [
            # English
            r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            # Bangla honorifics followed by names
            r"জনাব\s+\S+",
            r"ডঃ\s+\S+",
            r"প্রফেসর\s+\S+",
        ]
        self.name_pattern = re.compile("|".join(self.name_indicators))

        # Address patterns (basic detection)
        self.address_indicators = [
            r"\b\d+\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln)\b",
            r"\b(?:Dhaka|Chittagong|Sylhet|Rajshahi|Khulna|Barisal)\s*-?\s*\d{4}\b",
            r"রোড\s+\d+",
            r"বাড়ি\s*(?:নং|নম্বর)?\s*\d+",
        ]
        self.address_pattern = re.compile("|".join(self.address_indicators), re.IGNORECASE)

    def detect_emails(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect email addresses in text.

        Args:
            text: Input text

        Returns:
            List of (email, start_pos, end_pos) tuples
        """
        matches = []
        for match in self.email_pattern.finditer(text):
            matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_phones(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect phone numbers in text.

        Args:
            text: Input text

        Returns:
            List of (phone, start_pos, end_pos) tuples
        """
        matches = []
        for pattern in self.phone_patterns:
            for match in pattern.finditer(text):
                matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_urls(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect URLs in text.

        Args:
            text: Input text

        Returns:
            List of (url, start_pos, end_pos) tuples
        """
        matches = []
        for match in self.url_pattern.finditer(text):
            matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_credit_cards(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect potential credit card numbers.

        Args:
            text: Input text

        Returns:
            List of (number, start_pos, end_pos) tuples
        """
        matches = []
        for match in self.credit_card_pattern.finditer(text):
            # Simple Luhn check to reduce false positives
            number = match.group().replace("-", "").replace(" ", "")
            if self._luhn_check(number):
                matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_nids(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect Bangladesh National ID numbers.

        Args:
            text: Input text

        Returns:
            List of (nid, start_pos, end_pos) tuples
        """
        matches = []
        for match in self.nid_pattern.finditer(text):
            matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_ip_addresses(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect IP addresses.

        Args:
            text: Input text

        Returns:
            List of (ip, start_pos, end_pos) tuples
        """
        matches = []
        for match in self.ip_pattern.finditer(text):
            # Validate IP address
            parts = match.group().split(".")
            if all(0 <= int(p) <= 255 for p in parts):
                matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_names(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect potential person names (basic heuristic).

        Note: This is a simple pattern-based approach. For production,
        use NER models for better accuracy.

        Args:
            text: Input text

        Returns:
            List of (name, start_pos, end_pos) tuples
        """
        matches = []
        for match in self.name_pattern.finditer(text):
            matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_addresses(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect potential addresses (basic heuristic).

        Args:
            text: Input text

        Returns:
            List of (address, start_pos, end_pos) tuples
        """
        matches = []
        for match in self.address_pattern.finditer(text):
            matches.append((match.group(), match.start(), match.end()))
        return matches

    def detect_all(self, text: str) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        Detect all types of PII in text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping PII type to list of matches
        """
        return {
            "emails": self.detect_emails(text),
            "phones": self.detect_phones(text),
            "urls": self.detect_urls(text),
            "credit_cards": self.detect_credit_cards(text),
            "nids": self.detect_nids(text),
            "ip_addresses": self.detect_ip_addresses(text),
            "names": self.detect_names(text),
            "addresses": self.detect_addresses(text),
        }

    def redact_text(
        self, text: str, mode: str = "redact", pii_types: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Redact PII from text.

        Args:
            text: Input text
            mode: Redaction mode ('redact', 'remove', 'mask')
            pii_types: List of PII types to redact (None = all)

        Returns:
            Tuple of (redacted_text, statistics_dict)
        """
        if pii_types is None:
            pii_types = [
                "emails",
                "phones",
                "urls",
                "credit_cards",
                "nids",
                "ip_addresses",
                "names",
                "addresses",
            ]

        # Detect all PII
        all_pii = self.detect_all(text)

        # Collect all matches with their positions
        matches_to_redact = []
        stats = {pii_type: 0 for pii_type in pii_types}

        for pii_type in pii_types:
            if pii_type in all_pii:
                for match, start, end in all_pii[pii_type]:
                    matches_to_redact.append((start, end, pii_type))
                    stats[pii_type] += 1

        # Sort by position (reverse order to maintain indices)
        matches_to_redact.sort(reverse=True)

        # Redact matches
        redacted_text = text
        for start, end, pii_type in matches_to_redact:
            if mode == "redact":
                replacement = f"[{pii_type.upper()}]"
            elif mode == "remove":
                replacement = ""
            elif mode == "mask":
                replacement = "*" * (end - start)
            else:
                replacement = "[REDACTED]"

            redacted_text = redacted_text[:start] + replacement + redacted_text[end:]

        return redacted_text, stats

    @staticmethod
    def _luhn_check(card_number: str) -> bool:
        """
        Perform Luhn algorithm check for credit card validation.

        Args:
            card_number: Card number string

        Returns:
            True if valid according to Luhn algorithm
        """

        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10 == 0


def process_file(
    input_file: Path,
    output_file: Path,
    detector: PIIDetector,
    mode: str = "redact",
    report_only: bool = False,
) -> Dict:
    """
    Process a single file for PII detection/removal.

    Args:
        input_file: Input file path
        output_file: Output file path
        detector: PIIDetector instance
        mode: Redaction mode
        report_only: If True, only report PII without redacting

    Returns:
        Statistics dictionary
    """
    print(f"Processing: {input_file}")

    total_stats = {
        "samples_processed": 0,
        "samples_with_pii": 0,
        "total_pii_found": 0,
        "emails": 0,
        "phones": 0,
        "urls": 0,
        "credit_cards": 0,
        "nids": 0,
        "ip_addresses": 0,
        "names": 0,
        "addresses": 0,
    }

    output_data = []

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        if input_file.suffix == ".jsonl":
            samples = [json.loads(line) for line in f if line.strip()]
        elif input_file.suffix == ".json":
            samples = json.load(f)
            if not isinstance(samples, list):
                samples = [samples]
        else:
            # Plain text file
            samples = [{"text": line.strip()} for line in f if line.strip()]

    # Process each sample
    for sample in samples:
        # Determine fields to process
        fields_to_process = []
        if "text" in sample:
            fields_to_process.append("text")
        if "bn" in sample:
            fields_to_process.append("bn")
        if "en" in sample:
            fields_to_process.append("en")
            
        if not fields_to_process:
            continue

        total_stats["samples_processed"] += 1
        output_sample = sample.copy()
        sample_pii_found = False
        sample_pii_stats = {}

        for field in fields_to_process:
            text = sample.get(field, "")
            if not text:
                continue

            # Redact PII
            redacted_text, pii_stats = detector.redact_text(text, mode=mode)
            
            # Update output sample
            output_sample[field] = redacted_text

            # Update statistics
            field_pii_found = sum(pii_stats.values())
            if field_pii_found > 0:
                sample_pii_found = True
                
                # Merge stats
                for key, value in pii_stats.items():
                    total_stats[key] += value
                    sample_pii_stats[key] = sample_pii_stats.get(key, 0) + value

        if sample_pii_found:
            total_stats["samples_with_pii"] += 1
            total_stats["total_pii_found"] += sum(sample_pii_stats.values())

        # Prepare output
        if not report_only:
            if sample_pii_found:
                output_sample["pii_removed"] = sample_pii_stats
            output_data.append(output_sample)

    # Write output
    if not report_only and output_data:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            if output_file.suffix == ".jsonl":
                for sample in output_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to: {output_file}")

    return total_stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Detect and remove PII from bilingual corpus")

    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument(
        "--output", type=str, help="Output file or directory (required unless --report-only)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="redact",
        choices=["redact", "remove", "mask"],
        help="Redaction mode: redact=[TAG], remove=delete, mask=***",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="mixed",
        choices=["bn", "en", "mixed"],
        help="Target language",
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Only report PII statistics without redacting"
    )
    parser.add_argument(
        "--pii-types", type=str, nargs="+", help="Specific PII types to detect (default: all)"
    )

    args = parser.parse_args()

    if not args.report_only and not args.output:
        print("Error: --output required unless --report-only is specified")
        sys.exit(1)

    # Initialize detector
    detector = PIIDetector(language=args.language)

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        output_path = Path(args.output) if args.output else None
        stats = process_file(
            input_path, output_path, detector, mode=args.mode, report_only=args.report_only
        )

        # Print statistics
        print("\n" + "=" * 60)
        print("PII DETECTION STATISTICS")
        print("=" * 60)
        print(f"Samples processed: {stats['samples_processed']}")
        print(f"Samples with PII: {stats['samples_with_pii']}")
        print(f"Total PII instances: {stats['total_pii_found']}")
        print("\nBreakdown:")
        for key, value in stats.items():
            if key not in ["samples_processed", "samples_with_pii", "total_pii_found"]:
                if value > 0:
                    print(f"  {key}: {value}")

    elif input_path.is_dir():
        # Directory processing
        print(f"Processing directory: {input_path}")
        output_dir = Path(args.output) if args.output else None

        combined_stats = {
            "samples_processed": 0,
            "samples_with_pii": 0,
            "total_pii_found": 0,
        }

        for file in input_path.rglob("*.json*"):
            if file.suffix in [".json", ".jsonl"]:
                if output_dir:
                    rel_path = file.relative_to(input_path)
                    output_file = output_dir / rel_path
                else:
                    output_file = None

                stats = process_file(
                    file, output_file, detector, mode=args.mode, report_only=args.report_only
                )

                # Aggregate stats
                for key in combined_stats:
                    combined_stats[key] += stats.get(key, 0)

        # Print combined statistics
        print("\n" + "=" * 60)
        print("COMBINED PII DETECTION STATISTICS")
        print("=" * 60)
        print(f"Samples processed: {combined_stats['samples_processed']}")
        print(f"Samples with PII: {combined_stats['samples_with_pii']}")
        print(f"Total PII instances: {combined_stats['total_pii_found']}")

    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
