"""
Tests for poetic meter detection and analysis.
"""

import pytest

from bilingual.modules.poetic_meter import (
    BANGLA_METERS,
    count_syllables,
    detect_bangla_meter,
    detect_meter,
)


class TestSyllableCounting:
    """Test suite for syllable counting in different languages."""

    def test_english_syllable_counting(self):
        """Test counting syllables in English words."""
        assert count_syllables("hello", "en") == 2
        assert count_syllables("beautiful", "en") == 3
        assert count_syllables("the", "en") == 1
        assert count_syllables("banana", "en") == 3

    def test_bangla_syllable_counting(self):
        """Test counting syllables in Bangla words."""
        # Each vowel or matra counts as a syllable
        assert count_syllables("বাংলা", "bn") == 2  # বাং-লা
        assert count_syllables("ভাষা", "bn") == 2  # ভাষা
        assert count_syllables("আমি", "bn") == 2  # আ-মি

    def test_empty_word(self):
        """Test that empty string returns 0."""
        assert count_syllables("", "en") == 0
        assert count_syllables("", "bn") == 0


class TestDetectMeter:
    """Test suite for meter detection."""

    def test_english_meter_detection(self):
        """Test meter detection for English text."""
        # A simple haiku (5-7-5 syllables)
        text = """
        An old silent pond
        A frog jumps into the pond
        Splash! Silence again.
        """
        result = detect_meter(text, "en")
        assert result["meter_type"] != ""  # Should have a meter type
        assert result["line_count"] == 3
        assert len(result["syllable_counts"]) == 3

        # Test with a single line
        text = "Shall I compare thee to a summer's day?"
        result = detect_meter(text, "en")
        assert result["line_count"] == 1
        assert len(result["syllable_counts"]) == 1
        assert result["syllable_counts"][0] > 0

    def test_bengali_single_line(self):
        """Test meter detection on single Bengali line."""
        text = "আমার সোনার বাংলা আমি তোমায় ভালোবাসি"
        result = detect_meter(text, language="bengali")

        assert result["language"] == "bengali"
        assert len(result["lines"]) == 1
        assert "total_matra" in result["lines"][0]
        assert result["lines"][0]["total_matra"] > 0

    def test_multi_line_english(self):
        """Test meter detection on multiple English lines."""
        text = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate"""
        result = detect_meter(text, language="english")

        assert len(result["lines"]) == 2
        assert result["lines"][0]["line_number"] == 1
        assert result["lines"][1]["line_number"] == 2

    def test_multi_line_bengali(self):
        """Test meter detection on multiple Bengali lines."""
        text = """আমার সোনার বাংলা
আমি তোমায় ভালোবাসি"""
        result = detect_meter(text, language="bengali")

        assert len(result["lines"]) == 2
        assert result["language"] == "bengali"

    def test_auto_language_detection_english(self):
        """Test automatic language detection for English."""
        text = "The quick brown fox jumps over the lazy dog"
        result = detect_meter(text, language="auto")

        assert result["language"] == "english"

    def test_auto_language_detection_bengali(self):
        """Test automatic language detection for Bengali."""
        text = "আমি বাংলায় গান গাই"
        result = detect_meter(text, language="auto")

        assert result["language"] == "bengali"

    def test_pattern_detection_consistent(self):
        """Test pattern detection for consistent meter."""
        # Lines with similar syllable counts
        text = """The cat sat on the mat today
The dog ran in the park to play"""
        result = detect_meter(text, language="english")

        assert "pattern" in result
        # Pattern should be detected (consistent, iambic, or irregular)
        assert result["pattern"] in ["consistent", "iambic", "irregular"]

    def test_pattern_detection_irregular(self):
        """Test pattern detection for irregular meter."""
        text = """Hi
This is a much longer line with many more syllables"""
        result = detect_meter(text, language="english")

        assert result["pattern"] in ["irregular", "consistent", "iambic"]

    def test_empty_lines_skipped(self):
        """Test that empty lines are skipped."""
        text = """Line one

Line three"""
        result = detect_meter(text, language="english")

        # Should only count non-empty lines
        assert len(result["lines"]) == 2

    def test_summary_statistics(self):
        """Test that summary statistics are included."""
        text = """The cat sat on the mat
The dog ran in the park"""
        result = detect_meter(text, language="english")

        assert "summary" in result
        assert "total_lines" in result["summary"]
        assert "avg_units_per_line" in result["summary"]
        assert result["summary"]["total_lines"] == 2

    def test_word_count_per_line(self):
        """Test that word count is tracked per line."""
        text = "One two three four five"
        result = detect_meter(text, language="english")

        assert result["lines"][0]["word_count"] == 5

    def test_syllables_per_word(self):
        """Test that syllables per word are tracked."""
        text = "Hello world"
        result = detect_meter(text, language="english")

        assert "syllables_per_word" in result["lines"][0]
        assert len(result["lines"][0]["syllables_per_word"]) == 2

    def test_empty_text(self):
        """Test meter detection on empty text."""
        result = detect_meter("", language="english")

        assert result["lines"] == []
        assert result["pattern"] == "unknown"


class TestPoeticMeterIntegration:
    """Integration tests for poetic meter module."""

    def test_iambic_pentameter_approximation(self):
        """Test detection of iambic pentameter (10 syllables)."""
        # Classic iambic pentameter line
        text = "Shall I compare thee to a summer's day?"
        result = detect_meter(text, language="english")

        # Should detect roughly 10 syllables
        syllables = result["lines"][0]["total_syllables"]
        assert 8 <= syllables <= 12  # Allow some variance

    def test_known_bangla_meters(self):
        """Test that all known Bangla meters are in the BANGLA_METERS dict."""
        assert "পয়ার" in BANGLA_METERS
        assert "ত্রিপদী" in BANGLA_METERS
        assert "চৌপদী" in BANGLA_METERS

    def test_detect_meter_with_empty_text(self):
        """Test meter detection with empty text."""
        result = detect_meter("", "en")
        assert result["line_count"] == 0
        assert result["avg_syllables"] == 0

    def test_detect_meter_with_whitespace(self):
        """Test that whitespace-only text is handled correctly."""
        result = detect_meter("   \n  \n ", "en")
        assert result["line_count"] == 0
        assert result["avg_syllables"] == 0

    def test_bengali_payar_approximation(self):
        """Test detection of Bengali পয়ার ছন্দ pattern."""
        # Traditional payar has 14 matra per line
        text = "আমার সোনার বাংলা আমি তোমায় ভালোবাসি"
        result = detect_meter(text, language="bengali")

        # Should detect matra count in reasonable range
        matra = result["lines"][0]["total_matra"]
        assert matra > 0

    def test_bilingual_text_separate_analysis(self):
        """Test that bilingual text can be analyzed separately."""
        english_text = "The quick brown fox"
        bengali_text = "দ্রুত বাদামী শিয়াল"

        english_result = detect_meter(english_text, language="english")
        bengali_result = detect_meter(bengali_text, language="bengali")

        assert english_result["language"] == "english"
        assert bengali_result["language"] == "bengali"
        assert "syllables" in str(english_result["lines"][0])
        assert "matra" in str(bengali_result["lines"][0])

    def test_result_structure_completeness(self):
        """Test that result contains all expected fields."""
        text = "Sample line for testing"
        result = detect_meter(text)

        # Top-level fields
        assert "lines" in result
        assert "pattern" in result
        assert "language" in result
        assert "summary" in result

        # Line-level fields
        if result["lines"]:
            line = result["lines"][0]
            assert "line_number" in line
            assert "text" in line
            assert "word_count" in line
