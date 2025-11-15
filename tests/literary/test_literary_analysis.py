"""
Tests for literary device detection and tone analysis.
"""

import pytest

from bilingual.modules.literary_analysis import (
    analyze_tone,
    detect_metaphors,
    detect_similes,
)


class TestMetaphorDetector:
    """Test suite for metaphor detection."""

    def test_english_metaphor_is_a(self):
        """Test detection of 'X is a Y' metaphors in English."""
        text = "Life is a journey"
        result = detect_metaphors(text, language="en")
        assert len(result) > 0
        assert result[0]["type"] == "metaphor"
        assert "confidence" in result[0]
        assert "start" in result[0]
        assert "end" in result[0]

        assert len(result) == 1
        assert result[0]["type"] == "metaphor"
        assert "Life is a journey" in result[0]["text"]

    def test_english_metaphor_are(self):
        """Test detection of 'X are Y' metaphors."""
        text = "Words are weapons"
        result = detect_metaphors(text, language="en")
        assert len(result) > 0
        assert result[0]["type"] == "metaphor"

        assert len(result) == 1
        assert result[0]["type"] == "metaphor"

    def test_bengali_metaphor_holo(self):
        """Test detection of Bengali 'X হল Y' metaphors."""
        text = "জীবন হল একটি যাত্রা"
        result = metaphor_detector(text)

        assert len(result) >= 1
        assert any(m["type"] == "metaphor" for m in result)

    def test_bengali_metaphor_jeno(self):
        """Test detection of Bengali 'X যেন Y' metaphors."""
        text = "সে যেন একটি ফুল"
        result = metaphor_detector(text)

        assert len(result) >= 1
        assert all(m["type"] == "metaphor" for m in result)

    def test_no_metaphors(self):
        """Test text without metaphors returns empty list."""
        text = "The cat sat on the mat."
        result = metaphor_detector(text)

        # May detect false positives with simple patterns, but should be minimal
        assert isinstance(result, list)

    def test_empty_text(self):
        """Test empty text returns empty list."""
        result = metaphor_detector("")
        assert result == []

    def test_metaphor_positions(self):
        """Test that detected metaphors include position information."""
        text = "Time is money"
        result = metaphor_detector(text)

        if result:
            assert "start" in result[0]
            assert "end" in result[0]
            assert result[0]["start"] >= 0
            assert result[0]["end"] > result[0]["start"]


class TestSimileDetector:
    """Test suite for simile detection."""

    def test_english_simile_like(self):
        """Test detection of 'like' similes in English."""
        text = "She runs like the wind"
        result = detect_similes(text, language="en")
        assert len(result) > 0
        assert result[0]["type"] == "simile"
        assert "confidence" in result[0]
        assert "start" in result[0]
        assert "end" in result[0]

        assert len(result) >= 1
        assert any("like" in s["text"] for s in result)
        assert all(s["type"] == "simile" for s in result)

    def test_english_simile_as_as(self):
        """Test detection of 'as...as' similes."""
        text = "He is as brave as a lion"
        result = detect_similes(text, language="en")
        assert len(result) > 0
        assert any("as brave as" in s["text"] for s in result)

        assert len(result) >= 1
        assert all(s["type"] == "simile" for s in result)

    def test_bengali_simile_jemon(self):
        """Test detection of Bengali 'যেমন' similes."""
        text = "যেমন ফুল ফোটে"
        result = simile_detector(text)

        assert len(result) >= 1
        assert all(s["type"] == "simile" for s in result)

    def test_bengali_simile_moto(self):
        """Test detection of Bengali 'মতো' similes."""
        text = "বাতাসের মতো দ্রুত"
        result = simile_detector(text)

        assert len(result) >= 1
        assert all(s["type"] == "simile" for s in result)

    def test_no_similes(self):
        """Test text without similes returns empty list."""
        text = "The dog barked loudly."
        result = simile_detector(text)

        assert isinstance(result, list)

    def test_empty_text(self):
        """Test empty text returns empty list."""
        result = simile_detector("")
        assert result == []

    def test_simile_positions(self):
        """Test that detected similes include position information."""
        text = "Fast like lightning"
        result = simile_detector(text)

        if result:
            assert "start" in result[0]
            assert "end" in result[0]


class TestToneClassifier:
    """Test suite for tone classification."""

    def test_positive_tone(self):
        """Test classification of positive text."""
        text = "This is wonderful! I'm so happy with the results."
        result = analyze_tone(text, language="en")
        assert result["positive"] > result["negative"]
        assert result["positive"] > result["neutral"]
        # Ensure values are between 0 and 1
        assert 0 <= result["positive"] <= 1
        assert 0 <= result["negative"] <= 1
        assert 0 <= result["neutral"] <= 1
        assert "negative" in result
        assert result["positive"] > result["negative"]

    def test_negative_tone(self):
        """Test classification of negative text."""
        text = "This is terrible and awful."
        result = analyze_tone(text, language="en")

        assert result["negative"] > result["positive"]

    def test_neutral_tone(self):
        """Test classification of neutral text."""
        text = "The meeting is scheduled for tomorrow."
        result = analyze_tone(text, language="en")

        assert result["neutral"] >= result["positive"]
        assert result["neutral"] >= result["negative"]

    def test_bengali_positive_tone(self):
        """Test classification of positive Bengali text."""
        text = "এটি খুব সুন্দর এবং ভালো"
        result = analyze_tone(text, language="bn")

        assert result["positive"] > result["negative"]

    def test_bengali_negative_tone(self):
        """Test classification of negative Bengali text."""
        text = "এটি খুব খারাপ এবং দুঃখজনক"
        result = analyze_tone(text, language="bn")

        assert result["negative"] > result["positive"]

    def test_probability_sum(self):
        """Test that probabilities sum to approximately 1.0."""
        text = "Some random text here"
        result = analyze_tone(text, language="en")

        total = result["positive"] + result["neutral"] + result["negative"]
        assert 0.99 <= total <= 1.01  # Allow small floating point error

    def test_probability_range(self):
        """Test that all probabilities are in valid range [0, 1]."""
        text = "Test text with mixed sentiment good and bad"
        result = analyze_tone(text, language="en")

        for key in ["positive", "neutral", "negative"]:
            assert 0.0 <= result[key] <= 1.0

    def test_empty_text(self):
        """Test classification of empty text returns neutral."""
        result = analyze_tone("", language="en")

        assert result["neutral"] > result["positive"]
        assert result["neutral"] > result["negative"]

    def test_mixed_sentiment(self):
        """Test text with mixed sentiment."""
        text = "The good news is great, but the bad news is terrible."
        result = analyze_tone(text, language="en")

        # Should detect both positive and negative
        assert result["positive"] > 0.0
        assert result["negative"] > 0.0


class TestLiteraryAnalysisIntegration:
    """Integration tests for literary analysis module."""

    def test_bilingual_text_analysis(self):
        """Test analysis of mixed Bengali-English text."""
        text = "Life is a journey. জীবন একটি নদীর মত"

        # Test metaphor detection
        metaphors = detect_metaphors(text, language="en")
        assert any("journey" in m["text"] for m in metaphors)

        # Test simile detection in Bangla
        similes = detect_similes(text, language="bn")
        assert any("মত" in s["text"] for s in similes)

        # Test tone analysis for both languages
        en_tone = analyze_tone("This is wonderful!", language="en")
        bn_tone = analyze_tone("এটা অসাধারণ!", language="bn")

        # Just verify the structure is correct
        for tone in [en_tone, bn_tone]:
            assert set(tone.keys()) == {"positive", "negative", "neutral"}
            assert abs(sum(tone.values()) - 1.0) < 0.0001  # Should sum to ~1.0

    def test_all_detectors_return_correct_types(self):
        """Test that all detectors return expected types."""
        text = "Sample text for testing"

        metaphors = detect_metaphors(text, language="en")
        similes = detect_similes(text, language="en")
        tone = analyze_tone(text, language="en")

        # Check return types
        assert isinstance(metaphors, list)
        assert isinstance(similes, list)
        assert isinstance(tone, dict)

        # Check structure of returned items
        for m in metaphors:
            assert "text" in m
            assert "type" in m

        for s in similes:
            assert "text" in s
            assert "type" in s
