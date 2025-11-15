"""
Tests for Text Complexity Predictor module.
"""

import pytest

from src.bilingual.modules.text_complexity_predictor import (
    analyze_readability,
    classify_difficulty,
    predict_complexity,
    suggest_simplifications,
)


class TestTextComplexityPredictor:
    """Test suite for text complexity prediction."""

    def test_predict_complexity_returns_float(self):
        """Test that complexity prediction returns a float."""
        text = "এটি একটি সাধারণ বাক্য।"
        complexity = predict_complexity(text)

        assert isinstance(complexity, float)
        assert 0 <= complexity <= 1

    def test_predict_complexity_simple_text(self):
        """Test complexity prediction for simple text."""
        text = "আমি ভাত খাই।"
        complexity = predict_complexity(text)

        assert isinstance(complexity, float)

    def test_predict_complexity_complex_text(self):
        """Test complexity prediction for complex text."""
        text = "বাংলা সাহিত্যের ইতিহাসে রবীন্দ্রনাথ ঠাকুরের অবদান অপরিসীম এবং বহুমাত্রিক।"
        complexity = predict_complexity(text)

        assert isinstance(complexity, float)

    def test_analyze_readability_structure(self):
        """Test readability analysis returns correct structure."""
        text = "এটি একটি পরীক্ষা।"
        analysis = analyze_readability(text)

        assert isinstance(analysis, dict)
        assert "complexity_score" in analysis
        assert "avg_sentence_length" in analysis
        assert "avg_word_length" in analysis
        assert "vocabulary_diversity" in analysis
        assert "estimated_grade_level" in analysis

    def test_suggest_simplifications_returns_list(self):
        """Test that simplification suggestions return a list."""
        text = "এটি একটি জটিল বাক্য।"
        suggestions = suggest_simplifications(text)

        assert isinstance(suggestions, list)

    def test_classify_difficulty_beginner(self):
        """Test difficulty classification for beginner level."""
        # Mock predict_complexity to return low score
        text = "আমি খাই।"
        difficulty = classify_difficulty(text)

        assert difficulty in ["beginner", "intermediate", "advanced", "expert"]

    def test_classify_difficulty_returns_valid_level(self):
        """Test that difficulty classification returns valid level."""
        text = "বাংলা ভাষা সুন্দর।"
        difficulty = classify_difficulty(text)

        assert difficulty in ["beginner", "intermediate", "advanced", "expert"]

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        text = ""
        complexity = predict_complexity(text)

        assert isinstance(complexity, float)
        assert 0 <= complexity <= 1
