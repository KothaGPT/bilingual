"""
Tests for Text Complexity Predictor.
"""

import pytest

from bilingual.models.text_complexity_predictor import ComplexityAnalyzer


class TestComplexityAnalyzer:
    """Test suite for Text Complexity Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer instance."""
        return ComplexityAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.device in ["cuda", "cpu"]
        assert len(analyzer.METRICS) == 6

    def test_metrics_definition(self, analyzer):
        """Test metrics definitions."""
        expected_metrics = [
            "flesch_reading_ease",
            "gunning_fog_index",
            "smog_index",
            "coleman_liau_index",
            "automated_readability_index",
            "literary_complexity_score",
        ]
        assert analyzer.METRICS == expected_metrics

    def test_complexity_levels(self, analyzer):
        """Test complexity level definitions."""
        assert "very_easy" in analyzer.COMPLEXITY_LEVELS
        assert "easy" in analyzer.COMPLEXITY_LEVELS
        assert "moderate" in analyzer.COMPLEXITY_LEVELS
        assert "difficult" in analyzer.COMPLEXITY_LEVELS
        assert "very_difficult" in analyzer.COMPLEXITY_LEVELS

    def test_analyze_rule_based(self, analyzer):
        """Test rule-based complexity analysis."""
        text = "এটি একটি সাধারণ বাক্য। এটি পরীক্ষার জন্য।"
        result = analyzer.analyze(text)

        assert "metrics" in result
        assert "overall_complexity" in result
        assert "complexity_level" in result
        assert "method" in result
        assert result["method"] == "rule-based"

    def test_analyze_metrics(self, analyzer):
        """Test that all metrics are calculated."""
        text = "বাংলা ভাষা একটি সুন্দর ভাষা। এটি বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গে কথিত হয়।"
        result = analyzer.analyze(text)

        for metric in analyzer.METRICS:
            assert metric in result["metrics"]
            assert isinstance(result["metrics"][metric], (int, float))

    def test_complexity_level_assignment(self, analyzer):
        """Test complexity level assignment."""
        # Simple text
        simple_text = "আমি যাই। তুমি আসো।"
        result_simple = analyzer.analyze(simple_text)

        # Complex text
        complex_text = "বাংলা সাহিত্যের ইতিহাসে রবীন্দ্রনাথ ঠাকুরের অবদান অপরিসীম এবং তাঁর সৃষ্টিকর্ম বিশ্বসাহিত্যে এক অনন্য স্থান অধিকার করে আছে।"
        result_complex = analyzer.analyze(complex_text)

        assert result_simple["complexity_level"] in analyzer.COMPLEXITY_LEVELS
        assert result_complex["complexity_level"] in analyzer.COMPLEXITY_LEVELS

    def test_text_statistics(self, analyzer):
        """Test text statistics calculation."""
        text = "এটি একটি পরীক্ষা। আরেকটি বাক্য।"
        result = analyzer.analyze(text)

        if "text_statistics" in result:
            stats = result["text_statistics"]
            assert "total_words" in stats
            assert "total_sentences" in stats
            assert "avg_word_length" in stats
            assert "avg_sentence_length" in stats

    def test_batch_analyze(self, analyzer):
        """Test batch analysis."""
        texts = [
            "সহজ বাক্য।",
            "এটি একটু জটিল বাক্য যা বেশি শব্দ ধারণ করে।",
        ]

        results = analyzer.batch_analyze(texts)
        assert len(results) == 2
        assert all("metrics" in r for r in results)

    def test_compare_texts(self, analyzer):
        """Test text comparison."""
        texts = [
            "সহজ।",
            "মাঝারি জটিলতার বাক্য।",
            "এটি একটি অত্যন্ত জটিল এবং দীর্ঘ বাক্য যা অনেক শব্দ এবং ধারণা ধারণ করে।",
        ]

        comparison = analyzer.compare_texts(texts)

        assert "texts" in comparison
        assert "average_complexity" in comparison
        assert "min_complexity" in comparison
        assert "max_complexity" in comparison
        assert "complexity_range" in comparison

    def test_get_complexity_level(self, analyzer):
        """Test complexity level getter."""
        assert analyzer._get_complexity_level(20) == "very_easy"
        assert analyzer._get_complexity_level(40) == "easy"
        assert analyzer._get_complexity_level(60) == "moderate"
        assert analyzer._get_complexity_level(80) == "difficult"
        assert analyzer._get_complexity_level(95) == "very_difficult"

    def test_empty_text(self, analyzer):
        """Test analysis with empty text."""
        result = analyzer.analyze("")
        assert "metrics" in result
