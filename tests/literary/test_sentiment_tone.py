"""
Tests for Sentiment and Tone Classifier.
"""

import pytest

pytest.importorskip("torch")
from bilingual.models.sentiment_tone_classifier import SentimentToneAnalyzer


class TestSentimentToneAnalyzer:
    """Test suite for Sentiment and Tone Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer instance."""
        return SentimentToneAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.device in ["cuda", "cpu"]
        assert len(analyzer.SENTIMENT_LABELS) == 3
        assert len(analyzer.TONE_LABELS) == 8

    def test_sentiment_labels(self, analyzer):
        """Test sentiment label definitions."""
        assert "positive" in analyzer.SENTIMENT_LABELS
        assert "neutral" in analyzer.SENTIMENT_LABELS
        assert "negative" in analyzer.SENTIMENT_LABELS

    def test_tone_labels(self, analyzer):
        """Test tone label definitions."""
        expected_tones = [
            "formal",
            "informal",
            "poetic",
            "dramatic",
            "melancholic",
            "joyful",
            "satirical",
            "romantic",
        ]
        for tone in expected_tones:
            assert tone in analyzer.TONE_LABELS

    def test_analyze_without_model(self, analyzer):
        """Test analysis without pretrained model."""
        if analyzer.model is None:
            with pytest.raises(ValueError, match="Model not initialized"):
                analyzer.analyze("আমি খুব খুশি")

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_analyze_positive_sentiment(self, analyzer):
        """Test positive sentiment analysis."""
        text = "আজ আমি খুব খুশি এবং আনন্দিত"
        result = analyzer.analyze(text)

        assert "sentiment" in result
        assert "tones" in result
        assert result["sentiment"]["label"] in analyzer.SENTIMENT_LABELS

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_analyze_negative_sentiment(self, analyzer):
        """Test negative sentiment analysis."""
        text = "আমি খুব দুঃখিত এবং হতাশ"
        result = analyzer.analyze(text)

        assert result["sentiment"]["label"] in analyzer.SENTIMENT_LABELS

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_analyze_poetic_tone(self, analyzer):
        """Test poetic tone detection."""
        text = "চাঁদের আলোয় ভেসে যায় মন, স্বপ্নের দেশে হারিয়ে যাই"
        result = analyzer.analyze(text)

        assert "tones" in result
        assert isinstance(result["tones"], list)

    def test_batch_analyze(self, analyzer):
        """Test batch analysis."""
        if analyzer.model is None:
            pytest.skip("Model not available")

        texts = [
            "আমি খুব খুশি",
            "আমি দুঃখিত",
        ]

        results = analyzer.batch_analyze(texts)
        assert len(results) == 2

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_get_sentiment_distribution(self, analyzer):
        """Test sentiment distribution calculation."""
        texts = [
            "আমি খুশি",
            "আমি দুঃখিত",
            "এটি ঠিক আছে",
        ]

        distribution = analyzer.get_sentiment_distribution(texts)

        assert isinstance(distribution, dict)
        assert len(distribution) == 3
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_tone_threshold(self, analyzer):
        """Test tone threshold parameter."""
        if analyzer.model is None:
            pytest.skip("Model not available")

        text = "একটি সাধারণ বাক্য"

        # Lower threshold should detect more tones
        result_low = analyzer.analyze(text, tone_threshold=0.3)
        result_high = analyzer.analyze(text, tone_threshold=0.7)

        assert len(result_low["tones"]) >= len(result_high["tones"])
