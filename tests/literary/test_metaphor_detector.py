"""
Tests for Metaphor and Simile Detector.
"""

import pytest

from bilingual.models.metaphor_detector import MetaphorSimileDetector


class TestMetaphorSimileDetector:
    """Test suite for Metaphor and Simile Detector."""

    @pytest.fixture
    def detector(self):
        """Create a test detector instance."""
        return MetaphorSimileDetector()

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert detector.device in ["cuda", "cpu"]
        assert len(detector.LABEL_NAMES) == 5

    def test_label_mappings(self, detector):
        """Test label ID mappings."""
        assert detector.id2label[0] == "O"
        assert detector.id2label[1] == "B-METAPHOR"
        assert detector.label2id["O"] == 0
        assert detector.label2id["B-METAPHOR"] == 1

    def test_detect_without_model(self, detector):
        """Test detection without pretrained model."""
        if detector.model is None:
            with pytest.raises(ValueError, match="Model not initialized"):
                detector.detect("তুমি আমার হৃদয়ের চাঁদ")

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_detect_metaphor(self, detector):
        """Test metaphor detection."""
        text = "তুমি আমার জীবনের আলো"
        result = detector.detect(text)

        assert "text" in result
        assert "metaphors" in result
        assert "similes" in result
        assert "total_figurative" in result

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_detect_simile(self, detector):
        """Test simile detection."""
        text = "সে চাঁদের মতো সুন্দর"
        result = detector.detect(text)

        assert isinstance(result["similes"], list)

    def test_batch_detect(self, detector):
        """Test batch detection."""
        if detector.model is None:
            pytest.skip("Model not available")

        texts = [
            "তুমি আমার হৃদয়ের রাজা",
            "সে ফুলের মতো কোমল",
        ]

        results = detector.batch_detect(texts)
        assert len(results) == 2

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_get_statistics(self, detector):
        """Test figurative language statistics."""
        text = "তুমি আমার জীবনের আলো, আমার হৃদয়ের চাঁদ"
        stats = detector.get_statistics(text)

        assert "total_words" in stats
        assert "metaphor_count" in stats
        assert "simile_count" in stats
        assert "figurative_density" in stats

    def test_extract_spans_empty(self, detector):
        """Test span extraction with no entities."""
        spans = detector._extract_spans(
            text="একটি সাধারণ বাক্য",
            tokens=["একটি", "সাধারণ", "বাক্য"],
            labels=["O", "O", "O"],
            offset_mapping=[[0, 4], [5, 11], [12, 16]],
            entity_type="METAPHOR",
        )

        assert len(spans) == 0
