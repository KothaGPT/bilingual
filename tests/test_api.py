"""Tests for high-level API."""

from bilingual.normalize import normalize_text
from bilingual.api import readability_check, safety_check, classify


class TestNormalizeAPI:
    def test_normalize_text(self):
        result = normalize_text("আমি   স্কুলে যাই।", lang="bn")
        assert isinstance(result, str)
        assert "  " not in result

    def test_normalize_auto_detect(self):
        result = normalize_text("আমি স্কুলে যাই।")
        assert isinstance(result, str)


class TestReadabilityAPI:
    def test_readability_check(self):
        result = readability_check("আমি স্কুলে যাই।", lang="bn")

        assert "level" in result
        assert "age_range" in result
        assert "score" in result
        assert "language" in result

        assert isinstance(result["score"], (int, float))


class TestSafetyAPI:
    def test_safety_check(self):
        result = safety_check("This is a nice story about rabbits.")

        assert "is_safe" in result
        assert "confidence" in result
        assert "flags" in result
        assert "recommendation" in result

        assert isinstance(result["is_safe"], bool)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["flags"], list)


class TestClassifyAPI:
    def test_classify(self):
        labels = ["story", "news", "dialogue"]
        result = classify("Once upon a time...", labels=labels)

        assert isinstance(result, dict)
        assert len(result) == len(labels)

        for label in labels:
            assert label in result
            assert isinstance(result[label], (int, float))
