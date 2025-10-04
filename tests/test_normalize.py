"""Tests for text normalization utilities."""

from bilingual.normalize import (
    contains_bangla,
    detect_language,
    is_bangla_char,
    normalize_bangla_digits,
    normalize_text,
    normalize_unicode,
    split_sentences,
)


class TestUnicodeNormalization:
    def test_normalize_unicode_nfc(self):
        text = "café"  # May have combining characters
        result = normalize_unicode(text, form="NFC")
        assert isinstance(result, str)

    def test_normalize_unicode_nfd(self):
        text = "café"
        result = normalize_unicode(text, form="NFD")
        assert isinstance(result, str)


class TestBanglaDetection:
    def test_is_bangla_char(self):
        assert is_bangla_char("আ") is True
        assert is_bangla_char("a") is False
        assert is_bangla_char("1") is False

    def test_contains_bangla(self):
        assert contains_bangla("আমি") is True
        assert contains_bangla("Hello") is False
        assert contains_bangla("আমি Hello") is True

    def test_detect_language(self):
        assert detect_language("আমি স্কুলে যাই।") == "bn"
        assert detect_language("I go to school.") == "en"
        assert detect_language("আমি school যাই।") == "mixed"


class TestDigitNormalization:
    def test_bangla_to_arabic(self):
        text = "আমার বয়স ২৫ বছর।"
        result = normalize_bangla_digits(text, to_arabic=True)
        assert "25" in result
        assert "২৫" not in result

    def test_arabic_to_bangla(self):
        text = "আমার বয়স 25 বছর।"
        result = normalize_bangla_digits(text, to_arabic=False)
        assert "২৫" in result
        assert "25" not in result


class TestTextNormalization:
    def test_normalize_bangla_text(self):
        text = "আমি   স্কুলে যাচ্ছি।"
        result = normalize_text(text, lang="bn")
        assert "আমি স্কুলে যাচ্ছি" in result
        assert "  " not in result  # No double spaces

    def test_normalize_english_text(self):
        text = "I am   going to school."
        result = normalize_text(text, lang="en")
        assert "I am going to school." == result
        assert "  " not in result

    def test_normalize_empty_text(self):
        result = normalize_text("")
        assert result == ""

    def test_auto_detect_language(self):
        # Should auto-detect Bangla
        result = normalize_text("আমি স্কুলে যাই।")
        assert isinstance(result, str)

        # Should auto-detect English
        result = normalize_text("I go to school.")
        assert isinstance(result, str)


class TestSentenceSplitting:
    def test_split_bangla_sentences(self):
        text = "আমি স্কুলে যাই। আমি বই পড়ি।"
        sentences = split_sentences(text, lang="bn")
        assert len(sentences) == 2
        assert "আমি স্কুলে যাই" in sentences[0]
        assert "আমি বই পড়ি" in sentences[1]

    def test_split_english_sentences(self):
        text = "I go to school. I read books."
        sentences = split_sentences(text, lang="en")
        assert len(sentences) == 2
        assert "I go to school" in sentences[0]
        assert "I read books" in sentences[1]

    def test_split_empty_text(self):
        sentences = split_sentences("")
        assert len(sentences) == 0
