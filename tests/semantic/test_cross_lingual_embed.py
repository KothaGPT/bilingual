"""
Tests for Cross-lingual Embeddings module.
"""

import pytest

from src.bilingual.modules.cross_lingual_embed import (
    align_sentences,
    compute_similarity,
    embed_text,
)


class TestCrossLingualEmbed:
    """Test suite for cross-lingual embeddings."""

    def test_embed_single_text(self):
        """Test embedding a single text string."""
        text = "এটি একটি পরীক্ষা"
        embedding = embed_text(text, lang="bn")

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        texts = ["প্রথম বাক্য", "দ্বিতীয় বাক্য", "তৃতীয় বাক্য"]
        embeddings = embed_text(texts, lang="bn")

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)

    def test_embed_english_text(self):
        """Test embedding English text."""
        text = "This is a test"
        embedding = embed_text(text, lang="en")

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_compute_similarity_same_language(self):
        """Test similarity computation for same language."""
        text1 = "আমি ভাত খাই"
        text2 = "আমি খাবার খাই"

        similarity = compute_similarity(text1, text2, lang1="bn", lang2="bn")

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    def test_compute_similarity_cross_lingual(self):
        """Test similarity computation across languages."""
        text_bn = "আমি ভাত খাই"
        text_en = "I eat rice"

        similarity = compute_similarity(text_bn, text_en, lang1="bn", lang2="en")

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    def test_align_sentences_empty(self):
        """Test sentence alignment with empty lists."""
        alignments = align_sentences([], [])

        assert isinstance(alignments, list)
        assert len(alignments) == 0

    def test_align_sentences_basic(self):
        """Test basic sentence alignment."""
        source = ["আমি ভাত খাই", "তুমি কেমন আছ"]
        target = ["I eat rice", "How are you"]

        alignments = align_sentences(source, target, source_lang="bn", target_lang="en")

        assert isinstance(alignments, list)
