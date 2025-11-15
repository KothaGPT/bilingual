"""
Tests for Cross-Lingual Embeddings.
"""

import numpy as np
import pytest

from bilingual.models.cross_lingual_embeddings import CrossLingualEmbeddings


class TestCrossLingualEmbeddings:
    """Test suite for Cross-Lingual Embeddings."""

    @pytest.fixture
    def embeddings(self):
        """Create a test embeddings instance."""
        return CrossLingualEmbeddings(pooling_mode="mean")

    def test_embeddings_initialization(self, embeddings):
        """Test embeddings initialization."""
        assert embeddings is not None
        assert embeddings.device in ["cuda", "cpu"]
        assert embeddings.pooling_mode == "mean"

    def test_pooling_modes(self):
        """Test different pooling modes."""
        for mode in ["mean", "max", "cls"]:
            emb = CrossLingualEmbeddings(pooling_mode=mode)
            assert emb.pooling_mode == mode

    def test_encode_without_model(self, embeddings):
        """Test encoding without pretrained model."""
        if embeddings.model is None:
            with pytest.raises(ValueError, match="Model not initialized"):
                embeddings.encode("Hello world")

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_encode_single_sentence(self, embeddings):
        """Test encoding single sentence."""
        sentence = "আমি বাংলায় কথা বলি"
        embedding = embeddings.encode(sentence)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] > 0

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_encode_multiple_sentences(self, embeddings):
        """Test encoding multiple sentences."""
        sentences = [
            "আমি বাংলায় কথা বলি",
            "I speak in Bengali",
            "এটি একটি পরীক্ষা",
        ]

        embedding = embeddings.encode(sentences)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 3

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_encode_normalization(self, embeddings):
        """Test embedding normalization."""
        sentence = "Test sentence"

        # Normalized
        emb_norm = embeddings.encode(sentence, normalize=True)
        norm = np.linalg.norm(emb_norm[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

        # Not normalized
        emb_no_norm = embeddings.encode(sentence, normalize=False)
        norm_no = np.linalg.norm(emb_no_norm[0])
        assert not np.isclose(norm_no, 1.0, atol=1e-5)

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_similarity(self, embeddings):
        """Test similarity calculation."""
        sent1 = "আমি বাংলায় কথা বলি"
        sent2 = "I speak in Bengali"

        similarity = embeddings.similarity(sent1, sent2)

        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_find_similar(self, embeddings):
        """Test finding similar sentences."""
        query = "বাংলা ভাষা"
        candidates = [
            "Bengali language",
            "English language",
            "বাংলা সাহিত্য",
            "Computer science",
        ]

        results = embeddings.find_similar(query, candidates, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_translate_search(self, embeddings):
        """Test cross-lingual search."""
        query = "রবীন্দ্রনাথ ঠাকুর"
        targets = [
            "Rabindranath Tagore",
            "William Shakespeare",
            "Leo Tolstoy",
        ]

        results = embeddings.translate_search(query, targets, top_k=1)

        assert len(results) == 1
        assert "Tagore" in results[0][0]

    @pytest.mark.skipif(True, reason="Requires pretrained model")
    def test_align_sentences(self, embeddings):
        """Test sentence alignment."""
        sentences_bn = [
            "আমি বাংলায় কথা বলি",
            "এটি একটি পরীক্ষা",
        ]
        sentences_en = [
            "I speak in Bengali",
            "This is a test",
        ]

        alignments = embeddings.align_sentences(sentences_bn, sentences_en, threshold=0.5)

        assert isinstance(alignments, list)
        assert all(len(a) == 3 for a in alignments)

    def test_batch_encoding(self, embeddings):
        """Test batch encoding with different batch sizes."""
        if embeddings.model is None:
            pytest.skip("Model not available")

        sentences = ["Test"] * 10

        emb_small = embeddings.encode(sentences, batch_size=2)
        emb_large = embeddings.encode(sentences, batch_size=5)

        assert emb_small.shape == emb_large.shape
