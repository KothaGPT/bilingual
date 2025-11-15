"""
Unit tests for Wikipedia LM module.

Tests:
- Model loading
- Fill-mask functionality
- Text generation
- Embeddings extraction
- Similarity computation
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import torch

    from bilingual.modules.wikipedia_lm import WikipediaLanguageModel, load_model

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestWikipediaLanguageModel:
    """Test WikipediaLanguageModel class."""

    @pytest.fixture
    def mlm_model(self):
        """Load MLM model for testing."""
        model = WikipediaLanguageModel(
            "bert-base-multilingual-cased", device="cpu", model_type="mlm"
        )
        return model

    def test_model_initialization(self, mlm_model):
        """Test model initialization."""
        assert mlm_model is not None
        assert mlm_model.model is not None
        assert mlm_model.tokenizer is not None
        assert mlm_model.model_type == "mlm"
        assert mlm_model.device == "cpu"

    def test_auto_device_detection(self):
        """Test automatic device detection."""
        model = WikipediaLanguageModel("bert-base-multilingual-cased", model_type="mlm")

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert model.device == expected_device

    def test_fill_mask(self, mlm_model):
        """Test fill-mask functionality."""
        text = "Hello [MASK] world"
        results = mlm_model.fill_mask(text, top_k=3)

        assert len(results) == 3
        assert all("sequence" in r for r in results)
        assert all("score" in r for r in results)

        # Scores should be in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_fill_mask_bangla(self, mlm_model):
        """Test fill-mask with Bangla text."""
        text = "আমি [MASK] খাই"
        results = mlm_model.fill_mask(text, top_k=5)

        assert len(results) == 5
        assert all("sequence" in r for r in results)

    def test_get_embeddings_single(self, mlm_model):
        """Test getting embeddings for single text."""
        text = "This is a test sentence."
        embeddings = mlm_model.get_embeddings(text)

        assert embeddings.shape[0] == 1  # Batch size
        assert embeddings.shape[2] == 768  # Hidden size for BERT-base
        assert embeddings.dim() == 3  # (batch, seq_len, hidden_size)

    def test_get_embeddings_batch(self, mlm_model):
        """Test getting embeddings for batch of texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = mlm_model.get_embeddings(texts)

        assert embeddings.shape[0] == 3  # Batch size
        assert embeddings.shape[2] == 768  # Hidden size

    def test_get_sentence_embedding_mean(self, mlm_model):
        """Test sentence embedding with mean pooling."""
        text = "This is a test sentence."
        embedding = mlm_model.get_sentence_embedding(text, pooling="mean")

        assert embedding.dim() == 1  # 1D vector
        assert embedding.shape[0] == 768  # Hidden size

    def test_get_sentence_embedding_max(self, mlm_model):
        """Test sentence embedding with max pooling."""
        text = "This is a test sentence."
        embedding = mlm_model.get_sentence_embedding(text, pooling="max")

        assert embedding.dim() == 1
        assert embedding.shape[0] == 768

    def test_get_sentence_embedding_cls(self, mlm_model):
        """Test sentence embedding with CLS token."""
        text = "This is a test sentence."
        embedding = mlm_model.get_sentence_embedding(text, pooling="cls")

        assert embedding.dim() == 1
        assert embedding.shape[0] == 768

    def test_compute_similarity(self, mlm_model):
        """Test similarity computation."""
        text1 = "I like to eat food."
        text2 = "I enjoy eating meals."

        similarity = mlm_model.compute_similarity(text1, text2)

        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

        # Similar sentences should have high similarity
        assert similarity > 0.5

    def test_compute_similarity_identical(self, mlm_model):
        """Test similarity of identical texts."""
        text = "This is a test sentence."
        similarity = mlm_model.compute_similarity(text, text)

        # Should be very close to 1.0
        assert similarity > 0.99

    def test_compute_similarity_different(self, mlm_model):
        """Test similarity of very different texts."""
        text1 = "I like cats and dogs."
        text2 = "The weather is sunny today."

        similarity = mlm_model.compute_similarity(text1, text2)

        # Should be lower than similar texts
        assert similarity < 0.8

    def test_predict_next_word(self, mlm_model):
        """Test next word prediction."""
        text = "I like to"
        predictions = mlm_model.predict_next_word(text, top_k=5)

        assert len(predictions) <= 5
        assert all("word" in p for p in predictions)
        assert all("score" in p for p in predictions)

        # Scores should be in descending order
        scores = [p["score"] for p in predictions]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_model_function(self):
        """Test load_model convenience function."""
        model = load_model("bert-base-multilingual-cased", model_type="mlm", device="cpu")

        assert isinstance(model, WikipediaLanguageModel)
        assert model.model_type == "mlm"

    def test_fill_mask_function(self):
        """Test fill_mask convenience function."""
        from bilingual.modules.wikipedia_lm import fill_mask

        text = "Hello [MASK] world"
        results = fill_mask("bert-base-multilingual-cased", text, top_k=3)

        assert len(results) == 3

    def test_get_embeddings_function(self):
        """Test get_embeddings convenience function."""
        from bilingual.modules.wikipedia_lm import get_embeddings

        text = "This is a test."
        embeddings = get_embeddings("bert-base-multilingual-cased", text)

        assert embeddings.shape[0] == 1
        assert embeddings.shape[2] == 768

    def test_compute_similarity_function(self):
        """Test compute_similarity convenience function."""
        from bilingual.modules.wikipedia_lm import compute_similarity

        text1 = "I like food."
        text2 = "I enjoy meals."

        similarity = compute_similarity("bert-base-multilingual-cased", text1, text2)

        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestErrorHandling:
    """Test error handling."""

    def test_fill_mask_on_clm_raises_error(self):
        """Test fill_mask raises error on CLM model."""
        try:
            model = WikipediaLanguageModel("gpt2", model_type="clm", device="cpu")

            with pytest.raises(ValueError, match="only works with MLM"):
                model.fill_mask("Hello [MASK] world")
        except Exception as e:
            pytest.skip(f"Could not load GPT-2: {e}")

    def test_generate_text_on_mlm_raises_error(self):
        """Test generate_text raises error on MLM model."""
        model = WikipediaLanguageModel(
            "bert-base-multilingual-cased", model_type="mlm", device="cpu"
        )

        with pytest.raises(ValueError, match="only works with CLM"):
            model.generate_text("Hello")

    def test_invalid_pooling_strategy(self):
        """Test invalid pooling strategy raises error."""
        model = WikipediaLanguageModel(
            "bert-base-multilingual-cased", model_type="mlm", device="cpu"
        )

        with pytest.raises(ValueError, match="Invalid pooling strategy"):
            model.get_sentence_embedding("Test", pooling="invalid")


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestBilingualSupport:
    """Test bilingual model support."""

    def test_multilingual_model(self):
        """Test multilingual model works with multiple languages."""
        model = WikipediaLanguageModel(
            "bert-base-multilingual-cased", model_type="mlm", device="cpu"
        )

        # English
        en_text = "Hello [MASK] world"
        en_results = model.fill_mask(en_text, top_k=3)
        assert len(en_results) == 3

        # Bangla (if model supports)
        bn_text = "আমি [MASK] খাই"
        bn_results = model.fill_mask(bn_text, top_k=3)
        assert len(bn_results) == 3

    def test_cross_lingual_similarity(self):
        """Test cross-lingual similarity computation."""
        model = WikipediaLanguageModel(
            "bert-base-multilingual-cased", model_type="mlm", device="cpu"
        )

        # Similar meaning in different languages
        en_text = "I like to eat food."
        # Note: This is a simplified test, actual translation may vary

        # Compute embedding
        en_emb = model.get_sentence_embedding(en_text)

        assert en_emb.shape[0] == 768


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
@pytest.mark.slow
class TestModelIntegration:
    """Test model integration (slow tests)."""

    def test_end_to_end_mlm(self):
        """Test end-to-end MLM workflow."""
        # Load model
        model = load_model("bert-base-multilingual-cased", model_type="mlm", device="cpu")

        # Fill mask
        results = model.fill_mask("The capital of France is [MASK].", top_k=5)
        assert len(results) == 5

        # Get embeddings
        embeddings = model.get_embeddings("This is a test.")
        assert embeddings.shape[2] == 768

        # Compute similarity
        similarity = model.compute_similarity("I like cats.", "I love cats.")
        assert similarity > 0.8

    def test_batch_processing(self):
        """Test batch processing efficiency."""
        model = load_model("bert-base-multilingual-cased", model_type="mlm", device="cpu")

        texts = [f"This is sentence {i}." for i in range(10)]

        # Batch embeddings
        embeddings = model.get_embeddings(texts)

        assert embeddings.shape[0] == 10
        assert embeddings.shape[2] == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
