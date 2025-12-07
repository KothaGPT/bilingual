"""
Tests for Literary Language Model.
"""

import pytest
torch = pytest.importorskip("torch")

from bilingual.models.literary_lm import LiteraryLanguageModel


class TestLiteraryLanguageModel:
    """Test suite for Literary Language Model."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return LiteraryLanguageModel()

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.device in ["cuda", "cpu"]
        assert model.config is not None

    def test_generate_with_prompt(self, model):
        """Test text generation with prompt."""
        if not model.tokenizer:
            pytest.skip("Tokenizer not available")

        prompt = "একদিন"
        generated = model.generate(
            prompt=prompt,
            max_length=50,
            num_return_sequences=1,
        )

        assert len(generated) == 1
        assert isinstance(generated[0], str)
        assert len(generated[0]) > 0

    def test_generate_poetry(self, model):
        """Test poetry generation."""
        if not model.tokenizer:
            pytest.skip("Tokenizer not available")

        prompt = "চাঁদের আলো"
        poetry = model.generate_poetry(
            prompt=prompt,
            meter_type="অক্ষরবৃত্ত",
            max_length=100,
        )

        assert isinstance(poetry, str)
        assert len(poetry) > 0

    def test_complete_text(self, model):
        """Test text completion."""
        if not model.tokenizer:
            pytest.skip("Tokenizer not available")

        text = "রবীন্দ্রনাথ ঠাকুর ছিলেন"
        completed = model.complete_text(
            text=text,
            max_new_tokens=50,
        )

        assert isinstance(completed, str)
        assert text in completed or len(completed) > len(text)

    def test_get_perplexity(self, model):
        """Test perplexity calculation."""
        if not model.tokenizer:
            pytest.skip("Tokenizer not available")

        text = "এটি একটি সাধারণ বাক্য।"
        perplexity = model.get_perplexity(text)

        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_generate_multiple_sequences(self, model):
        """Test generating multiple sequences."""
        if not model.tokenizer:
            pytest.skip("Tokenizer not available")

        prompt = "বাংলা সাহিত্য"
        generated = model.generate(
            prompt=prompt,
            max_length=50,
            num_return_sequences=3,
        )

        assert len(generated) == 3
        assert all(isinstance(text, str) for text in generated)

    def test_temperature_effect(self, model):
        """Test temperature parameter effect."""
        if not model.tokenizer:
            pytest.skip("Tokenizer not available")

        prompt = "কবিতা"

        # Low temperature (more deterministic)
        gen_low = model.generate(prompt, temperature=0.3, num_return_sequences=2)

        # High temperature (more random)
        gen_high = model.generate(prompt, temperature=1.5, num_return_sequences=2)

        assert len(gen_low) == 2
        assert len(gen_high) == 2
