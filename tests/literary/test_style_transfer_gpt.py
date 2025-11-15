"""
Tests for Style Transfer GPT module.
"""

import pytest

from src.bilingual.modules.style_transfer_gpt import StyleTransferGPT


class TestStyleTransferGPT:
    """Test suite for Style Transfer GPT."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return StyleTransferGPT("dummy_model_path")

    def test_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.model_path == "dummy_model_path"
        assert model.model is None

    def test_load_model(self, model):
        """Test model loading."""
        model.load_model()
        assert model.model is not None

    def test_convert_without_loading_raises_error(self, model):
        """Test that converting without loading model raises error."""
        with pytest.raises(RuntimeError):
            model.convert("test text", "formal")

    def test_convert_to_formal(self, model):
        """Test style conversion to formal."""
        model.load_model()
        text = "আমি ভাত খাই"
        result = model.convert(text, "formal")

        assert isinstance(result, str)
        assert "[formal]" in result
        assert text in result

    def test_convert_to_informal(self, model):
        """Test style conversion to informal."""
        model.load_model()
        text = "আমি ভাত খাই"
        result = model.convert(text, "informal")

        assert isinstance(result, str)
        assert "[informal]" in result

    def test_convert_to_poetic(self, model):
        """Test style conversion to poetic."""
        model.load_model()
        text = "চাঁদ উঠেছে"
        result = model.convert(text, "poetic")

        assert isinstance(result, str)
        assert "[poetic]" in result

    def test_available_styles(self, model):
        """Test getting available styles."""
        styles = model.available_styles()

        assert isinstance(styles, list)
        assert len(styles) > 0
        assert "formal" in styles
        assert "informal" in styles
        assert "poetic" in styles
        assert "colloquial" in styles
