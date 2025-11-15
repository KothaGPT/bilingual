"""
Tests for style transfer model.
"""

import pytest

from bilingual.modules.style_transfer_gan import StyleTransferModel


class TestStyleTransferModelInit:
    """Test suite for model initialization."""

    def test_default_initialization(self):
        """Test model initializes with default parameters."""
        model = StyleTransferModel()

        assert model.model_type == "rule_based"
        assert model.loaded is False

    def test_transformer_initialization(self):
        """Test model initializes with transformer type."""
        model = StyleTransferModel(model_type="transformer")

        assert model.model_type == "transformer"
        assert model.loaded is False

    def test_gan_initialization(self):
        """Test model initializes with GAN type."""
        model = StyleTransferModel(model_type="gan")

        assert model.model_type == "gan"
        assert model.loaded is False

    def test_repr(self):
        """Test string representation of model."""
        model = StyleTransferModel()
        repr_str = repr(model)

        assert "StyleTransferModel" in repr_str
        assert "rule_based" in repr_str
        assert "loaded=False" in repr_str


class TestStyleTransferModelLoad:
    """Test suite for model loading."""

    def test_load_rule_based(self):
        """Test loading rule-based model."""
        model = StyleTransferModel(model_type="rule_based")
        model.load()

        assert model.loaded is True

    def test_load_with_path(self):
        """Test loading with model path (no-op for rule-based)."""
        model = StyleTransferModel(model_type="rule_based")
        model.load(model_path="/fake/path")

        assert model.loaded is True

    def test_load_transformer_placeholder(self):
        """Test loading transformer model (placeholder)."""
        model = StyleTransferModel(model_type="transformer")
        model.load()

        assert model.loaded is True


class TestStyleTransferConvert:
    """Test suite for style conversion."""

    def test_convert_to_formal_english(self):
        """Test conversion to formal style in English."""
        model = StyleTransferModel()
        model.load()

        text = "I can't do this"
        result = model.convert(text, target_style="formal")

        assert "cannot" in result
        assert "can't" not in result

    def test_convert_to_informal_english(self):
        """Test conversion to informal style in English."""
        model = StyleTransferModel()
        model.load()

        text = "I cannot do this"
        result = model.convert(text, target_style="informal")

        assert "can't" in result
        assert "cannot" not in result

    def test_convert_to_poetic_english(self):
        """Test conversion to poetic style in English."""
        model = StyleTransferModel()
        model.load()

        text = "the night"
        result = model.convert(text, target_style="poetic")

        # Should add poetic flourishes
        assert len(result) > len(text)
        assert "night" in result

    def test_convert_to_formal_bengali(self):
        """Test conversion to formal style in Bengali."""
        model = StyleTransferModel()
        model.load()

        text = "তুমি কেমন আছো"
        result = model.convert(text, target_style="formal")

        # Should convert তুমি to আপনি
        assert "আপনি" in result

    def test_convert_to_informal_bengali(self):
        """Test conversion to informal style in Bengali."""
        model = StyleTransferModel()
        model.load()

        text = "আপনি কেমন আছেন"
        result = model.convert(text, target_style="informal")

        # Should convert আপনি to তুমি
        assert "তুমি" in result

    def test_convert_without_load(self):
        """Test that convert auto-loads if not loaded."""
        model = StyleTransferModel()

        text = "I can't do this"
        result = model.convert(text, target_style="formal")

        assert model.loaded is True
        assert "cannot" in result

    def test_convert_preserves_meaning_flag(self):
        """Test preserve_meaning parameter is accepted."""
        model = StyleTransferModel()
        model.load()

        text = "Test text"
        result = model.convert(text, target_style="formal", preserve_meaning=True)

        # Should not raise error
        assert isinstance(result, str)

    def test_convert_unknown_style(self):
        """Test conversion with unknown style returns original text."""
        model = StyleTransferModel()
        model.load()

        text = "Test text"
        result = model.convert(text, target_style="unknown")  # type: ignore

        # Should return original text unchanged
        assert result == text

    def test_convert_empty_text(self):
        """Test conversion of empty text."""
        model = StyleTransferModel()
        model.load()

        result = model.convert("", target_style="formal")
        assert result == ""

    def test_convert_deterministic(self):
        """Test that conversion is deterministic."""
        model = StyleTransferModel()
        model.load()

        text = "I can't believe it's true"
        result1 = model.convert(text, target_style="formal")
        result2 = model.convert(text, target_style="formal")

        assert result1 == result2


class TestStyleTransferBatchConvert:
    """Test suite for batch conversion."""

    def test_batch_convert_multiple_texts(self):
        """Test batch conversion of multiple texts."""
        model = StyleTransferModel()
        model.load()
        texts = ["This is a test.", "Another test here.", "It's fine"]
        results = model.batch_convert(texts=texts, target_style="formal")

        assert len(results) == 3
        assert isinstance(results[0], str)
        assert isinstance(results[1], str)
        assert isinstance(results[2], str)

    def test_batch_convert_empty_list(self):
        """Test batch conversion of empty list."""
        model = StyleTransferModel()
        model.load()

        results = model.batch_convert(texts=[], target_style="formal")
        assert results == []

    def test_batch_convert_single_text(self):
        """Test batch conversion with single text."""
        model = StyleTransferModel()
        model.load()

        texts = ["I can't do this"]
        results = model.batch_convert(texts=texts, target_style="formal")

        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_batch_convert_mixed_languages(self):
        """Test batch conversion with mixed Bengali and English."""
        model = StyleTransferModel()
        model.load()

        texts = ["I can't do this", "তুমি কেমন আছো"]
        results = model.batch_convert(texts=texts, target_style="formal")

        assert len(results) == 2
        assert isinstance(results[0], str)
        assert isinstance(results[1], str)


class TestStyleTransferAvailableStyles:
    """Test suite for available styles query."""

    def test_available_styles_returns_list(self):
        """Test that available_styles returns a list."""
        model = StyleTransferModel()
        styles = model.available_styles()

        assert isinstance(styles, list)
        assert len(styles) > 0

    def test_available_styles_contains_expected(self):
        """Test that available styles contain expected values."""
        model = StyleTransferModel()
        styles = model.available_styles()

        assert "formal" in styles
        assert "informal" in styles
        assert "poetic" in styles

    def test_available_styles_count(self):
        """Test that correct number of styles are available."""
        model = StyleTransferModel()
        styles = model.available_styles()

        assert len(styles) == 3


class TestStyleTransferIntegration:
    """Integration tests for style transfer module."""

    def test_full_workflow_formal(self):
        """Test complete workflow: init -> load -> convert to formal."""
        model = StyleTransferModel()
        model.load()

        text = "I can't believe it's not working"
        result = model.convert(text, target_style="formal")

        assert "cannot" in result
        assert "it is" in result

    def test_full_workflow_informal(self):
        """Test complete workflow: init -> load -> convert to informal."""
        model = StyleTransferModel()
        model.load()

        text = "I cannot believe it is not working"
        result = model.convert(text, target_style="informal")

        assert "can't" in result
        assert "it's" in result

    def test_full_workflow_poetic(self):
        """Test complete workflow: init -> load -> convert to poetic."""
        model = StyleTransferModel()
        model.load()

        text = "the night was dark"
        result = model.convert(text, target_style="poetic")

        # Should add poetic elements
        assert len(result) >= len(text)

    def test_multiple_conversions_same_model(self):
        """Test multiple conversions with same model instance."""
        model = StyleTransferModel()
        model.load()

        text1 = "I can't do this"
        text2 = "I won't go there"

        result1 = model.convert(text1, target_style="formal")
        result2 = model.convert(text2, target_style="formal")

        assert "cannot" in result1
        assert "will not" in result2

    def test_switch_styles_same_text(self):
        """Test converting same text to different styles."""
        model = StyleTransferModel()
        model.load()

        text = "I cannot do this"

        formal = model.convert(text, target_style="formal")
        informal = model.convert(text, target_style="informal")

        # Formal should keep "cannot", informal should change to "can't"
        assert "cannot" in formal
        assert "can't" in informal

    def test_bilingual_text_conversion(self):
        """Test conversion of text with both Bengali and English."""
        model = StyleTransferModel()
        model.load()

        # Mixed text
        text = "I can't speak তুমি বুঝতে পারছো"
        result = model.convert(text, target_style="formal")

        # Should convert both languages
        assert "cannot" in result
        assert "আপনি" in result

    def test_model_state_persistence(self):
        """Test that model state persists across operations."""
        model = StyleTransferModel()

        assert model.loaded is False

        model.load()
        assert model.loaded is True

        # Should remain loaded after conversion
        model.convert("test", target_style="formal")
        assert model.loaded is True
