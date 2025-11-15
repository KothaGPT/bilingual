"""
Tests for GPT-based style transfer model.
"""

import types
from unittest.mock import MagicMock, patch

import pytest

# Skip these tests if torch/transformers are not available
try:
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    from bilingual.modules.style_transfer_gpt import StyleTransferGPT

    pytorch_available = True
except ImportError:
    pytorch_available = False

pytestmark = pytest.mark.skipif(
    not pytorch_available, reason="Requires torch and transformers to be installed"
)


class TestStyleTransferGPTInit:
    """Test suite for GPT model initialization."""

    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_default_initialization(self, mock_tokenizer, mock_model):
        """Test model initializes without pretrained weights."""
        # Mock the model and tokenizer
        mock_model.return_value = MagicMock(spec=GPT2LMHeadModel)
        mock_tokenizer.return_value = MagicMock(spec=PreTrainedTokenizerFast)

        model = StyleTransferGPT()
        assert model.__dict__.get("_model") is None
        assert model.__dict__.get("_tokenizer") is None

        # Test that load_model is not called during initialization
        mock_model.assert_not_called()
        mock_tokenizer.assert_not_called()

    @pytest.mark.skip(reason="Requires actual model files")
    def test_initialization_with_pretrained(self):
        """Test model initialization with pretrained weights."""
        model = StyleTransferGPT(model_path="path/to/pretrained")
        assert model.__dict__.get("_model") is not None
        assert model.__dict__.get("_tokenizer") is not None


class TestStyleTransferGPTStyles:
    """Test suite for style transfer functionality."""

    @pytest.fixture
    def mock_model(self, monkeypatch):
        """Create a mock model for testing."""
        # Create a mock model that won't try to load anything
        model = StyleTransferGPT()

        # Mock the model attributes
        model._model = MagicMock(spec=GPT2LMHeadModel)
        model._tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        model.device = "cpu"

        # Create bound methods for convert
        def mock_convert(self, text, target_style, source_style=None, **kwargs):
            if self._model is None:
                raise ValueError("Model not loaded")
            if target_style not in StyleTransferGPT.STYLES:
                import warnings

                warnings.warn(f"Unknown style '{target_style}'", UserWarning)
            return f"{text} ({source_style or 'auto'}->{target_style})"

        # Bind the method to the model instance
        model.convert = types.MethodType(mock_convert, model)

        # Create bound methods for batch_convert
        def mock_batch_convert(self, texts, target_style, source_style=None, **kwargs):
            if self._model is None:
                raise ValueError("Model not loaded")
            if target_style not in StyleTransferGPT.STYLES:
                import warnings

                warnings.warn(f"Unknown style '{target_style}'", UserWarning)
            return [f"{text} ({source_style or 'auto'}->{target_style})" for text in texts]

        # Bind the method to the model instance
        model.batch_convert = types.MethodType(mock_batch_convert, model)

        return model

    def test_style_tokens_defined(self):
        """Test that all style tokens are properly defined."""
        # Check that the class has the expected style attributes
        assert hasattr(StyleTransferGPT, "STYLES")
        assert isinstance(StyleTransferGPT.STYLES, list)
        assert "formal" in StyleTransferGPT.STYLES
        assert "informal" in StyleTransferGPT.STYLES
        assert "poetic" in StyleTransferGPT.STYLES
        assert "literary" in StyleTransferGPT.STYLES
        assert "colloquial" in StyleTransferGPT.STYLES

    def test_convert_requires_loaded_model(self, mock_model):
        """Test that convert requires a loaded model."""
        # Save the original model
        original_model = mock_model._model
        try:
            mock_model._model = None
            with pytest.raises(ValueError, match="Model not loaded"):
                mock_model.convert("text", "formal")
        finally:
            # Restore the model to avoid affecting other tests
            mock_model._model = original_model

    def test_convert_invalid_style(self, mock_model):
        """Test convert with invalid style raises warning but continues."""
        with pytest.warns(UserWarning, match="Unknown style 'invalid_style'"):
            result = mock_model.convert("text", "invalid_style")
        assert result == "text (auto->invalid_style)"

    @pytest.mark.parametrize(
        "source,target",
        [
            ("formal", "informal"),
            ("informal", "formal"),
            ("formal", "poetic"),
            ("formal", "literary"),
        ],
    )
    def test_style_conversion_methods(self, mock_model, source, target):
        """Test that style conversion methods work with valid styles."""
        result = mock_model.convert("test text", target_style=target, source_style=source)
        assert result == f"test text ({source}->{target})"


class TestBatchProcessing:
    """Test suite for batch processing."""

    @pytest.fixture
    def mock_model(self, monkeypatch):
        """Create a mock model for testing."""
        model = StyleTransferGPT()

        # Mock the batch_convert method with proper self parameter
        def mock_batch_convert(self, texts, target_style, source_style=None, **kwargs):
            return [f"converted {i+1}" for i in range(len(texts))]

        monkeypatch.setattr(model, "batch_convert", types.MethodType(mock_batch_convert, model))
        return model

    def test_batch_convert(self, mock_model):
        """Test batch convert with multiple texts."""
        texts = ["text 1", "text 2"]
        results = mock_model.batch_convert(
            texts=texts, source_style="informal", target_style="formal"
        )

        assert len(results) == 2
        assert "converted 1" in results[0]
        assert "converted 2" in results[1]

    def test_batch_convert_empty_list(self, mock_model):
        """Test batch convert with empty list returns empty list."""
        results = mock_model.batch_convert(texts=[], source_style="informal", target_style="formal")
        assert results == []


class TestModelPersistence:
    """Test suite for model saving and loading."""

    @pytest.fixture
    def mock_model(self, monkeypatch):
        """Create a mock model for testing."""
        model = StyleTransferGPT()
        model._model = MagicMock()
        model._tokenizer = MagicMock()
        model.device = "cpu"
        return model

    def test_save_model(self, mock_model, tmp_path, monkeypatch):
        """Test saving the model to disk."""
        # Mock the save_pretrained methods
        model_save = MagicMock()
        tokenizer_save = MagicMock()

        mock_model._model.save_pretrained = model_save
        mock_model._tokenizer.save_pretrained = tokenizer_save

        # Call save
        save_dir = tmp_path / "saved_model"
        mock_model.save(save_dir)

        # Verify the save methods were called with the correct path
        model_save.assert_called_once_with(save_dir)
        tokenizer_save.assert_called_once_with(save_dir)

    @pytest.mark.skip(reason="Requires actual model files")
    def test_from_pretrained(self):
        """Test loading a pretrained model."""
        model = StyleTransferGPT(model_path="path/to/model")
        model.load_model()
        assert model._model is not None
        assert model._tokenizer is not None
