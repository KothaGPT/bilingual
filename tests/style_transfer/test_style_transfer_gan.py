"""
Tests for GAN-based style transfer model.
"""

import pytest

# Skip these tests if torch is not available
try:
    import torch

    from bilingual.modules.style_transfer_gan import StyleTransferGAN

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="Requires PyTorch to be installed")


class TestStyleTransferGANInit:
    """Test suite for GAN model initialization."""

    def test_default_initialization(self):
        """Test model initializes without pretrained weights."""
        model = StyleTransferGAN()
        assert model.generator is None
        assert model.discriminator is None
        assert model.device == "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skip(reason="Requires actual model files")
    def test_initialization_with_pretrained(self):
        """Test model initialization with pretrained weights."""
        model = StyleTransferGAN(
            generator_path="path/to/generator", discriminator_path="path/to/discriminator"
        )
        assert model.generator is not None
        assert model.discriminator is not None


class TestStyleTransferGANStyles:
    """Test suite for GAN style transfer functionality."""

    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model for testing."""
        model = StyleTransferGAN()
        model.generator = mocker.MagicMock()
        model.discriminator = mocker.MagicMock()
        model.device = "cpu"
        return model

    def test_available_styles(self):
        """Test that all expected styles are available."""
        assert hasattr(StyleTransferGAN, "AVAILABLE_STYLES")
        assert isinstance(StyleTransferGAN.AVAILABLE_STYLES, list)
        assert "formal" in StyleTransferGAN.AVAILABLE_STYLES
        assert "informal" in StyleTransferGAN.AVAILABLE_STYLES
        assert "poetic" in StyleTransferGAN.AVAILABLE_STYLES

    def test_transfer_requires_loaded_model(self, mock_model):
        """Test that transfer requires a loaded generator."""
        mock_model.generator = None
        with pytest.raises(ValueError, match="Generator not loaded"):
            mock_model.transfer("text", "source", "target")

    def test_transfer_invalid_style(self, mock_model):
        """Test transfer with invalid style raises error."""
        with pytest.raises(ValueError, match="Invalid target style"):
            mock_model.transfer("text", "source", "invalid_style")

    def test_style_conversion(self, mock_model):
        """Test style conversion with mock generator."""
        # Mock the generator to return a fixed output
        mock_output = torch.tensor([[1.0, 2.0, 3.0]])
        mock_model.generator.return_value = mock_output
        mock_model.tokenizer = mocker.MagicMock()
        mock_model.tokenizer.decode.return_value = "transformed text"

        result = mock_model.transfer("test", "formal", "informal")
        assert result == "transformed text"
        mock_model.generator.assert_called_once()


class TestGANModelPersistence:
    """Test suite for GAN model saving and loading."""

    def test_save_model(self, tmp_path, mocker):
        """Test saving the GAN model to disk."""
        model = StyleTransferGAN()
        model.generator = mocker.MagicMock()
        model.discriminator = mocker.MagicMock()

        save_dir = tmp_path / "saved_gan"
        model.save(save_dir)

        # Check that save was called with the correct path
        model.generator.save_pretrained.assert_called_once()
        model.discriminator.save_pretrained.assert_called_once()

    @pytest.mark.skip(reason="Requires actual model files")
    def test_from_pretrained(self):
        """Test loading a pretrained GAN model."""
        model = StyleTransferGAN.from_pretrained(
            generator_path="path/to/generator", discriminator_path="path/to/discriminator"
        )
        assert model.generator is not None
        assert model.discriminator is not None


class TestGANBatchProcessing:
    """Test suite for batch processing with GAN."""

    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model for testing batch processing."""
        model = StyleTransferGAN()
        model.transfer = lambda text, s, t: f"{text} ({s}->{t})"
        return model

    def test_batch_transfer(self, mock_model):
        """Test batch transfer with multiple texts."""
        texts = ["First text", "Second text"]
        results = mock_model.batch_transfer(texts, "formal", "informal")

        assert len(results) == len(texts)
        for i, text in enumerate(texts):
            assert results[i] == f"{text} (formal->informal)"

    def test_batch_empty_list(self, mock_model):
        """Test batch transfer with empty list returns empty list."""
        assert mock_model.batch_transfer([], "formal", "informal") == []


class TestGANTraining:
    """Test suite for GAN training functionality."""

    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model with training components."""
        model = StyleTransferGAN()
        model.generator = mocker.MagicMock()
        model.discriminator = mocker.MagicMock()
        model.optimizer_g = mocker.MagicMock()
        model.optimizer_d = mocker.MagicMock()
        model.criterion_gan = mocker.MagicMock()
        model.criterion_cycle = mocker.MagicMock()
        model.criterion_identity = mocker.MagicMock()
        model.device = "cpu"
        return model

    def test_train_step(self, mock_model):
        """Test a single training step."""
        # Mock data
        real_data = ["sample text"]
        style_labels = ["formal"]

        # Mock forward/backward passes
        mock_model.forward = mocker.MagicMock(
            return_value={
                "fake": torch.tensor([[1.0]]),
                "rec": torch.tensor([[1.0]]),
                "identity": torch.tensor([[1.0]]),
            }
        )

        # Run training step
        loss = mock_model.train_step(real_data, style_labels)

        # Check that optimizers were stepped
        mock_model.optimizer_g.step.assert_called_once()
        mock_model.optimizer_d.step.assert_called_once()

        # Check that loss is returned
        assert isinstance(loss, dict)
        assert "g_loss" in loss
        assert "d_loss" in loss
