"""
Unit tests for model preparation for Hugging Face Hub.

Tests:
- Model validation
- File copying
- Metadata generation
- .gitattributes creation
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scripts.huggingface.prepare_model import ModelPreparer


class TestModelValidation:
    """Test model validation."""

    @pytest.fixture
    def valid_model_dir(self):
        """Create a valid model directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create required files
            (tmpdir / "config.json").write_text("{}")
            (tmpdir / "pytorch_model.bin").write_text("fake model")
            (tmpdir / "tokenizer_config.json").write_text("{}")

            yield tmpdir

    @pytest.fixture
    def invalid_model_dir(self):
        """Create an invalid model directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Missing required files
            yield tmpdir

    def test_validate_valid_model(self, valid_model_dir):
        """Test validation of valid model."""
        with tempfile.TemporaryDirectory() as output_dir:
            preparer = ModelPreparer(valid_model_dir, Path(output_dir))
            assert preparer.validate_model() is True

    def test_validate_invalid_model(self, invalid_model_dir):
        """Test validation of invalid model."""
        with tempfile.TemporaryDirectory() as output_dir:
            preparer = ModelPreparer(invalid_model_dir, Path(output_dir))
            assert preparer.validate_model() is False

    def test_validate_with_safetensors(self):
        """Test validation with SafeTensors format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create model with SafeTensors
            (tmpdir / "config.json").write_text("{}")
            (tmpdir / "model.safetensors").write_text("fake model")

            with tempfile.TemporaryDirectory() as output_dir:
                preparer = ModelPreparer(tmpdir, Path(output_dir))
                assert preparer.validate_model() is True


class TestFileCopying:
    """Test file copying."""

    @pytest.fixture
    def model_with_artifacts(self):
        """Create model with training artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Required files
            (tmpdir / "config.json").write_text("{}")
            (tmpdir / "pytorch_model.bin").write_text("model")

            # Training artifacts (should be excluded)
            (tmpdir / "optimizer.pt").write_text("optimizer")
            (tmpdir / "scheduler.pt").write_text("scheduler")
            (tmpdir / "trainer_state.json").write_text("{}")

            # Checkpoint directory (should be excluded)
            checkpoint_dir = tmpdir / "checkpoint-1000"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "pytorch_model.bin").write_text("checkpoint")

            yield tmpdir

    def test_copy_excludes_artifacts(self, model_with_artifacts):
        """Test that training artifacts are excluded."""
        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)

            preparer = ModelPreparer(model_with_artifacts, output_dir)
            preparer.copy_model_files()

            # Required files should be copied
            assert (output_dir / "config.json").exists()
            assert (output_dir / "pytorch_model.bin").exists()

            # Artifacts should not be copied
            assert not (output_dir / "optimizer.pt").exists()
            assert not (output_dir / "scheduler.pt").exists()
            assert not (output_dir / "trainer_state.json").exists()
            assert not (output_dir / "checkpoint-1000").exists()

    def test_copy_preserves_structure(self):
        """Test that directory structure is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create nested structure
            (tmpdir / "config.json").write_text("{}")
            (tmpdir / "pytorch_model.bin").write_text("model")

            subdir = tmpdir / "tokenizer"
            subdir.mkdir()
            (subdir / "vocab.txt").write_text("vocab")

            with tempfile.TemporaryDirectory() as output_dir:
                output_dir = Path(output_dir)

                preparer = ModelPreparer(tmpdir, output_dir)
                preparer.copy_model_files()

                # Check structure
                assert (output_dir / "config.json").exists()
                assert (output_dir / "pytorch_model.bin").exists()
                assert (output_dir / "tokenizer" / "vocab.txt").exists()


class TestMetadataGeneration:
    """Test metadata generation."""

    def test_generate_metadata(self):
        """Test metadata generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create model with config
            config = {
                "model_type": "bert",
                "architectures": ["BertForMaskedLM"],
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 30000,
            }

            (tmpdir / "config.json").write_text(json.dumps(config))
            (tmpdir / "pytorch_model.bin").write_text("x" * 1024 * 1024)  # 1 MB

            with tempfile.TemporaryDirectory() as output_dir:
                output_dir = Path(output_dir)

                preparer = ModelPreparer(tmpdir, output_dir)
                preparer.copy_model_files()
                metadata = preparer.generate_metadata()

                # Check metadata
                assert metadata["model_type"] == "bert"
                assert metadata["architecture"] == "BertForMaskedLM"
                assert metadata["hidden_size"] == 768
                assert metadata["num_layers"] == 12
                assert metadata["vocab_size"] == 30000
                assert metadata["model_size_mb"] > 0

                # Check metadata file
                metadata_path = output_dir / "model_metadata.json"
                assert metadata_path.exists()

                with open(metadata_path, "r") as f:
                    saved_metadata = json.load(f)
                    assert saved_metadata == metadata

    def test_metadata_parameter_estimation(self):
        """Test parameter count estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            config = {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "vocab_size": 30000,
            }

            (tmpdir / "config.json").write_text(json.dumps(config))
            (tmpdir / "pytorch_model.bin").write_text("model")

            with tempfile.TemporaryDirectory() as output_dir:
                output_dir = Path(output_dir)

                preparer = ModelPreparer(tmpdir, output_dir)
                preparer.copy_model_files()
                metadata = preparer.generate_metadata()

                # Should have estimated parameters
                assert metadata["num_parameters"] > 0
                assert isinstance(metadata["num_parameters"], int)


class TestGitAttributes:
    """Test .gitattributes creation."""

    def test_create_gitattributes(self):
        """Test .gitattributes file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal model
            (tmpdir / "config.json").write_text("{}")
            (tmpdir / "pytorch_model.bin").write_text("model")

            with tempfile.TemporaryDirectory() as output_dir:
                output_dir = Path(output_dir)

                preparer = ModelPreparer(tmpdir, output_dir)
                preparer.copy_model_files()
                preparer.create_gitattributes()

                # Check file exists
                gitattributes_path = output_dir / ".gitattributes"
                assert gitattributes_path.exists()

                # Check content
                content = gitattributes_path.read_text()
                assert "*.bin filter=lfs" in content
                assert "*.safetensors filter=lfs" in content


class TestFullPreparation:
    """Test full preparation pipeline."""

    def test_prepare_success(self):
        """Test successful preparation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create valid model
            config = {
                "model_type": "bert",
                "hidden_size": 768,
                "vocab_size": 30000,
            }

            (tmpdir / "config.json").write_text(json.dumps(config))
            (tmpdir / "pytorch_model.bin").write_text("model")
            (tmpdir / "tokenizer_config.json").write_text("{}")

            with tempfile.TemporaryDirectory() as output_dir:
                output_dir = Path(output_dir)

                preparer = ModelPreparer(tmpdir, output_dir)
                success = preparer.prepare()

                assert success is True

                # Check all expected files
                assert (output_dir / "config.json").exists()
                assert (output_dir / "pytorch_model.bin").exists()
                assert (output_dir / "model_metadata.json").exists()
                assert (output_dir / ".gitattributes").exists()

    def test_prepare_failure(self):
        """Test preparation failure with invalid model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Invalid model (missing files)

            with tempfile.TemporaryDirectory() as output_dir:
                output_dir = Path(output_dir)

                preparer = ModelPreparer(tmpdir, output_dir)
                success = preparer.prepare()

                assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
