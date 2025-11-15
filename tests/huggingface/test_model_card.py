"""
Unit tests for model card generation.

Tests:
- Model card template selection
- Metadata extraction
- Card generation
- Format validation
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scripts.huggingface.generate_model_card import MODEL_CARD_TEMPLATES, generate_model_card


class TestModelCardGeneration:
    """Test model card generation."""

    @pytest.fixture
    def model_with_metadata(self):
        """Create model directory with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            metadata = {
                "model_type": "bert",
                "architecture": "BertForMaskedLM",
                "num_parameters": 110000000,
                "model_size_mb": 420,
                "vocab_size": 30000,
            }

            (tmpdir / "model_metadata.json").write_text(json.dumps(metadata))

            yield tmpdir

    def test_generate_base_model_card(self, model_with_metadata):
        """Test base model card generation."""
        card = generate_model_card(
            model_with_metadata, model_type="base", repo_id="test-user/test-model"
        )

        assert isinstance(card, str)
        assert len(card) > 0

        # Check required sections
        assert "# Bangla Wikipedia Language Model" in card
        assert "## Model Description" in card
        assert "## Training Data" in card
        assert "## Usage" in card
        assert "## License" in card

        # Check metadata is included
        assert "110,000,000" in card
        assert "420 MB" in card
        assert "30,000" in card

        # Check repo ID
        assert "test-user/test-model" in card

    def test_generate_literary_model_card(self, model_with_metadata):
        """Test literary model card generation."""
        card = generate_model_card(
            model_with_metadata, model_type="literary", repo_id="test-user/literary-model"
        )

        assert isinstance(card, str)
        assert "# Bangla Wikipedia + Literary Language Model" in card
        assert "literary" in card.lower()
        assert "poetry" in card.lower()

    def test_model_card_without_metadata(self):
        """Test model card generation without metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # No metadata file
            card = generate_model_card(tmpdir, model_type="base", repo_id="test-user/test-model")

            # Should still generate card with default values
            assert isinstance(card, str)
            assert len(card) > 0


class TestModelCardTemplates:
    """Test model card templates."""

    def test_base_template_exists(self):
        """Test base template exists."""
        assert "base" in MODEL_CARD_TEMPLATES
        assert isinstance(MODEL_CARD_TEMPLATES["base"], str)
        assert len(MODEL_CARD_TEMPLATES["base"]) > 0

    def test_literary_template_exists(self):
        """Test literary template exists."""
        assert "literary" in MODEL_CARD_TEMPLATES
        assert isinstance(MODEL_CARD_TEMPLATES["literary"], str)
        assert len(MODEL_CARD_TEMPLATES["literary"]) > 0

    def test_template_has_required_sections(self):
        """Test templates have required sections."""
        required_sections = [
            "## Model Description",
            "## Training Data",
            "## Usage",
            "## License",
        ]

        for template_name, template in MODEL_CARD_TEMPLATES.items():
            for section in required_sections:
                assert section in template, f"Missing {section} in {template_name}"

    def test_template_has_yaml_frontmatter(self):
        """Test templates have YAML frontmatter."""
        for template_name, template in MODEL_CARD_TEMPLATES.items():
            assert template.startswith("---"), f"Missing YAML frontmatter in {template_name}"
            assert "language:" in template
            assert "license:" in template
            assert "tags:" in template


class TestModelCardFormatting:
    """Test model card formatting."""

    def test_parameter_formatting(self):
        """Test parameter count formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            metadata = {
                "num_parameters": 110000000,
                "model_size_mb": 420.5,
                "vocab_size": 30000,
            }

            (tmpdir / "model_metadata.json").write_text(json.dumps(metadata))

            card = generate_model_card(tmpdir, "base", "test/model")

            # Check formatting with commas
            assert "110,000,000" in card
            assert "30,000" in card

    def test_code_blocks(self):
        """Test code blocks are properly formatted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            card = generate_model_card(tmpdir, "base", "test/model")

            # Check Python code blocks
            assert "```python" in card
            assert "```" in card

            # Check bibtex code block
            assert "```bibtex" in card

    def test_links(self):
        """Test links are properly formatted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            card = generate_model_card(tmpdir, "base", "test-user/test-model")

            # Check HuggingFace link
            assert "https://huggingface.co/test-user/test-model" in card

            # Check GitHub link
            assert "https://github.com/KothaGPT/bilingual" in card


class TestModelCardSaving:
    """Test saving model cards."""

    def test_save_model_card(self):
        """Test saving model card to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            card = generate_model_card(tmpdir, "base", "test/model")

            # Save
            readme_path = tmpdir / "README.md"
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(card)

            # Verify
            assert readme_path.exists()

            # Read back
            with open(readme_path, "r", encoding="utf-8") as f:
                loaded_card = f.read()

            assert loaded_card == card

    def test_model_card_encoding(self):
        """Test model card handles Unicode properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            card = generate_model_card(tmpdir, "base", "test/model")

            # Should contain Bangla text
            assert "বাংলা" in card or "Bangla" in card

            # Save with UTF-8
            readme_path = tmpdir / "README.md"
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(card)

            # Read back
            with open(readme_path, "r", encoding="utf-8") as f:
                loaded_card = f.read()

            assert loaded_card == card


class TestModelCardValidation:
    """Test model card validation."""

    def test_yaml_frontmatter_valid(self):
        """Test YAML frontmatter is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            card = generate_model_card(tmpdir, "base", "test/model")

            # Extract YAML frontmatter
            lines = card.split("\n")
            assert lines[0] == "---"

            # Find closing ---
            yaml_end = None
            for i, line in enumerate(lines[1:], 1):
                if line == "---":
                    yaml_end = i
                    break

            assert yaml_end is not None, "YAML frontmatter not properly closed"

            yaml_content = "\n".join(lines[1:yaml_end])

            # Check required fields
            assert "language:" in yaml_content
            assert "license:" in yaml_content
            assert "tags:" in yaml_content

    def test_required_sections_present(self):
        """Test all required sections are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            card = generate_model_card(tmpdir, "base", "test/model")

            required_sections = [
                "# Bangla Wikipedia Language Model",
                "## Model Description",
                "## Intended Use",
                "## Training Data",
                "## Training Procedure",
                "## Evaluation",
                "## Usage",
                "## Limitations",
                "## Citation",
                "## License",
            ]

            for section in required_sections:
                assert section in card, f"Missing section: {section}"

    def test_usage_examples_valid(self):
        """Test usage examples are syntactically valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            card = generate_model_card(tmpdir, "base", "test/model")

            # Check for common Python syntax
            assert "from transformers import" in card
            assert "AutoTokenizer" in card
            assert "AutoModel" in card or "AutoModelForMaskedLM" in card


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
