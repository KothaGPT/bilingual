"""
Unit tests for Wikipedia preprocessing.

Tests:
- Text cleaning (markup removal)
- Sentence tokenization
- Normalization
- Filtering
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scripts.preprocess_wiki import WikiTextCleaner


class TestWikiTextCleaner:
    """Test Wikipedia text cleaning functionality."""

    @pytest.fixture
    def cleaner(self):
        """Create WikiTextCleaner instance."""
        return WikiTextCleaner(language="bn")

    def test_remove_html_tags(self, cleaner):
        """Test HTML tag removal."""
        text = "This is <b>bold</b> and <i>italic</i> text"
        cleaned = cleaner.clean_text(text)
        assert "<b>" not in cleaned
        assert "<i>" not in cleaned
        assert "bold" in cleaned
        assert "italic" in cleaned

    def test_remove_citations(self, cleaner):
        """Test citation removal."""
        text = "This is a fact[1] and another fact[2]."
        cleaned = cleaner.clean_text(text)
        assert "[1]" not in cleaned
        assert "[2]" not in cleaned
        assert "fact" in cleaned

    def test_remove_templates(self, cleaner):
        """Test template removal."""
        text = "Text before {{template|param=value}} text after"
        cleaned = cleaner.clean_text(text)
        assert "{{" not in cleaned
        assert "}}" not in cleaned
        assert "Text before" in cleaned
        assert "text after" in cleaned

    def test_remove_file_references(self, cleaner):
        """Test file/image reference removal."""
        text = "Text [[File:image.jpg|thumb|caption]] more text"
        cleaned = cleaner.clean_text(text)
        assert "[[File:" not in cleaned
        assert "image.jpg" not in cleaned
        assert "more text" in cleaned

    def test_remove_category_links(self, cleaner):
        """Test category link removal."""
        text = "Text [[Category:Some Category]] more text"
        cleaned = cleaner.clean_text(text)
        assert "[[Category:" not in cleaned
        assert "more text" in cleaned

    def test_clean_wiki_links(self, cleaner):
        """Test wiki link cleaning."""
        text = "This is a [[link|display text]] in the article"
        cleaned = cleaner.clean_text(text)
        assert "[[" not in cleaned
        assert "]]" not in cleaned
        assert "display text" in cleaned

    def test_remove_external_links(self, cleaner):
        """Test external link removal."""
        text = "Visit [http://example.com Example Site] for more"
        cleaned = cleaner.clean_text(text)
        assert "http://example.com" not in cleaned
        assert "for more" in cleaned

    def test_normalize_whitespace(self, cleaner):
        """Test whitespace normalization."""
        text = "Text  with   multiple    spaces"
        cleaned = cleaner.clean_text(text)
        assert "  " not in cleaned
        assert "Text with multiple spaces" == cleaned

    def test_normalize_text_unicode(self, cleaner):
        """Test Unicode normalization."""
        # Different Unicode representations of same character
        text = "café"  # e + combining acute
        normalized = cleaner.normalize_text(text)
        assert normalized == "café"  # Should be NFC normalized

    def test_sentence_tokenization_bangla(self, cleaner):
        """Test Bangla sentence tokenization."""
        text = "এটি প্রথম বাক্য। এটি দ্বিতীয় বাক্য। এটি তৃতীয় বাক্য।"
        sentences = cleaner.tokenize_sentences(text)
        assert len(sentences) == 3
        assert "প্রথম" in sentences[0]
        assert "দ্বিতীয়" in sentences[1]
        assert "তৃতীয়" in sentences[2]

    def test_sentence_tokenization_danda(self, cleaner):
        """Test sentence tokenization with Bangla danda (।)."""
        text = "এটি প্রথম বাক্য। এটি দ্বিতীয় বাক্য।"
        sentences = cleaner.tokenize_sentences(text)
        assert len(sentences) >= 2

    def test_filter_sentence_length(self, cleaner):
        """Test sentence length filtering."""
        # Too short
        assert not cleaner.filter_sentence("short", min_length=10)

        # Just right
        assert cleaner.filter_sentence(
            "This is a good length sentence", min_length=10, max_length=100
        )

        # Too long
        long_text = "x" * 600
        assert not cleaner.filter_sentence(long_text, min_length=10, max_length=500)

    def test_empty_text_handling(self, cleaner):
        """Test handling of empty text."""
        cleaned = cleaner.clean_text("")
        assert cleaned == ""

        sentences = cleaner.tokenize_sentences("")
        assert len(sentences) == 0

    def test_complex_markup_removal(self, cleaner):
        """Test removal of complex nested markup."""
        text = """
        This is text with [[File:image.jpg|thumb|[[nested link]]]] and
        {{template|param={{nested template}}}} and [http://example.com link]
        """
        cleaned = cleaner.clean_text(text)

        # Should not contain any markup
        assert "[[" not in cleaned
        assert "]]" not in cleaned
        assert "{{" not in cleaned
        assert "}}" not in cleaned
        assert "http://" not in cleaned

        # Should contain actual text
        assert "This is text" in cleaned


class TestPreprocessingPipeline:
    """Test full preprocessing pipeline."""

    def test_extract_and_process_sample(self):
        """Test processing a sample Wikipedia article."""
        # Sample Wikipedia-style text
        sample_text = """
        {{Infobox}}
        '''বাংলাদেশ''' দক্ষিণ এশিয়ার একটি রাষ্ট্র।[1]
        
        [[File:map.jpg|thumb|মানচিত্র]]
        
        ঢাকা বাংলাদেশের রাজধানী।[2] এটি একটি বড় শহর।
        
        [[Category:দেশ]]
        """

        cleaner = WikiTextCleaner(language="bn")

        # Clean
        cleaned = cleaner.clean_text(sample_text)

        # Should not contain markup
        assert "{{" not in cleaned
        assert "[[" not in cleaned
        assert "[1]" not in cleaned

        # Should contain actual content
        assert "বাংলাদেশ" in cleaned
        assert "ঢাকা" in cleaned

        # Normalize
        normalized = cleaner.normalize_text(cleaned)

        # Tokenize
        sentences = cleaner.tokenize_sentences(normalized)

        # Should have multiple sentences
        assert len(sentences) > 0

        # Filter
        filtered = [s for s in sentences if cleaner.filter_sentence(s)]

        # Should have valid sentences
        assert len(filtered) > 0


class TestDatasetSplitting:
    """Test dataset splitting functionality."""

    def test_split_ratios(self):
        """Test train/val/test split ratios."""
        # Create sample data
        sentences = [f"Sentence {i}" for i in range(1000)]

        # Split 80/10/10
        total = len(sentences)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)

        train = sentences[:train_size]
        val = sentences[train_size : train_size + val_size]
        test = sentences[train_size + val_size :]

        # Check sizes
        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100

        # Check no overlap
        assert set(train).isdisjoint(set(val))
        assert set(train).isdisjoint(set(test))
        assert set(val).isdisjoint(set(test))

    def test_file_writing(self):
        """Test writing processed data to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create sample sentences
            sentences = ["বাক্য ১", "বাক্য ২", "বাক্য ৩"]

            # Write to file
            output_file = tmpdir / "test.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(sentences))

            # Read back
            with open(output_file, "r", encoding="utf-8") as f:
                loaded = [line.strip() for line in f]

            # Verify
            assert loaded == sentences


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
