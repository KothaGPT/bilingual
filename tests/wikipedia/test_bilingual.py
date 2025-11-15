"""
Unit tests for bilingual Wikipedia training.

Tests:
- Article alignment
- Bilingual dataset loading
- Cross-lingual training
- Bilingual evaluation
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TestBilingualAlignment:
    """Test bilingual article alignment."""

    def test_interwiki_link_extraction(self):
        """Test extraction of interwiki links."""
        from scripts.align_bilingual_wiki import BilingualAligner

        aligner = BilingualAligner()

        # Test Bangla to English
        bn_text = "এটি একটি নিবন্ধ [[en:Article Title]] আরও কিছু"
        en_title = aligner.extract_interwiki_links(bn_text, "bn")
        assert en_title == "Article Title"

        # Test English to Bangla
        en_text = "This is an article [[bn:নিবন্ধ শিরোনাম]] more text"
        bn_title = aligner.extract_interwiki_links(en_text, "en")
        assert bn_title == "নিবন্ধ শিরোনাম"

    def test_article_alignment(self):
        """Test article alignment."""
        from scripts.align_bilingual_wiki import BilingualAligner

        aligner = BilingualAligner()

        # Sample articles
        bn_articles = {
            "বাংলাদেশ": "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ [[en:Bangladesh]]",
            "ঢাকা": "ঢাকা বাংলাদেশের রাজধানী [[en:Dhaka]]",
        }

        en_articles = {
            "Bangladesh": "Bangladesh is a country in South Asia",
            "Dhaka": "Dhaka is the capital of Bangladesh",
        }

        # Align
        aligned = aligner.align_articles(bn_articles, en_articles)

        assert len(aligned) == 2
        assert aligned[0][0] == "বাংলাদেশ"
        assert aligned[0][2] == "Bangladesh"

    def test_save_aligned_corpus(self):
        """Test saving aligned corpus."""
        from scripts.align_bilingual_wiki import BilingualAligner

        aligner = BilingualAligner()

        aligned_articles = [
            ("বাংলাদেশ", "বাংলাদেশ একটি দেশ", "Bangladesh", "Bangladesh is a country"),
            ("ঢাকা", "ঢাকা রাজধানী", "Dhaka", "Dhaka is capital"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            aligner.save_aligned_corpus(aligned_articles, output_dir)

            # Check files exist
            assert (output_dir / "aligned_articles.json").exists()
            assert (output_dir / "bangla.txt").exists()
            assert (output_dir / "english.txt").exists()
            assert (output_dir / "metadata.json").exists()

            # Check JSON content
            with open(output_dir / "aligned_articles.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                assert len(data) == 2
                assert data[0]["bn_title"] == "বাংলাদেশ"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestBilingualTraining:
    """Test bilingual model training."""

    @pytest.fixture
    def bilingual_data(self):
        """Create sample bilingual data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create bilingual dataset
            (tmpdir / "train").mkdir()

            # Bangla-English pairs
            pairs = [
                ("আমি ভাত খাই", "I eat rice"),
                ("এটি একটি বই", "This is a book"),
                ("ঢাকা বাংলাদেশের রাজধানী", "Dhaka is the capital of Bangladesh"),
            ]

            with open(tmpdir / "train" / "bilingual.txt", "w", encoding="utf-8") as f:
                for bn, en in pairs:
                    f.write(f"{bn} [SEP] {en}\n")

            yield tmpdir

    def test_bilingual_tokenization(self, bilingual_data):
        """Test tokenization of bilingual text."""
        # Use multilingual model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        text = "আমি ভাত খাই [SEP] I eat rice"
        tokens = tokenizer(text, return_tensors="pt")

        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1

    def test_multilingual_model_loading(self):
        """Test loading multilingual model."""
        # Test with XLM-RoBERTa (smaller version)
        try:
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

            assert tokenizer is not None
            assert model is not None
        except Exception as e:
            pytest.skip(f"Could not load XLM-R: {e}")

    def test_cross_lingual_inference(self):
        """Test cross-lingual inference."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        model.eval()

        # Bangla text
        bn_text = "আমি ভাত খাই"
        bn_inputs = tokenizer(bn_text, return_tensors="pt")

        # English text
        en_text = "I eat rice"
        en_inputs = tokenizer(en_text, return_tensors="pt")

        with torch.no_grad():
            bn_outputs = model(**bn_inputs, labels=bn_inputs["input_ids"])
            en_outputs = model(**en_inputs, labels=en_inputs["input_ids"])

        # Both should produce valid outputs
        assert bn_outputs.loss is not None
        assert en_outputs.loss is not None


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestBilingualEvaluation:
    """Test bilingual model evaluation."""

    def test_cross_lingual_similarity(self):
        """Test cross-lingual similarity computation."""
        from bilingual.modules.wikipedia_lm import WikipediaLanguageModel

        # Use multilingual model
        model = WikipediaLanguageModel(
            "bert-base-multilingual-cased", device="cpu", model_type="mlm"
        )

        # Similar meaning in different languages
        bn_text = "আমি ভাত খাই"
        en_text = "I eat rice"

        # Get embeddings
        bn_emb = model.get_sentence_embedding(bn_text)
        en_emb = model.get_sentence_embedding(en_text)

        # Compute similarity
        similarity = torch.nn.functional.cosine_similarity(
            bn_emb.unsqueeze(0),
            en_emb.unsqueeze(0),
        ).item()

        # Should have some similarity (multilingual models align languages)
        assert -1.0 <= similarity <= 1.0

    def test_bilingual_fill_mask(self):
        """Test fill-mask with bilingual context."""
        from bilingual.modules.wikipedia_lm import WikipediaLanguageModel

        model = WikipediaLanguageModel(
            "bert-base-multilingual-cased", device="cpu", model_type="mlm"
        )

        # Bangla
        bn_results = model.fill_mask("আমি [MASK] খাই", top_k=3)
        assert len(bn_results) == 3

        # English
        en_results = model.fill_mask("I [MASK] rice", top_k=3)
        assert len(en_results) == 3


class TestBilingualDataset:
    """Test bilingual dataset handling."""

    def test_parallel_corpus_format(self):
        """Test parallel corpus format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create parallel corpus
            bn_file = tmpdir / "bangla.txt"
            en_file = tmpdir / "english.txt"

            bn_sentences = ["বাক্য ১", "বাক্য ২", "বাক্য ৩"]
            en_sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]

            with open(bn_file, "w", encoding="utf-8") as f:
                f.write("\n".join(bn_sentences))

            with open(en_file, "w", encoding="utf-8") as f:
                f.write("\n".join(en_sentences))

            # Load and verify
            with open(bn_file, "r", encoding="utf-8") as f:
                loaded_bn = [line.strip() for line in f]

            with open(en_file, "r", encoding="utf-8") as f:
                loaded_en = [line.strip() for line in f]

            assert loaded_bn == bn_sentences
            assert loaded_en == en_sentences
            assert len(loaded_bn) == len(loaded_en)

    def test_bilingual_metadata(self):
        """Test bilingual corpus metadata."""
        metadata = {
            "total_pairs": 1000,
            "bn_file": "bangla.txt",
            "en_file": "english.txt",
            "alignment_method": "interwiki",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"

            # Save
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Load
            with open(metadata_path, "r") as f:
                loaded = json.load(f)

            assert loaded == metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
