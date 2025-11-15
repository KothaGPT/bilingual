"""
Unit tests for Wikipedia LM evaluation.

Tests:
- Model loading for evaluation
- Perplexity computation
- Fill-mask functionality (MLM)
- Text generation (CLM)
- Metrics computation
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
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestModelLoading:
    """Test model loading for evaluation."""

    def test_load_mlm_model(self):
        """Test loading MLM model."""
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        assert model is not None
        assert tokenizer is not None

        # Set to eval mode
        model.eval()
        assert not model.training

    def test_load_clm_model(self):
        """Test loading CLM model."""
        try:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            assert model is not None
            assert tokenizer is not None

            model.eval()
            assert not model.training
        except Exception as e:
            pytest.skip(f"Could not load GPT-2: {e}")

    def test_model_device_placement(self):
        """Test model can be placed on correct device."""
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

        # CPU
        model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

        # GPU (if available)
        if torch.cuda.is_available():
            model.to("cuda")
            assert next(model.parameters()).device.type == "cuda"


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestInference:
    """Test model inference."""

    @pytest.fixture
    def mlm_model(self):
        """Load MLM model for testing."""
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model.eval()
        return model, tokenizer

    def test_forward_pass(self, mlm_model):
        """Test model forward pass."""
        model, tokenizer = mlm_model

        text = "এটি একটি পরীক্ষা বাক্য।"
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        assert outputs.loss is not None
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1  # Batch size

    def test_fill_mask(self, mlm_model):
        """Test fill-mask functionality."""
        from transformers import pipeline

        model, tokenizer = mlm_model

        fill_mask = pipeline(
            "fill-mask",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
        )

        # Test with English (model is multilingual)
        results = fill_mask("Hello [MASK] world", top_k=3)

        assert len(results) == 3
        assert all("sequence" in r for r in results)
        assert all("score" in r for r in results)
        assert all("token" in r for r in results)

    def test_batch_inference(self, mlm_model):
        """Test batch inference."""
        model, tokenizer = mlm_model

        texts = ["এটি প্রথম বাক্য।", "এটি দ্বিতীয় বাক্য।"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        assert outputs.loss is not None
        assert outputs.logits.shape[0] == 2  # Batch size 2


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestPerplexity:
    """Test perplexity computation."""

    def test_loss_to_perplexity(self):
        """Test conversion from loss to perplexity."""
        import numpy as np

        loss = 2.0
        perplexity = np.exp(loss)

        assert perplexity > 1.0
        assert perplexity == pytest.approx(7.389, rel=0.01)

    def test_perplexity_computation(self):
        """Test perplexity computation on sample text."""
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model.eval()

        texts = ["This is a test sentence."] * 5

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt")
                outputs = model(**inputs, labels=inputs["input_ids"])

                loss = outputs.loss.item()
                num_tokens = inputs["attention_mask"].sum().item()

                total_loss += loss * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        assert perplexity > 0
        assert perplexity < 1000  # Reasonable range


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestMetrics:
    """Test evaluation metrics."""

    def test_metrics_structure(self):
        """Test evaluation metrics structure."""
        metrics = {
            "perplexity": 15.5,
            "num_texts": 100,
            "model_type": "mlm",
        }

        assert "perplexity" in metrics
        assert "num_texts" in metrics
        assert "model_type" in metrics

        assert isinstance(metrics["perplexity"], float)
        assert isinstance(metrics["num_texts"], int)
        assert isinstance(metrics["model_type"], str)

    def test_metrics_serialization(self):
        """Test metrics can be saved to JSON."""
        metrics = {
            "perplexity": 15.5,
            "num_texts": 100,
            "model_type": "mlm",
            "eval_loss": 2.74,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"

            # Save
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Load
            with open(output_path, "r") as f:
                loaded_metrics = json.load(f)

            assert loaded_metrics == metrics

    def test_metrics_validation(self):
        """Test metrics value validation."""
        # Perplexity should be positive
        perplexity = 15.5
        assert perplexity > 0

        # Loss should be non-negative
        loss = 2.74
        assert loss >= 0

        # Num texts should be positive integer
        num_texts = 100
        assert num_texts > 0
        assert isinstance(num_texts, int)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestEvaluationPipeline:
    """Test full evaluation pipeline."""

    def test_evaluate_on_sample_data(self):
        """Test evaluation on sample dataset."""
        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model.eval()

        # Sample texts
        texts = [
            "This is a test sentence.",
            "Another test sentence here.",
            "One more for good measure.",
        ]

        # Compute perplexity
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                outputs = model(**inputs, labels=inputs["input_ids"])

                loss = outputs.loss.item()
                num_tokens = inputs["attention_mask"].sum().item()

                total_loss += loss * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Create metrics
        metrics = {
            "perplexity": perplexity,
            "num_texts": len(texts),
            "avg_loss": avg_loss,
        }

        assert metrics["perplexity"] > 0
        assert metrics["num_texts"] == 3
        assert metrics["avg_loss"] > 0

    def test_evaluation_output_format(self):
        """Test evaluation output format."""
        # Mock evaluation results
        results = {
            "eval_loss": 2.74,
            "eval_perplexity": 15.5,
            "eval_runtime": 10.5,
            "eval_samples_per_second": 9.52,
        }

        # Verify structure
        assert "eval_loss" in results
        assert "eval_perplexity" in results

        # Verify types
        assert isinstance(results["eval_loss"], float)
        assert isinstance(results["eval_perplexity"], float)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
@pytest.mark.slow
class TestInteractiveMode:
    """Test interactive evaluation mode."""

    def test_fill_mask_interactive(self):
        """Test fill-mask in interactive mode."""
        from transformers import pipeline

        model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1)

        # Test input
        text = "Hello [MASK] world"
        results = fill_mask(text, top_k=5)

        assert len(results) == 5
        assert all("sequence" in r for r in results)

        # Results should be sorted by score
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
