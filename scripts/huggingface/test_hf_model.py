#!/usr/bin/env python3
"""
Test Hugging Face model inference.

Usage:
    python scripts/huggingface/test_hf_model.py --repo your-username/bn-wikipedia-lm
    python scripts/huggingface/test_hf_model.py --model models/huggingface_ready/bn-wikipedia-lm --local
"""

import argparse
import logging
import sys
from typing import Dict, List

try:
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.error("transformers not installed. Install with: pip install transformers torch")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelTester:
    """Test Hugging Face model."""

    TEST_CASES = {
        "fill_mask": [
            "বাংলাদেশের রাজধানী [MASK]",
            "আমি [MASK] খাই",
            "সূর্য [MASK] উঠে",
        ],
        "embeddings": [
            "আমি বাংলায় কথা বলি",
            "বাংলাদেশ একটি সুন্দর দেশ",
        ],
    }

    def __init__(self, model_id: str, local: bool = False):
        self.model_id = model_id
        self.local = local
        self.tokenizer = None
        self.model = None

    def load_model(self) -> bool:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_id}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)

            logger.info("✓ Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def test_fill_mask(self) -> bool:
        """Test fill-mask functionality."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing Fill-Mask")
        logger.info("=" * 60)

        try:
            fill_mask = pipeline(
                "fill-mask", model=self.model, tokenizer=self.tokenizer, device=-1  # CPU
            )

            for text in self.TEST_CASES["fill_mask"]:
                logger.info(f"\nInput: {text}")
                results = fill_mask(text, top_k=3)

                for i, result in enumerate(results, 1):
                    logger.info(f"  {i}. {result['sequence']} (score: {result['score']:.4f})")

            logger.info("\n✓ Fill-mask test passed")
            return True

        except Exception as e:
            logger.error(f"Fill-mask test failed: {e}")
            return False

    def test_embeddings(self) -> bool:
        """Test embedding generation."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing Embeddings")
        logger.info("=" * 60)

        try:
            # Load as base model for embeddings
            model = AutoModel.from_pretrained(self.model_id)

            for text in self.TEST_CASES["embeddings"]:
                logger.info(f"\nText: {text}")

                inputs = self.tokenizer(text, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state

                # Mean pooling
                sentence_embedding = embeddings.mean(dim=1)

                logger.info(f"  Embedding shape: {sentence_embedding.shape}")
                logger.info(f"  Embedding (first 5): {sentence_embedding[0, :5].tolist()}")

            logger.info("\n✓ Embeddings test passed")
            return True

        except Exception as e:
            logger.error(f"Embeddings test failed: {e}")
            return False

    def test_tokenizer(self) -> bool:
        """Test tokenizer functionality."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing Tokenizer")
        logger.info("=" * 60)

        try:
            test_text = "আমি বাংলায় কথা বলি"

            # Tokenize
            tokens = self.tokenizer.tokenize(test_text)
            token_ids = self.tokenizer.encode(test_text)

            logger.info(f"Text: {test_text}")
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Token IDs: {token_ids}")

            # Decode
            decoded = self.tokenizer.decode(token_ids)
            logger.info(f"Decoded: {decoded}")

            # Check special tokens
            logger.info(f"\nSpecial tokens:")
            logger.info(f"  PAD: {self.tokenizer.pad_token}")
            logger.info(f"  MASK: {self.tokenizer.mask_token}")
            logger.info(f"  CLS: {self.tokenizer.cls_token}")
            logger.info(f"  SEP: {self.tokenizer.sep_token}")

            logger.info("\n✓ Tokenizer test passed")
            return True

        except Exception as e:
            logger.error(f"Tokenizer test failed: {e}")
            return False

    def test_model_info(self) -> bool:
        """Test model configuration."""
        logger.info("\n" + "=" * 60)
        logger.info("Model Information")
        logger.info("=" * 60)

        try:
            config = self.model.config

            logger.info(f"Model type: {config.model_type}")
            logger.info(f"Hidden size: {config.hidden_size}")
            logger.info(f"Num layers: {config.num_hidden_layers}")
            logger.info(f"Num attention heads: {config.num_attention_heads}")
            logger.info(f"Vocab size: {config.vocab_size}")
            logger.info(f"Max position embeddings: {config.max_position_embeddings}")

            # Count parameters
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Total parameters: {num_params:,}")

            logger.info("\n✓ Model info retrieved")
            return True

        except Exception as e:
            logger.error(f"Model info test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests."""
        logger.info("=" * 60)
        logger.info("Starting Model Tests")
        logger.info("=" * 60)

        # Load model
        if not self.load_model():
            return False

        # Run tests
        tests = [
            ("Model Info", self.test_model_info),
            ("Tokenizer", self.test_tokenizer),
            ("Fill-Mask", self.test_fill_mask),
            ("Embeddings", self.test_embeddings),
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"{test_name} test error: {e}")
                results[test_name] = False

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)

        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{test_name}: {status}")

        all_passed = all(results.values())

        if all_passed:
            logger.info("\n✓ All tests passed!")
        else:
            logger.error("\n✗ Some tests failed")

        logger.info("=" * 60)

        return all_passed


def main():
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Please install transformers: pip install transformers torch")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test Hugging Face model")
    parser.add_argument("--repo", type=str, help="Hugging Face repository ID (username/model-name)")
    parser.add_argument("--model", type=str, help="Local model path")
    parser.add_argument("--local", action="store_true", help="Load from local path instead of Hub")

    args = parser.parse_args()

    # Determine model ID
    if args.local and args.model:
        model_id = args.model
        local = True
    elif args.repo:
        model_id = args.repo
        local = False
    else:
        logger.error("Please specify either --repo or --model with --local")
        sys.exit(1)

    # Run tests
    tester = ModelTester(model_id, local)
    success = tester.run_all_tests()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
