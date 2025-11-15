#!/usr/bin/env python3
"""
Evaluate trained Wikipedia language model.

Usage:
    python scripts/evaluate_wiki_lm.py --model models/wikipedia/base --data datasets/wikipedia/processed/test
    python scripts/evaluate_wiki_lm.py --model models/wikipedia/base --interactive
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WikipediaLMEvaluator:
    """Evaluate Wikipedia language model."""

    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = model_path
        self.device = (
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = None
        self.model = None
        self.model_type = None

    def load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Try to load as MLM first, then CLM
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            self.model_type = "mlm"
            logger.info("Loaded as Masked Language Model")
        except:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model_type = "clm"
            logger.info("Loaded as Causal Language Model")

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on {self.device}")

    def compute_perplexity(self, texts: List[str], batch_size: int = 8) -> float:
        """
        Compute perplexity on a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Perplexity score
        """
        logger.info(f"Computing perplexity on {len(texts)} texts...")

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)

                # Compute loss
                if self.model_type == "mlm":
                    # For MLM, use input_ids as labels
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                else:
                    # For CLM, shift labels
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )

                loss = outputs.loss

                # Accumulate
                num_tokens = attention_mask.sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity

    def fill_mask(self, text: str, top_k: int = 5) -> List[Dict]:
        """
        Fill masked tokens in text (for MLM models).

        Args:
            text: Text with [MASK] tokens
            top_k: Number of top predictions to return

        Returns:
            List of predictions
        """
        if self.model_type != "mlm":
            logger.warning("fill_mask only works with MLM models")
            return []

        fill_mask_pipeline = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

        results = fill_mask_pipeline(text, top_k=top_k)
        return results

    def generate_text(
        self, prompt: str, max_length: int = 100, num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from prompt (for CLM models).

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated texts
        """
        if self.model_type != "clm":
            logger.warning("generate_text only works with CLM models")
            return []

        text_gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

        results = text_gen_pipeline(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
        )

        return [r["generated_text"] for r in results]

    def evaluate_dataset(self, data_path: Path, batch_size: int = 8) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            data_path: Path to dataset file or directory
            batch_size: Batch size for processing

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on dataset: {data_path}")

        # Load dataset
        if data_path.is_file():
            with open(data_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Load from directory
            dataset = load_dataset("text", data_files=str(data_path / "*.txt"))
            texts = dataset["train"]["text"]

        logger.info(f"Loaded {len(texts)} texts")

        # Compute perplexity
        perplexity = self.compute_perplexity(texts, batch_size)

        metrics = {
            "perplexity": float(perplexity),
            "num_texts": len(texts),
            "model_type": self.model_type,
        }

        return metrics

    def interactive_mode(self) -> None:
        """Run interactive evaluation mode."""
        logger.info("Starting interactive mode...")
        logger.info(f"Model type: {self.model_type}")

        if self.model_type == "mlm":
            logger.info("Use [MASK] token in your text to fill masked positions")
            logger.info("Example: আমি [MASK] খাই")
        else:
            logger.info("Enter a prompt to generate text")
            logger.info("Example: আমি বাংলায়")

        print("\nType 'quit' to exit\n")

        while True:
            try:
                user_input = input(">>> ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if not user_input:
                    continue

                if self.model_type == "mlm":
                    # Fill mask
                    if "[MASK]" not in user_input:
                        print("Please include [MASK] token in your text")
                        continue

                    results = self.fill_mask(user_input, top_k=5)

                    print("\nPredictions:")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['sequence']} (score: {result['score']:.4f})")
                    print()

                else:
                    # Generate text
                    results = self.generate_text(user_input, max_length=100, num_return_sequences=3)

                    print("\nGenerated texts:")
                    for i, text in enumerate(results, 1):
                        print(f"{i}. {text}")
                    print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wikipedia language model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, help="Path to test dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="results/wikipedia_eval.json",
        help="Output file for evaluation metrics",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for evaluation",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    try:
        # Initialize evaluator
        evaluator = WikipediaLMEvaluator(model_path, args.device)
        evaluator.load_model()

        if args.interactive:
            # Interactive mode
            evaluator.interactive_mode()

        elif args.data:
            # Evaluate on dataset
            data_path = Path(args.data)
            metrics = evaluator.evaluate_dataset(data_path, args.batch_size)

            # Save metrics
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Saved evaluation metrics: {output_path}")

            # Print metrics
            print("\n" + "=" * 60)
            print("EVALUATION METRICS")
            print("=" * 60)
            for key, value in metrics.items():
                print(f"{key}: {value}")
            print("=" * 60 + "\n")

        else:
            logger.error("Please specify --data or --interactive")
            sys.exit(1)

        logger.info("Evaluation complete!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
