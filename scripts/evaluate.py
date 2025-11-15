#!/usr/bin/env python3
"""
Evaluation script for the Bangla-English translation model.

This script provides functionality to evaluate a trained translation model using:
- BLEU score (using SacreBLEU)
- ROUGE score
- Custom beam search decoding
- Interactive translation mode
"""
import argparse
import json
import logging

# Add the project root to the Python path
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from rouge import Rouge
from sacrebleu.metrics import BLEU
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from bilingual.models.transformer import TransformerModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Translator:
    """
    A class for translating text using a trained sequence-to-sequence model.

    This class handles model loading, text preprocessing, and translation with
    support for both greedy decoding and beam search.

    Args:
        config_path: Path to the YAML configuration file
        checkpoint_path: Path to the model checkpoint
        beam_size: Size of the beam for beam search (default: 5)
        max_length: Maximum length of generated translations (default: 128)
    """

    def __init__(
        self, config_path: str, checkpoint_path: str, beam_size: int = 5, max_length: int = 128
    ):
        """Initialize the translator with model and decoding parameters."""
        # Initialize token indices first (needed for model loading)
        self.bos_token = 2  # BOS token ID
        self.eos_token = 3  # EOS token ID
        self.pad_token = 0  # PAD token ID
        self.unk_token = 1  # UNK token ID

        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beam_size = beam_size
        self.max_length = max_length

        # Initialize model and move to device
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Initialize metrics
        self.bleu = BLEU()
        self.rouge = Rouge()

        logger.info(f"Initialized Translator with beam_size={beam_size}, max_length={max_length}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate the configuration file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the configuration
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate required model parameters
        required_model_params = [
            "src_vocab_size",
            "tgt_vocab_size",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "dropout",
            "max_seq_length",
        ]

        for param in required_model_params:
            if param not in config.get("model", {}):
                raise ValueError(f"Missing required model parameter: {param}")

        return config

    def _load_model(self, checkpoint_path: str) -> TransformerModel:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint file

        Returns:
            Loaded and configured TransformerModel
        """
        logger.info(f"Loading model from {checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Initialize model
            model_config = self.config["model"]
            model = TransformerModel(
                src_vocab_size=model_config["src_vocab_size"],
                tgt_vocab_size=model_config["tgt_vocab_size"],
                d_model=model_config["d_model"],
                nhead=model_config["nhead"],
                num_encoder_layers=model_config["num_encoder_layers"],
                num_decoder_layers=model_config["num_decoder_layers"],
                dim_feedforward=model_config["dim_feedforward"],
                dropout=model_config["dropout"],
                max_seq_length=model_config["max_seq_length"],
                pad_idx=self.pad_token,
            ).to(self.device)

            # Load model weights
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def translate(
        self,
        src_text: str,
        max_length: Optional[int] = None,
        beam_size: Optional[int] = None,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        """
        Translate a source text to the target language.

        Args:
            src_text: Source text to translate
            max_length: Maximum length of generated translation (default: self.max_length)
            beam_size: Size of the beam for beam search (default: self.beam_size)
            length_penalty: Length penalty for beam search (alpha)
            temperature: Temperature for sampling (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (1.0 = disabled)
            no_repeat_ngram_size: If > 0, all ngrams of this size can only occur once

        Returns:
            Translated text
        """
        max_length = max_length or self.max_length
        beam_size = beam_size or self.beam_size

        # Tokenize input (in a real scenario, use the actual tokenizer)
        src_tokens = src_text.split()
        src_ids = [int(t) if t.isdigit() else self.unk_token for t in src_tokens]

        # Convert to tensor and add batch dimension
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # Generate translation
        if beam_size > 1:
            output_ids = self._beam_search(
                src,
                beam_size=beam_size,
                max_length=max_length,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        else:
            output_ids = self._greedy_decode(
                src,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        # Convert token IDs to text
        output_tokens = [
            str(t) for t in output_ids if t not in [self.bos_token, self.eos_token, self.pad_token]
        ]
        return " ".join(output_tokens)

    def _greedy_decode(
        self,
        src: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> List[int]:
        """Greedy decoding with optional sampling."""
        with torch.no_grad():
            # Encode source
            memory = self.model.encode(src)

            # Initialize target with BOS token
            tgt = torch.ones(1, 1, dtype=torch.long, device=self.device) * self.bos_token

            # Track generated n-grams
            generated_ngrams = [set() for _ in range(no_repeat_ngram_size + 1)]

            # Generate translation token by token
            for _ in range(max_length):
                # Get next token probabilities
                output = self.model.decode(tgt, memory)
                next_token_logits = output[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    min_val = top_k_logits[:, -1].unsqueeze(1)
                    next_token_logits = torch.where(
                        next_token_logits < min_val,
                        torch.ones_like(next_token_logits) * -float("inf"),
                        next_token_logits,
                    )

                # Apply nucleus sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = -float("inf")

                # Apply softmax and sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Stop if EOS token is generated
                if next_token.item() == self.eos_token:
                    break

                # Apply n-gram blocking if enabled
                if no_repeat_ngram_size > 0:
                    # Get the last n-1 tokens
                    prev_tokens = tgt[0].tolist()
                    ngram_prefix = prev_tokens[-(no_repeat_ngram_size - 1) :] + [next_token.item()]

                    # Check if this n-gram has been generated before
                    ngram_tuple = tuple(ngram_prefix)
                    if ngram_tuple in generated_ngrams[no_repeat_ngram_size - 1]:
                        continue

                    # Add to generated n-grams
                    for n in range(1, no_repeat_ngram_size + 1):
                        if len(ngram_prefix) >= n:
                            generated_ngrams[n - 1].add(tuple(ngram_prefix[-n:]))

                # Append predicted token to the sequence
                tgt = torch.cat([tgt, next_token], dim=1)

            return tgt[0].cpu().tolist()

    def _beam_search(
        self,
        src: torch.Tensor,
        beam_size: int,
        max_length: int,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> List[int]:
        """Beam search decoding."""
        with torch.no_grad():
            # Encode source
            memory = self.model.encode(src)
            batch_size = memory.size(0)

            # Initialize beams
            beams = [
                {
                    "tokens": [self.bos_token],
                    "score": 0.0,
                    "memory": memory,
                    "generated_ngrams": [set() for _ in range(no_repeat_ngram_size + 1)],
                }
            ]

            for _ in range(max_length):
                candidates = []

                for beam in beams:
                    if beam["tokens"][-1] == self.eos_token:
                        candidates.append(beam)
                        continue

                    # Prepare decoder input
                    tgt = torch.tensor([beam["tokens"]], device=self.device)

                    # Get next token probabilities
                    output = self.model.decode(tgt, beam["memory"])
                    next_token_logits = output[:, -1, :]

                    # Get top-k tokens
                    topk_scores, topk_indices = torch.topk(
                        next_token_logits,
                        k=beam_size * 2,  # Get more candidates for filtering
                        dim=-1,
                    )

                    # Create new beams
                    for i in range(topk_scores.size(-1)):
                        token = topk_indices[0, i].item()
                        score = beam["score"] + topk_scores[0, i].item()

                        # Apply n-gram blocking if enabled
                        if no_repeat_ngram_size > 0:
                            # Get the last n-1 tokens
                            prev_tokens = beam["tokens"][-(no_repeat_ngram_size - 1) :] + [token]

                            # Check if this n-gram has been generated before
                            ngram_tuple = tuple(prev_tokens)
                            if ngram_tuple in beam["generated_ngrams"][no_repeat_ngram_size - 1]:
                                continue

                            # Update n-grams
                            new_ngrams = [s.copy() for s in beam["generated_ngrams"]]
                            for n in range(1, no_repeat_ngram_size + 1):
                                if len(prev_tokens) >= n:
                                    new_ngrams[n - 1].add(tuple(prev_tokens[-n:]))
                        else:
                            new_ngrams = beam["generated_ngrams"]

                        # Add new candidate
                        candidates.append(
                            {
                                "tokens": beam["tokens"] + [token],
                                "score": score,
                                "memory": beam["memory"],
                                "generated_ngrams": new_ngrams,
                            }
                        )

                # Sort candidates by score
                candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

                # Keep top-k beams
                beams = candidates[:beam_size]

                # If all beams end with EOS, we're done
                if all(beam["tokens"][-1] == self.eos_token for beam in beams):
                    break

            # Return best beam
            best_beam = max(beams, key=lambda x: x["score"])
            return best_beam["tokens"]

    def evaluate_on_test_set(
        self,
        test_loader,
        output_file: str = None,
        max_examples: Optional[int] = None,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
    ) -> Dict[str, float]:
        """
        Evaluate the model on the test set and compute BLEU and ROUGE scores.

        Args:
            test_loader: DataLoader for the test set
            output_file: Path to save evaluation results (optional)
            max_examples: Maximum number of examples to evaluate (None for all)
            beam_size: Beam size for decoding
            length_penalty: Length penalty for beam search
            no_repeat_ngram_size: If > 0, all ngrams of this size can only occur once

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token)
        total_loss = 0.0

        # For storing predictions and references
        all_predictions = []
        all_references = []
        all_sources = []

        # For progress bar
        pbar = tqdm(test_loader, desc="Evaluating")

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if max_examples is not None and len(all_sources) >= max_examples:
                    break

                # Move batch to device
                src = batch["src"].to(self.device)
                tgt_input = batch["tgt_input"].to(self.device)
                tgt_output = batch["tgt_output"].to(self.device)
                src_padding_mask = batch["src_padding_mask"].to(self.device)
                tgt_padding_mask = batch["tgt_padding_mask"].to(self.device)

                # Forward pass
                output = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                )

                # Calculate loss
                output_flat = output.view(-1, output.size(-1))
                tgt_flat = tgt_output.view(-1)
                loss = criterion(output_flat, tgt_flat)
                total_loss += loss.item()

                # Generate translations
                for i in range(src.size(0)):
                    # Get source and reference
                    src_seq = src[i].cpu().tolist()
                    ref_seq = tgt_output[i].cpu().tolist()

                    # Convert token IDs to text (in a real scenario, use the actual tokenizer)
                    src_text = " ".join(str(t) for t in src_seq if t not in [self.pad_token])
                    ref_text = " ".join(
                        str(t)
                        for t in ref_seq
                        if t not in [self.bos_token, self.eos_token, self.pad_token]
                    )

                    # Generate prediction
                    pred_text = self.translate(
                        src_text,
                        max_length=self.max_length,
                        beam_size=beam_size,
                        length_penalty=length_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                    )

                    all_sources.append(src_text)
                    all_references.append(ref_text)
                    all_predictions.append(pred_text)

                # Update progress
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "examples": len(all_sources)})

        # Calculate BLEU score
        bleu_score = self.bleu.corpus_score(
            hypotheses=all_predictions, references=[[ref] for ref in all_references]
        ).score

        # Calculate ROUGE scores
        rouge_scores = {}
        try:
            if all_predictions and all_references:
                rouge_scores = self.rouge.get_scores(all_predictions, all_references, avg=True)
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {e}")
            rouge_scores = {
                "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0},
            }

        # Calculate average loss
        avg_loss = total_loss / len(test_loader)

        # Prepare metrics
        metrics = {
            "loss": avg_loss,
            "bleu": bleu_score,
            "rouge": rouge_scores,
            "num_examples": len(all_sources),
        }

        # Save predictions if output file is provided
        if output_file:
            output_data = {
                "metrics": metrics,
                "examples": [
                    {"source": src, "reference": ref, "prediction": pred}
                    for src, ref, pred in zip(all_sources, all_references, all_predictions)
                ],
            }

            # Create directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Evaluation results saved to {output_file}")

        return metrics


def interactive_mode(translator: Translator):
    """Run the translator in interactive mode."""
    print("\nInteractive translation mode (type 'quit' to exit)")
    print("----------------------------------------")

    while True:
        try:
            # Get input
            src_text = input("\nEnter source text (or 'quit'): ").strip()

            if src_text.lower() in ["quit", "exit", "q"]:
                break

            if not src_text:
                continue

            # Get decoding parameters
            try:
                beam_size = int(
                    input(f"Beam size [{translator.beam_size}]: ") or translator.beam_size
                )
                max_length = int(
                    input(f"Max length [{translator.max_length}]: ") or translator.max_length
                )
                length_penalty = float(input("Length penalty [1.0]: ") or "1.0")
                no_repeat_ngram_size = int(input("No repeat n-gram size [3]: ") or "3")

                # Translate
                start_time = time.time()
                translation = translator.translate(
                    src_text,
                    max_length=max_length,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                elapsed = time.time() - start_time

                # Print results
                print("\nTranslation:")
                print(f"  Source: {src_text}")
                print(f"  Translation: {translation}")
                print(f"  Time: {elapsed:.2f}s")

            except ValueError as e:
                print(f"Error: {e}")
                continue

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    """
    Main function for evaluation.

    This script can be used in two modes:
    1. Batch evaluation: Evaluate the model on a test set and compute metrics
    2. Interactive mode: Translate text interactively
    """
    parser = argparse.ArgumentParser(description="Evaluate Bangla-English translation model")
    parser.add_argument(
        "--config", type=str, default="config/train.yaml", help="Path to configuration file"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results",
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument(
        "--max_examples", type=int, default=None, help="Maximum number of examples to evaluate"
    )
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument(
        "--length_penalty", type=float, default=1.0, help="Length penalty for beam search"
    )
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, default=3, help="Size of n-grams to avoid repeating"
    )

    args = parser.parse_args()

    # Initialize translator
    try:
        translator = Translator(
            config_path=args.config, checkpoint_path=args.checkpoint, beam_size=args.beam_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize translator: {e}")
        return

    # Run in interactive mode
    if args.interactive:
        interactive_mode(translator)
        return

    # Example translation
    print("\nExample translation:")
    example_src = "This is a test sentence ."
    translation = translator.translate(
        example_src,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(f"Source: {example_src}")
    print(f"Translation: {translation}")

    # In a real scenario, you would load the test dataset and evaluate
    # test_loader = ...
    # metrics = translator.evaluate_on_test_set(
    #     test_loader,
    #     output_file=args.output,
    #     max_examples=args.max_examples,
    #     beam_size=args.beam_size,
    #     length_penalty=args.length_penalty,
    #     no_repeat_ngram_size=args.no_repeat_ngram_size
    # )
    #
    # print("\nEvaluation metrics:")
    # print(f"  Loss: {metrics['loss']:.4f}")
    # print(f"  BLEU: {metrics['bleu']:.2f}")
    # print(f"  ROUGE-1 F1: {metrics['rouge']['rouge-1']['f']:.4f}")
    # print(f"  ROUGE-2 F1: {metrics['rouge']['rouge-2']['f']:.4f}")
    # print(f"  ROUGE-L F1: {metrics['rouge']['rouge-l']['f']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred during evaluation")
        sys.exit(1)
