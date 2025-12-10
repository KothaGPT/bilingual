#!/usr/bin/env python3
"""
Metaphor and Simile Prediction Script.

This script loads a trained model and makes predictions on new text.

Usage:
    python scripts/predict_metaphors.py \
        --model_path models/metaphor-simile-detector/checkpoints \
        --input_text "Life is a journey with many paths to choose from." \
        --output_file predictions.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MetaphorDetector:
    """A class to handle metaphor and simile detection using a trained model."""

    def __init__(self, model_path: str, device: str = None):
        """Initialize the detector with a trained model.

        Args:
            model_path: Path to the directory containing the trained model
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_path = Path(model_path)
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        self.pipeline = None

        self._load_model()
        self._setup_pipeline()

    def _load_model(self):
        """Load the model, tokenizer, and label mappings."""
        logger.info(f"Loading model from {self.model_path}")

        # Load model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Move model to the appropriate device
        self.model.to(self.device)

        # Load label mappings
        label_map_path = self.model_path / "label_mapping.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_mapping = json.load(f)
            self.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
            self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            logger.warning("label_mapping.json not found. Using default label mapping.")
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

    def _setup_pipeline(self):
        """Set up the NER pipeline for inference."""
        self.pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple",
        )

    def predict(self, text: str) -> List[Dict]:
        """Make predictions on a single text.

        Args:
            text: Input text to analyze

        Returns:
            List of dictionaries containing prediction results
        """
        if not text.strip():
            return []

        # Make prediction
        results = self.pipeline(text)

        # Format results
        predictions = self._process_predictions(results)
        return predictions

    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Make predictions on a batch of texts.

        Args:
            texts: List of input texts to analyze

        Returns:
            List of prediction results for each input text
        """
        if not texts:
            return []

        # Filter out empty strings
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []

        # Make batch prediction
        results = self.pipeline(texts)

        # Process results
        all_predictions = []
        for result in results:
            if not result:  # Handle case when no entities are found
                all_predictions.append([])
                continue

            # If results are not already grouped by text, process as single text
            if not isinstance(result[0], list):
                all_predictions.append(self._process_predictions(result))
            else:
                # Results are already grouped by text
                all_predictions.extend([self._process_predictions(r) for r in result])

        return all_predictions

    def _process_predictions(self, entities: List[Dict]) -> List[Dict]:
        """Process raw model predictions into a more usable format.

        Args:
            entities: List of entity predictions from the model

        Returns:
            List of processed predictions
        """
        predictions = []
        current_entity = None

        for entity in entities:
            label = entity["entity_group"]

            if label.startswith("B-"):
                if current_entity:
                    predictions.append(current_entity)
                current_entity = {
                    "entity": label[2:],  # Remove B- or I- prefix
                    "score": entity["score"],
                    "word": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                }
            elif label.startswith("I-") and current_entity:
                # Handle multi-word entities
                current_entity["word"] += " " + entity["word"]
                current_entity["end"] = entity["end"]
                # Update score as average of all tokens
                current_entity["score"] = (current_entity["score"] + entity["score"]) / 2

        # Add the last entity if it exists
        if current_entity:
            predictions.append(current_entity)

        return predictions


def predict_from_file(
    model_path: str,
    input_file: str,
    output_file: Optional[str] = None,
    batch_size: int = 8,
    device: str = None,
) -> List[Dict]:
    """Make predictions on texts from a file.

    Args:
        model_path: Path to the trained model
        input_file: Path to input text file (one example per line)
        output_file: Optional path to save predictions
        batch_size: Number of texts to process in each batch
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        List of prediction results
    """
    # Read input file
    with open(input_file) as f:
        texts = [line.strip() for line in f if line.strip()]

    if not texts:
        logger.warning(f"No valid texts found in {input_file}")
        return []

    # Initialize detector
    detector = MetaphorDetector(model_path, device=device)

    # Process in batches
    all_predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_predictions = detector.predict_batch(batch)
        all_predictions.extend(batch_predictions)
        logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    # Prepare results
    results = [{"text": text, "predictions": preds} for text, preds in zip(texts, all_predictions)]

    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions with the metaphor-simile detector."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        help="Input text to analyze",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to a text file with one example per line",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save predictions (JSON format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing multiple texts (default: 8)",
    )

    args = parser.parse_args()

    if not (args.input_text or args.input_file):
        parser.error("Either --input_text or --input_file must be provided")

    try:
        if args.input_text:
            # Single text prediction
            detector = MetaphorDetector(args.model_path, device=args.device)
            predictions = detector.predict(args.input_text)

            # Prepare result
            result = {"text": args.input_text, "predictions": predictions}

            # Print to console
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # Save to file if specified
            if args.output_file:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    json.dump([result], f, indent=2, ensure_ascii=False)
                logger.info(f"Predictions saved to {args.output_file}")

        elif args.input_file:
            # Batch prediction from file
            predict_from_file(
                model_path=args.model_path,
                input_file=args.input_file,
                output_file=args.output_file,
                batch_size=args.batch_size,
                device=args.device,
            )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
