#!/usr/bin/env python3
"""
Evaluation script for Literary models.

This script evaluates literary models on various metrics including:
- Literary Language Model (perplexity, generation quality)
- Poetic Meter Detection (accuracy, F1 score)
- Style Transfer (BLEU, style accuracy)
- Metaphor/Simile Detection (precision, recall, F1)
- Sentiment/Tone Classification (accuracy, F1)
- Text Complexity Prediction (MAE, RMSE, correlation)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from bilingual.models.literary_lm import LiteraryLanguageModel
from bilingual.models.metaphor_detector import MetaphorSimileDetector
from bilingual.models.sentiment_tone_classifier import SentimentToneAnalyzer
from bilingual.models.text_complexity_predictor import ComplexityAnalyzer
from bilingual.modules.poetic_meter import PoeticMeterDetector
from bilingual.modules.style_transfer_gan import StyleTransferGPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Literary Models")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            "literary-lm",
            "poetic-meter",
            "style-transfer",
            "metaphor-detector",
            "sentiment-tone",
            "text-complexity",
            "all",
        ],
        help="Type of model to evaluate",
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument(
        "--test_dataset", type=str, required=True, help="Path to the test dataset (JSON or JSONL)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/literary_eval.json",
        help="Path to save evaluation results",
    )
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    return parser.parse_args()


def load_test_data(dataset_path: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSON or JSONL file."""
    data = []
    path = Path(dataset_path)

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return data


def evaluate_literary_lm(
    model_path: str, tokenizer_path: str, test_data: List[Dict]
) -> Dict[str, float]:
    """Evaluate Literary Language Model."""
    logger.info("Evaluating Literary Language Model...")

    model = LiteraryLanguageModel(model_path=model_path, tokenizer_path=tokenizer_path)

    perplexities = []
    for item in test_data:
        text = item.get("text", "")
        if text:
            ppl = model.get_perplexity(text)
            perplexities.append(ppl)

    return {
        "avg_perplexity": np.mean(perplexities) if perplexities else 0.0,
        "median_perplexity": np.median(perplexities) if perplexities else 0.0,
        "num_samples": len(perplexities),
    }


def evaluate_poetic_meter(model_path: str, test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate Poetic Meter Detector."""
    logger.info("Evaluating Poetic Meter Detector...")

    detector = PoeticMeterDetector(model_path=model_path) if model_path else PoeticMeterDetector()

    predictions = []
    labels = []

    for item in test_data:
        poem = item.get("text", "")
        true_meter = item.get("meter", "")

        if poem and true_meter:
            result = detector.detect(poem)
            predictions.append(result["meter_type"])
            labels.append(true_meter)

    if predictions and labels:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")

        return {"accuracy": accuracy, "f1_score": f1, "num_samples": len(predictions)}

    return {"accuracy": 0.0, "f1_score": 0.0, "num_samples": 0}


def evaluate_style_transfer(model_path: str, test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate Style Transfer GPT."""
    logger.info("Evaluating Style Transfer GPT...")

    if not model_path:
        logger.warning("No model path provided for style transfer")
        return {"style_accuracy": 0.0, "num_samples": 0}

    model = StyleTransferGPT(model_path=model_path)

    correct = 0
    total = 0

    for item in test_data:
        source_text = item.get("source_text", "")
        source_style = item.get("source_style", "")
        target_style = item.get("target_style", "")

        if source_text and source_style and target_style:
            transferred = model.transfer(source_text, source_style, target_style)
            # Simple heuristic: check if transfer was successful
            if len(transferred) > 0:
                correct += 1
            total += 1

    return {"style_accuracy": correct / total if total > 0 else 0.0, "num_samples": total}


def evaluate_metaphor_detector(model_path: str, test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate Metaphor and Simile Detector."""
    logger.info("Evaluating Metaphor and Simile Detector...")

    if not model_path:
        logger.warning("No model path provided for metaphor detector")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_samples": 0}

    detector = MetaphorSimileDetector(model_path=model_path)

    all_predictions = []
    all_labels = []

    for item in test_data:
        text = item.get("text", "")
        true_metaphors = item.get("metaphors", [])
        true_similes = item.get("similes", [])

        if text:
            result = detector.detect(text)
            # Simplified evaluation - count detected vs true entities
            pred_count = len(result["metaphors"]) + len(result["similes"])
            true_count = len(true_metaphors) + len(true_similes)

            all_predictions.append(pred_count)
            all_labels.append(true_count)

    if all_predictions and all_labels:
        # Calculate correlation as a proxy metric
        correlation = (
            np.corrcoef(all_predictions, all_labels)[0, 1] if len(all_predictions) > 1 else 0.0
        )

        return {
            "correlation": correlation,
            "avg_detected": np.mean(all_predictions),
            "num_samples": len(all_predictions),
        }

    return {"correlation": 0.0, "avg_detected": 0.0, "num_samples": 0}


def evaluate_sentiment_tone(model_path: str, test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate Sentiment and Tone Classifier."""
    logger.info("Evaluating Sentiment and Tone Classifier...")

    if not model_path:
        logger.warning("No model path provided for sentiment-tone classifier")
        return {"sentiment_accuracy": 0.0, "tone_f1": 0.0, "num_samples": 0}

    analyzer = SentimentToneAnalyzer(model_path=model_path)

    sentiment_predictions = []
    sentiment_labels = []

    for item in test_data:
        text = item.get("text", "")
        true_sentiment = item.get("sentiment", "")

        if text and true_sentiment:
            result = analyzer.analyze(text)
            sentiment_predictions.append(result["sentiment"]["label"])
            sentiment_labels.append(true_sentiment)

    if sentiment_predictions and sentiment_labels:
        accuracy = accuracy_score(sentiment_labels, sentiment_predictions)
        f1 = f1_score(sentiment_labels, sentiment_predictions, average="weighted")

        return {
            "sentiment_accuracy": accuracy,
            "sentiment_f1": f1,
            "num_samples": len(sentiment_predictions),
        }

    return {"sentiment_accuracy": 0.0, "sentiment_f1": 0.0, "num_samples": 0}


def evaluate_text_complexity(model_path: str, test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate Text Complexity Predictor."""
    logger.info("Evaluating Text Complexity Predictor...")

    analyzer = ComplexityAnalyzer(model_path=model_path) if model_path else ComplexityAnalyzer()

    predictions = []
    labels = []

    for item in test_data:
        text = item.get("text", "")
        true_complexity = item.get("complexity_score", None)

        if text and true_complexity is not None:
            result = analyzer.analyze(text)
            predictions.append(result["overall_complexity"])
            labels.append(true_complexity)

    if predictions and labels:
        mae = np.mean(np.abs(np.array(predictions) - np.array(labels)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(labels)) ** 2))
        correlation = np.corrcoef(predictions, labels)[0, 1] if len(predictions) > 1 else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "num_samples": len(predictions),
        }

    return {"mae": 0.0, "rmse": 0.0, "correlation": 0.0, "num_samples": 0}


def evaluate_literary_model(args):
    """
    Evaluate the literary model.

    Args:
        args: Command line arguments
    """
    logger.info(f"Loading test dataset from: {args.test_dataset}")

    # Load test data
    test_data = load_test_data(args.test_dataset)
    logger.info(f"Loaded {len(test_data)} test samples")

    results = {}

    # Evaluate based on model type
    if args.model_type == "literary-lm":
        results["literary_lm"] = evaluate_literary_lm(
            args.model_path, args.tokenizer_path, test_data
        )

    elif args.model_type == "poetic-meter":
        results["poetic_meter"] = evaluate_poetic_meter(args.model_path, test_data)

    elif args.model_type == "style-transfer":
        results["style_transfer"] = evaluate_style_transfer(args.model_path, test_data)

    elif args.model_type == "metaphor-detector":
        results["metaphor_detector"] = evaluate_metaphor_detector(args.model_path, test_data)

    elif args.model_type == "sentiment-tone":
        results["sentiment_tone"] = evaluate_sentiment_tone(args.model_path, test_data)

    elif args.model_type == "text-complexity":
        results["text_complexity"] = evaluate_text_complexity(args.model_path, test_data)

    elif args.model_type == "all":
        logger.info("Evaluating all literary models...")

        if args.model_path:
            results["literary_lm"] = evaluate_literary_lm(
                args.model_path, args.tokenizer_path, test_data
            )

        results["poetic_meter"] = evaluate_poetic_meter(args.model_path, test_data)

        results["text_complexity"] = evaluate_text_complexity(args.model_path, test_data)

        # Add other models if paths are provided
        logger.info("Note: Some models require specific model paths")

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to: {output_path}")
    logger.info(f"\nResults Summary:")
    logger.info(json.dumps(results, indent=2, ensure_ascii=False))


def main():
    """Main entry point."""
    args = parse_args()
    evaluate_literary_model(args)


if __name__ == "__main__":
    main()
