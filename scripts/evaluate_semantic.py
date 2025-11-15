#!/usr/bin/env python3
"""
Evaluation script for Semantic models.

This script evaluates semantic models including:
- Cross-Lingual Embeddings (similarity correlation, retrieval accuracy)
- Named Entity Recognition (precision, recall, F1)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from bilingual.models.cross_lingual_embeddings import CrossLingualEmbeddings
from bilingual.models.named_entity_recognizer import BanglaNER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Semantic Models")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["cross-lingual", "ner", "all"],
        help="Type of model to evaluate",
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument(
        "--test_dataset", type=str, required=True, help="Path to the test dataset (JSON or JSONL)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/semantic_eval.json",
        help="Path to save evaluation results",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
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


def evaluate_cross_lingual_embeddings(
    model_path: str, test_data: List[Dict], batch_size: int = 32
) -> Dict[str, float]:
    """Evaluate Cross-Lingual Embeddings."""
    logger.info("Evaluating Cross-Lingual Embeddings...")

    if not model_path:
        logger.warning("No model path provided for cross-lingual embeddings")
        return {"similarity_correlation": 0.0, "retrieval_accuracy": 0.0, "num_samples": 0}

    embeddings = CrossLingualEmbeddings(model_path=model_path)

    # Evaluate similarity correlation
    similarities_pred = []
    similarities_true = []

    for item in test_data:
        sent1 = item.get("sentence1", "")
        sent2 = item.get("sentence2", "")
        true_similarity = item.get("similarity", None)

        if sent1 and sent2 and true_similarity is not None:
            pred_similarity = embeddings.similarity(sent1, sent2)
            similarities_pred.append(pred_similarity)
            similarities_true.append(true_similarity)

    # Calculate correlation
    if len(similarities_pred) > 1:
        correlation = np.corrcoef(similarities_pred, similarities_true)[0, 1]
    else:
        correlation = 0.0

    # Evaluate retrieval accuracy
    retrieval_correct = 0
    retrieval_total = 0

    for item in test_data:
        query = item.get("query", "")
        candidates = item.get("candidates", [])
        true_match_idx = item.get("match_index", None)

        if query and candidates and true_match_idx is not None:
            results = embeddings.find_similar(query, candidates, top_k=1)
            if results:
                pred_match = results[0][0]
                if pred_match == candidates[true_match_idx]:
                    retrieval_correct += 1
                retrieval_total += 1

    retrieval_accuracy = retrieval_correct / retrieval_total if retrieval_total > 0 else 0.0

    return {
        "similarity_correlation": correlation,
        "retrieval_accuracy": retrieval_accuracy,
        "num_similarity_pairs": len(similarities_pred),
        "num_retrieval_queries": retrieval_total,
    }


def evaluate_ner(model_path: str, test_data: List[Dict]) -> Dict[str, float]:
    """Evaluate Named Entity Recognition."""
    logger.info("Evaluating Named Entity Recognition...")

    if not model_path:
        logger.warning("No model path provided for NER")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_samples": 0}

    ner = BanglaNER(model_path=model_path)

    all_predictions = []
    all_labels = []

    for item in test_data:
        text = item.get("text", "")
        true_entities = item.get("entities", [])

        if text:
            result = ner.recognize(text)
            pred_entities = result["entities"]

            # Convert to entity type lists for evaluation
            pred_types = [e["type"] for e in pred_entities]
            true_types = [e["type"] for e in true_entities]

            # Pad to same length for comparison
            max_len = max(len(pred_types), len(true_types))
            pred_types += ["O"] * (max_len - len(pred_types))
            true_types += ["O"] * (max_len - len(true_types))

            all_predictions.extend(pred_types)
            all_labels.extend(true_types)

    if all_predictions and all_labels:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted", zero_division=0
        )

        return {"precision": precision, "recall": recall, "f1": f1, "num_samples": len(test_data)}

    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_samples": 0}


def evaluate_semantic_models(args):
    """
    Evaluate semantic models.

    Args:
        args: Command line arguments
    """
    logger.info(f"Loading test dataset from: {args.test_dataset}")

    # Load test data
    test_data = load_test_data(args.test_dataset)
    logger.info(f"Loaded {len(test_data)} test samples")

    results = {}

    # Evaluate based on model type
    if args.model_type == "cross-lingual":
        results["cross_lingual_embeddings"] = evaluate_cross_lingual_embeddings(
            args.model_path, test_data, args.batch_size
        )

    elif args.model_type == "ner":
        results["named_entity_recognition"] = evaluate_ner(args.model_path, test_data)

    elif args.model_type == "all":
        logger.info("Evaluating all semantic models...")

        if args.model_path:
            results["cross_lingual_embeddings"] = evaluate_cross_lingual_embeddings(
                args.model_path, test_data, args.batch_size
            )

            results["named_entity_recognition"] = evaluate_ner(args.model_path, test_data)
        else:
            logger.warning("Model path required for evaluation")

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
    evaluate_semantic_models(args)


if __name__ == "__main__":
    main()
