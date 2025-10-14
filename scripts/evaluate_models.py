#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Benchmarking Script.

This script evaluates bilingual models across multiple metrics:
- Perplexity (language modeling)
- BLEU, chrF, COMET (translation)
- Accuracy, F1 (classification)
- Language parity metrics
- Inference speed benchmarks

Usage:
    python scripts/evaluate_models.py \
        --model models/bilingual-lm/ \
        --test-data datasets/processed/final/test.jsonl \
        --task generation \
        --output results/evaluation_report.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual import bilingual_api as bb  # noqa: E402
from bilingual.data_utils import BilingualDataset  # noqa: E402
from bilingual.evaluation import (  # noqa: E402
    compute_bleu,
    compute_f1,
    compute_perplexity,
    compute_rouge,
)


class ModelEvaluator:
    """Comprehensive model evaluation system."""

    def __init__(
        self,
        model_name: str,
        test_data_path: str,
        task: str = "generation",
        output_file: Optional[str] = None
    ):
        """
        Initialize evaluator.

        Args:
            model_name: Name or path to model
            test_data_path: Path to test dataset
            task: Task type (generation, translation, classification)
            output_file: Path to save evaluation results
        """
        self.model_name = model_name
        self.test_data_path = test_data_path
        self.task = task
        self.output_file = output_file

        # Load test data
        print(f"Loading test data from: {test_data_path}")
        self.test_data = BilingualDataset(file_path=test_data_path)
        print(f"Loaded {len(self.test_data)} test samples\n")

        # Results storage
        self.results = {
            'model': model_name,
            'task': task,
            'test_samples': len(self.test_data),
            'metrics': {},
            'per_language_metrics': {},
            'examples': [],
            'inference_stats': {},
        }

    def evaluate_generation(self) -> Dict:
        """
        Evaluate language generation model.

        Returns:
            Dictionary of generation metrics
        """
        print("=" * 60)
        print("EVALUATING GENERATION MODEL")
        print("=" * 60)
        print()

        metrics = {
            'perplexity_overall': 0.0,
            'perplexity_bn': 0.0,
            'perplexity_en': 0.0,
            'avg_generation_time': 0.0,
            'tokens_per_second': 0.0,
        }

        # Group samples by language
        samples_by_lang = defaultdict(list)
        for sample in self.test_data:
            lang = sample.get('language', 'mixed')
            samples_by_lang[lang].append(sample)

        print(f"Samples by language:")
        for lang, samples in samples_by_lang.items():
            print(f"  {lang}: {len(samples)}")
        print()

        # Calculate perplexity
        print("Calculating perplexity...")
        try:
            texts = [s['text'] for s in self.test_data]
            metrics['perplexity_overall'] = compute_perplexity(texts)
            print(f"  Overall perplexity: {metrics['perplexity_overall']:.2f}")

            # Per-language perplexity
            for lang in ['bn', 'en']:
                if lang in samples_by_lang:
                    lang_texts = [s['text'] for s in samples_by_lang[lang]]
                    perp = compute_perplexity(lang_texts)
                    metrics[f'perplexity_{lang}'] = perp
                    print(f"  {lang.upper()} perplexity: {perp:.2f}")

        except Exception as e:
            print(f"  Warning: Could not calculate perplexity: {e}")

        print()

        # Benchmark inference speed
        print("Benchmarking inference speed...")
        generation_times = []
        sample_prompts = [
            "আমার নাম",
            "Once upon a time",
            "বাংলাদেশ একটি",
            "The quick brown"
        ]

        for prompt in sample_prompts[:3]:
            try:
                start_time = time.time()
                _ = bb.generate(prompt, model_name=self.model_name, max_tokens=50)
                elapsed = time.time() - start_time
                generation_times.append(elapsed)
            except Exception as e:
                print(f"  Warning: Generation failed for '{prompt}': {e}")

        if generation_times:
            metrics['avg_generation_time'] = np.mean(generation_times)
            metrics['tokens_per_second'] = 50 / metrics['avg_generation_time']
            print(f"  Avg generation time: {metrics['avg_generation_time']:.3f}s")
            print(f"  Tokens/second: {metrics['tokens_per_second']:.1f}")
        print()

        # Sample generations
        print("Sample generations:")
        for i, prompt in enumerate(sample_prompts[:2], 1):
            try:
                generated = bb.generate(prompt, model_name=self.model_name, max_tokens=30)
                print(f"  {i}. Prompt: {prompt}")
                print(f"     Generated: {generated}")
                self.results['examples'].append({
                    'prompt': prompt,
                    'generated': generated
                })
            except Exception as e:
                print(f"  {i}. Prompt: {prompt}")
                print(f"     Error: {e}")
        print()

        return metrics

    def evaluate_translation(self) -> Dict:
        """
        Evaluate translation model.

        Returns:
            Dictionary of translation metrics
        """
        print("=" * 60)
        print("EVALUATING TRANSLATION MODEL")
        print("=" * 60)
        print()

        metrics = {
            'bleu_bn_to_en': 0.0,
            'bleu_en_to_bn': 0.0,
            'rouge_l_bn_to_en': 0.0,
            'rouge_l_en_to_bn': 0.0,
            'avg_translation_time': 0.0,
        }

        # Filter samples with parallel text
        parallel_samples = [
            s for s in self.test_data
            if 'bn_text' in s and 'en_text' in s
        ]

        if not parallel_samples:
            print("Warning: No parallel samples found in test data")
            return metrics

        print(f"Found {len(parallel_samples)} parallel samples\n")

        # Evaluate BN -> EN translation
        print("Evaluating Bangla → English translation...")
        bn_sources = [s['bn_text'] for s in parallel_samples[:50]]  # Sample 50
        en_references = [s['en_text'] for s in parallel_samples[:50]]

        bn_to_en_predictions = []
        translation_times = []

        for bn_text in bn_sources:
            try:
                start_time = time.time()
                prediction = bb.translate(bn_text, src='bn', tgt='en', model_name=self.model_name)
                elapsed = time.time() - start_time
                translation_times.append(elapsed)
                bn_to_en_predictions.append(prediction)
            except Exception as e:
                print(f"  Warning: Translation failed: {e}")
                bn_to_en_predictions.append("")

        if bn_to_en_predictions:
            try:
                bleu_score = compute_bleu(bn_to_en_predictions, en_references)
                metrics['bleu_bn_to_en'] = bleu_score
                print(f"  BLEU (BN→EN): {bleu_score:.3f}")

                rouge_scores = compute_rouge(bn_to_en_predictions, en_references)
                metrics['rouge_l_bn_to_en'] = rouge_scores.get('rouge-l', 0.0)
                print(f"  ROUGE-L (BN→EN): {metrics['rouge_l_bn_to_en']:.3f}")
            except Exception as e:
                print(f"  Warning: Could not compute translation metrics: {e}")

        # Evaluate EN -> BN translation
        print("\nEvaluating English → Bangla translation...")
        en_sources = [s['en_text'] for s in parallel_samples[:50]]
        bn_references = [s['bn_text'] for s in parallel_samples[:50]]

        en_to_bn_predictions = []
        for en_text in en_sources:
            try:
                prediction = bb.translate(en_text, src='en', tgt='bn', model_name=self.model_name)
                en_to_bn_predictions.append(prediction)
            except Exception as e:
                print(f"  Warning: Translation failed: {e}")
                en_to_bn_predictions.append("")

        if en_to_bn_predictions:
            try:
                bleu_score = compute_bleu(en_to_bn_predictions, bn_references)
                metrics['bleu_en_to_bn'] = bleu_score
                print(f"  BLEU (EN→BN): {bleu_score:.3f}")

                rouge_scores = compute_rouge(en_to_bn_predictions, bn_references)
                metrics['rouge_l_en_to_bn'] = rouge_scores.get('rouge-l', 0.0)
                print(f"  ROUGE-L (EN→BN): {metrics['rouge_l_en_to_bn']:.3f}")
            except Exception as e:
                print(f"  Warning: Could not compute translation metrics: {e}")

        # Average translation time
        if translation_times:
            metrics['avg_translation_time'] = np.mean(translation_times)
            print(f"\n  Avg translation time: {metrics['avg_translation_time']:.3f}s")

        # Sample translations
        print("\nSample translations:")
        for i in range(min(3, len(parallel_samples))):
            sample = parallel_samples[i]
            print(f"  {i+1}. BN: {sample['bn_text'][:100]}...")
            print(f"     EN: {sample['en_text'][:100]}...")
            try:
                pred = bb.translate(sample['bn_text'], src='bn', tgt='en', model_name=self.model_name)
                print(f"     Predicted: {pred[:100]}...")
            except Exception as e:
                print(f"     Prediction failed: {e}")
            print()

        return metrics

    def evaluate_classification(self) -> Dict:
        """
        Evaluate classification model.

        Returns:
            Dictionary of classification metrics
        """
        print("=" * 60)
        print("EVALUATING CLASSIFICATION MODEL")
        print("=" * 60)
        print()

        metrics = {
            'accuracy_overall': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'accuracy_bn': 0.0,
            'accuracy_en': 0.0,
        }

        # Check if test data has labels
        labeled_samples = [s for s in self.test_data if 'label' in s]
        if not labeled_samples:
            print("Warning: No labeled samples found for classification evaluation")
            return metrics

        print(f"Found {len(labeled_samples)} labeled samples\n")

        # Get predictions and true labels
        predictions = []
        true_labels = []

        print("Getting predictions...")
        for sample in labeled_samples[:100]:  # Sample 100 for speed
            text = sample['text']
            true_label = sample['label']

            try:
                # Use readability or safety check as classification
                if self.task == 'readability':
                    result = bb.readability_check(text)
                    pred_label = result.get('level', 'unknown')
                elif self.task == 'safety':
                    result = bb.safety_check(text)
                    pred_label = 'safe' if result.get('is_safe', False) else 'unsafe'
                else:
                    pred_label = 'unknown'

                predictions.append(pred_label)
                true_labels.append(true_label)

            except Exception as e:
                print(f"  Warning: Classification failed: {e}")

        if predictions and true_labels:
            try:
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score
                
                metrics['accuracy_overall'] = accuracy_score(true_labels, predictions)
                metrics['f1_macro'] = f1_score(true_labels, predictions, average='macro')
                metrics['f1_weighted'] = f1_score(true_labels, predictions, average='weighted')

                print(f"  Overall accuracy: {metrics['accuracy_overall']:.3f}")
                print(f"  F1 (macro): {metrics['f1_macro']:.3f}")
                print(f"  F1 (weighted): {metrics['f1_weighted']:.3f}")

            except ImportError:
                print("  Warning: scikit-learn not available for detailed metrics")
            except Exception as e:
                print(f"  Warning: Could not compute classification metrics: {e}")

        return metrics

    def run_evaluation(self) -> Dict:
        """
        Run complete evaluation based on task type.

        Returns:
            Complete evaluation results
        """
        print(f"Starting evaluation for task: {self.task}")
        print(f"Model: {self.model_name}")
        print(f"Test samples: {len(self.test_data)}")
        print()

        # Run task-specific evaluation
        if self.task == 'generation':
            self.results['metrics'] = self.evaluate_generation()
        elif self.task == 'translation':
            self.results['metrics'] = self.evaluate_translation()
        elif self.task in ['classification', 'readability', 'safety']:
            self.results['metrics'] = self.evaluate_classification()
        else:
            print(f"Warning: Unknown task '{self.task}', running generation evaluation")
            self.results['metrics'] = self.evaluate_generation()

        # Language parity analysis
        self.analyze_language_parity()

        # Save results
        if self.output_file:
            self.save_results()

        return self.results

    def analyze_language_parity(self):
        """Analyze performance parity between languages."""
        print("=" * 60)
        print("LANGUAGE PARITY ANALYSIS")
        print("=" * 60)
        print()

        # Group samples by language
        samples_by_lang = defaultdict(list)
        for sample in self.test_data:
            lang = sample.get('language', 'mixed')
            samples_by_lang[lang].append(sample)

        parity_metrics = {}

        # Compare metrics between languages
        metrics = self.results['metrics']
        
        if 'perplexity_bn' in metrics and 'perplexity_en' in metrics:
            bn_perp = metrics['perplexity_bn']
            en_perp = metrics['perplexity_en']
            if en_perp > 0:
                parity_ratio = bn_perp / en_perp
                parity_metrics['perplexity_parity'] = parity_ratio
                print(f"Perplexity parity (BN/EN): {parity_ratio:.3f}")
                if abs(parity_ratio - 1.0) < 0.1:
                    print("  ✓ Good parity (within 10%)")
                else:
                    print("  ⚠ Parity concern (>10% difference)")

        if 'bleu_bn_to_en' in metrics and 'bleu_en_to_bn' in metrics:
            bn_to_en = metrics['bleu_bn_to_en']
            en_to_bn = metrics['bleu_en_to_bn']
            if en_to_bn > 0:
                translation_parity = bn_to_en / en_to_bn
                parity_metrics['translation_parity'] = translation_parity
                print(f"Translation parity (BN→EN / EN→BN): {translation_parity:.3f}")

        self.results['per_language_metrics'] = parity_metrics
        print()

    def save_results(self):
        """Save evaluation results to file."""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Evaluate bilingual models comprehensively'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name or path to evaluate'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test dataset (JSONL format)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='generation',
        choices=['generation', 'translation', 'classification', 'readability', 'safety'],
        help='Task type for evaluation'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for evaluation results (JSON)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100,
        help='Number of samples to evaluate (for speed)'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_name=args.model,
        test_data_path=args.test_data,
        task=args.task,
        output_file=args.output
    )

    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.3f}")
            else:
                print(f"{metric}: {value}")
                
        if results['per_language_metrics']:
            print("\nLanguage Parity:")
            for metric, value in results['per_language_metrics'].items():
                print(f"{metric}: {value:.3f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
