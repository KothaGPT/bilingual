#!/usr/bin/env python3
"""
Model Benchmarking and Inference Performance Script.

This script benchmarks bilingual models for:
- Inference speed (tokens/second)
- Memory usage
- Latency measurements
- Throughput testing
- Batch processing performance
- Resource utilization

Usage:
    python scripts/benchmark_models.py \
        --models models/bilingual-lm/ models/translation/ \
        --tasks generation translation \
        --output results/benchmark_report.json
"""

import argparse
import json
import psutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual import bilingual_api as bb  # noqa: E402


class ModelBenchmark:
    """Benchmark bilingual models for performance metrics."""

    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize benchmarker.

        Args:
            output_file: Path to save benchmark results
        """
        self.output_file = output_file
        self.results = {
            'benchmark_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'system_info': self.get_system_info()
            },
            'models': {}
        }

    def get_system_info(self) -> Dict:
        """Get system information for benchmarking context."""
        try:
            import torch
            gpu_info = {
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except ImportError:
            gpu_info = {'cuda_available': False, 'gpu_count': 0, 'gpu_name': None}

        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': sys.version.split()[0],
            **gpu_info
        }

    def benchmark_generation(
        self,
        model_name: str,
        prompts: List[str],
        max_tokens: int = 50,
        batch_sizes: List[int] = [1, 4, 8]
    ) -> Dict:
        """
        Benchmark text generation performance.

        Args:
            model_name: Model to benchmark
            prompts: List of prompts to test
            max_tokens: Maximum tokens to generate
            batch_sizes: Batch sizes to test

        Returns:
            Generation benchmark results
        """
        print(f"Benchmarking generation for: {model_name}")
        
        results = {
            'task': 'generation',
            'model': model_name,
            'max_tokens': max_tokens,
            'batch_performance': {},
            'latency_stats': {},
            'memory_usage': {},
            'error_rate': 0.0
        }

        total_attempts = 0
        total_errors = 0

        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            batch_prompts = prompts[:batch_size] * (batch_size // len(prompts) + 1)
            batch_prompts = batch_prompts[:batch_size]

            # Warm up
            try:
                _ = bb.generate(batch_prompts[0], model_name=model_name, max_tokens=10)
            except Exception:
                pass

            # Memory before
            memory_before = psutil.virtual_memory().used / (1024**2)  # MB

            # Benchmark
            latencies = []
            successful_generations = 0

            for i in range(min(10, len(batch_prompts))):  # Test up to 10 samples
                prompt = batch_prompts[i]
                total_attempts += 1

                try:
                    start_time = time.time()
                    generated = bb.generate(
                        prompt,
                        model_name=model_name,
                        max_tokens=max_tokens
                    )
                    end_time = time.time()

                    if generated:  # Successful generation
                        latency = end_time - start_time
                        latencies.append(latency)
                        successful_generations += 1
                    else:
                        total_errors += 1

                except Exception as e:
                    print(f"    Generation error: {e}")
                    total_errors += 1

            # Memory after
            memory_after = psutil.virtual_memory().used / (1024**2)  # MB
            memory_used = memory_after - memory_before

            # Calculate metrics
            if latencies:
                avg_latency = np.mean(latencies)
                tokens_per_second = max_tokens / avg_latency if avg_latency > 0 else 0
                throughput = batch_size / avg_latency if avg_latency > 0 else 0

                results['batch_performance'][str(batch_size)] = {
                    'avg_latency_ms': avg_latency * 1000,
                    'tokens_per_second': tokens_per_second,
                    'throughput_requests_per_second': throughput,
                    'successful_generations': successful_generations,
                    'memory_used_mb': memory_used
                }

                print(f"    Avg latency: {avg_latency*1000:.1f}ms")
                print(f"    Tokens/sec: {tokens_per_second:.1f}")
                print(f"    Memory used: {memory_used:.1f}MB")

        # Overall latency statistics
        all_latencies = []
        for batch_data in results['batch_performance'].values():
            if 'avg_latency_ms' in batch_data:
                all_latencies.append(batch_data['avg_latency_ms'] / 1000)

        if all_latencies:
            results['latency_stats'] = {
                'mean_latency_ms': np.mean(all_latencies) * 1000,
                'median_latency_ms': np.median(all_latencies) * 1000,
                'p95_latency_ms': np.percentile(all_latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(all_latencies, 99) * 1000
            }

        # Error rate
        results['error_rate'] = total_errors / total_attempts if total_attempts > 0 else 0

        print(f"  Error rate: {results['error_rate']*100:.1f}%")
        print()

        return results

    def benchmark_translation(
        self,
        model_name: str,
        text_pairs: List[Dict],
        batch_sizes: List[int] = [1, 4, 8]
    ) -> Dict:
        """
        Benchmark translation performance.

        Args:
            model_name: Model to benchmark
            text_pairs: List of {'text': str, 'src': str, 'tgt': str}
            batch_sizes: Batch sizes to test

        Returns:
            Translation benchmark results
        """
        print(f"Benchmarking translation for: {model_name}")
        
        results = {
            'task': 'translation',
            'model': model_name,
            'batch_performance': {},
            'latency_stats': {},
            'memory_usage': {},
            'error_rate': 0.0
        }

        total_attempts = 0
        total_errors = 0

        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            batch_pairs = text_pairs[:batch_size] * (batch_size // len(text_pairs) + 1)
            batch_pairs = batch_pairs[:batch_size]

            # Warm up
            try:
                first_pair = batch_pairs[0]
                _ = bb.translate(
                    first_pair['text'][:50],
                    src=first_pair['src'],
                    tgt=first_pair['tgt'],
                    model_name=model_name
                )
            except Exception:
                pass

            # Memory before
            memory_before = psutil.virtual_memory().used / (1024**2)  # MB

            # Benchmark
            latencies = []
            successful_translations = 0

            for i in range(min(10, len(batch_pairs))):  # Test up to 10 samples
                pair = batch_pairs[i]
                total_attempts += 1

                try:
                    start_time = time.time()
                    translated = bb.translate(
                        pair['text'],
                        src=pair['src'],
                        tgt=pair['tgt'],
                        model_name=model_name
                    )
                    end_time = time.time()

                    if translated:  # Successful translation
                        latency = end_time - start_time
                        latencies.append(latency)
                        successful_translations += 1
                    else:
                        total_errors += 1

                except Exception as e:
                    print(f"    Translation error: {e}")
                    total_errors += 1

            # Memory after
            memory_after = psutil.virtual_memory().used / (1024**2)  # MB
            memory_used = memory_after - memory_before

            # Calculate metrics
            if latencies:
                avg_latency = np.mean(latencies)
                throughput = batch_size / avg_latency if avg_latency > 0 else 0

                results['batch_performance'][str(batch_size)] = {
                    'avg_latency_ms': avg_latency * 1000,
                    'throughput_translations_per_second': throughput,
                    'successful_translations': successful_translations,
                    'memory_used_mb': memory_used
                }

                print(f"    Avg latency: {avg_latency*1000:.1f}ms")
                print(f"    Throughput: {throughput:.1f} translations/sec")
                print(f"    Memory used: {memory_used:.1f}MB")

        # Overall latency statistics
        all_latencies = []
        for batch_data in results['batch_performance'].values():
            if 'avg_latency_ms' in batch_data:
                all_latencies.append(batch_data['avg_latency_ms'] / 1000)

        if all_latencies:
            results['latency_stats'] = {
                'mean_latency_ms': np.mean(all_latencies) * 1000,
                'median_latency_ms': np.median(all_latencies) * 1000,
                'p95_latency_ms': np.percentile(all_latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(all_latencies, 99) * 1000
            }

        # Error rate
        results['error_rate'] = total_errors / total_attempts if total_attempts > 0 else 0

        print(f"  Error rate: {results['error_rate']*100:.1f}%")
        print()

        return results

    def benchmark_classification(
        self,
        model_name: str,
        texts: List[str],
        task: str = 'readability',
        batch_sizes: List[int] = [1, 8, 16]
    ) -> Dict:
        """
        Benchmark classification performance.

        Args:
            model_name: Model to benchmark
            texts: List of texts to classify
            task: Classification task (readability, safety)
            batch_sizes: Batch sizes to test

        Returns:
            Classification benchmark results
        """
        print(f"Benchmarking {task} classification for: {model_name}")
        
        results = {
            'task': f'{task}_classification',
            'model': model_name,
            'batch_performance': {},
            'latency_stats': {},
            'memory_usage': {},
            'error_rate': 0.0
        }

        total_attempts = 0
        total_errors = 0

        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            batch_texts = texts[:batch_size] * (batch_size // len(texts) + 1)
            batch_texts = batch_texts[:batch_size]

            # Warm up
            try:
                if task == 'readability':
                    _ = bb.readability_check(batch_texts[0][:100])
                elif task == 'safety':
                    _ = bb.safety_check(batch_texts[0][:100])
            except Exception:
                pass

            # Memory before
            memory_before = psutil.virtual_memory().used / (1024**2)  # MB

            # Benchmark
            latencies = []
            successful_classifications = 0

            for i in range(min(20, len(batch_texts))):  # Test up to 20 samples
                text = batch_texts[i]
                total_attempts += 1

                try:
                    start_time = time.time()
                    
                    if task == 'readability':
                        result = bb.readability_check(text)
                    elif task == 'safety':
                        result = bb.safety_check(text)
                    else:
                        result = None
                    
                    end_time = time.time()

                    if result:  # Successful classification
                        latency = end_time - start_time
                        latencies.append(latency)
                        successful_classifications += 1
                    else:
                        total_errors += 1

                except Exception as e:
                    print(f"    Classification error: {e}")
                    total_errors += 1

            # Memory after
            memory_after = psutil.virtual_memory().used / (1024**2)  # MB
            memory_used = memory_after - memory_before

            # Calculate metrics
            if latencies:
                avg_latency = np.mean(latencies)
                throughput = batch_size / avg_latency if avg_latency > 0 else 0

                results['batch_performance'][str(batch_size)] = {
                    'avg_latency_ms': avg_latency * 1000,
                    'throughput_classifications_per_second': throughput,
                    'successful_classifications': successful_classifications,
                    'memory_used_mb': memory_used
                }

                print(f"    Avg latency: {avg_latency*1000:.1f}ms")
                print(f"    Throughput: {throughput:.1f} classifications/sec")
                print(f"    Memory used: {memory_used:.1f}MB")

        # Overall latency statistics
        all_latencies = []
        for batch_data in results['batch_performance'].values():
            if 'avg_latency_ms' in batch_data:
                all_latencies.append(batch_data['avg_latency_ms'] / 1000)

        if all_latencies:
            results['latency_stats'] = {
                'mean_latency_ms': np.mean(all_latencies) * 1000,
                'median_latency_ms': np.median(all_latencies) * 1000,
                'p95_latency_ms': np.percentile(all_latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(all_latencies, 99) * 1000
            }

        # Error rate
        results['error_rate'] = total_errors / total_attempts if total_attempts > 0 else 0

        print(f"  Error rate: {results['error_rate']*100:.1f}%")
        print()

        return results

    def benchmark_model(
        self,
        model_name: str,
        tasks: List[str],
        test_data: Optional[Dict] = None
    ) -> Dict:
        """
        Benchmark a model across multiple tasks.

        Args:
            model_name: Model to benchmark
            tasks: List of tasks to benchmark
            test_data: Optional test data for each task

        Returns:
            Complete benchmark results for the model
        """
        print("=" * 60)
        print(f"BENCHMARKING MODEL: {model_name}")
        print("=" * 60)
        print()

        model_results = {
            'model_name': model_name,
            'tasks': {},
            'summary': {}
        }

        # Default test data
        default_prompts = [
            "আমার নাম",
            "Once upon a time",
            "বাংলাদেশ একটি সুন্দর দেশ",
            "The quick brown fox jumps"
        ]

        default_translations = [
            {'text': 'আমি স্কুলে যাচ্ছি।', 'src': 'bn', 'tgt': 'en'},
            {'text': 'I am going to school.', 'src': 'en', 'tgt': 'bn'},
            {'text': 'বই পড়া ভালো অভ্যাস।', 'src': 'bn', 'tgt': 'en'},
            {'text': 'Reading books is a good habit.', 'src': 'en', 'tgt': 'bn'}
        ]

        default_texts = [
            "আমি একটি ছোট গল্প বলব।",
            "This is a simple story for children.",
            "বাচ্চাদের জন্য শিক্ষামূলক বই।",
            "Educational content for young learners."
        ]

        # Run benchmarks for each task
        for task in tasks:
            try:
                if task == 'generation':
                    prompts = test_data.get('prompts', default_prompts) if test_data else default_prompts
                    task_results = self.benchmark_generation(model_name, prompts)
                
                elif task == 'translation':
                    translations = test_data.get('translations', default_translations) if test_data else default_translations
                    task_results = self.benchmark_translation(model_name, translations)
                
                elif task in ['readability', 'safety']:
                    texts = test_data.get('texts', default_texts) if test_data else default_texts
                    task_results = self.benchmark_classification(model_name, texts, task)
                
                else:
                    print(f"Unknown task: {task}")
                    continue

                model_results['tasks'][task] = task_results

            except Exception as e:
                print(f"Error benchmarking {task}: {e}")
                model_results['tasks'][task] = {'error': str(e)}

        # Calculate summary metrics
        self.calculate_summary_metrics(model_results)

        return model_results

    def calculate_summary_metrics(self, model_results: Dict):
        """Calculate summary metrics across all tasks."""
        summary = {
            'avg_latency_ms': 0.0,
            'total_throughput': 0.0,
            'avg_memory_usage_mb': 0.0,
            'overall_error_rate': 0.0,
            'tasks_completed': 0
        }

        latencies = []
        throughputs = []
        memory_usages = []
        error_rates = []

        for task_name, task_results in model_results['tasks'].items():
            if 'error' in task_results:
                continue

            summary['tasks_completed'] += 1

            # Collect metrics
            if 'latency_stats' in task_results and 'mean_latency_ms' in task_results['latency_stats']:
                latencies.append(task_results['latency_stats']['mean_latency_ms'])

            if 'error_rate' in task_results:
                error_rates.append(task_results['error_rate'])

            # Collect batch performance metrics
            for batch_size, batch_data in task_results.get('batch_performance', {}).items():
                if 'memory_used_mb' in batch_data:
                    memory_usages.append(batch_data['memory_used_mb'])

                # Collect throughput (different keys for different tasks)
                for key in batch_data:
                    if 'throughput' in key or 'per_second' in key:
                        throughputs.append(batch_data[key])

        # Calculate averages
        if latencies:
            summary['avg_latency_ms'] = np.mean(latencies)
        if throughputs:
            summary['total_throughput'] = np.sum(throughputs)
        if memory_usages:
            summary['avg_memory_usage_mb'] = np.mean(memory_usages)
        if error_rates:
            summary['overall_error_rate'] = np.mean(error_rates)

        model_results['summary'] = summary

    def run_benchmarks(
        self,
        models: List[str],
        tasks: List[str],
        test_data: Optional[Dict] = None
    ) -> Dict:
        """
        Run benchmarks for multiple models and tasks.

        Args:
            models: List of models to benchmark
            tasks: List of tasks to benchmark
            test_data: Optional test data

        Returns:
            Complete benchmark results
        """
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 58 + "║")
        print("║" + "  BILINGUAL MODEL BENCHMARKING SUITE".center(58) + "║")
        print("║" + " " * 58 + "║")
        print("╚" + "=" * 58 + "╝")
        print()

        print(f"Models to benchmark: {len(models)}")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"System: {self.results['benchmark_info']['system_info']}")
        print()

        # Benchmark each model
        for model_name in models:
            try:
                model_results = self.benchmark_model(model_name, tasks, test_data)
                self.results['models'][model_name] = model_results
            except Exception as e:
                print(f"Error benchmarking model {model_name}: {e}")
                self.results['models'][model_name] = {'error': str(e)}

        # Save results
        if self.output_file:
            self.save_results()

        # Print summary
        self.print_summary()

        return self.results

    def save_results(self):
        """Save benchmark results to file."""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Benchmark results saved to: {output_path}")

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for model_name, model_data in self.results['models'].items():
            if 'error' in model_data:
                print(f"\n{model_name}: ERROR - {model_data['error']}")
                continue

            summary = model_data.get('summary', {})
            print(f"\n{model_name}:")
            print(f"  Tasks completed: {summary.get('tasks_completed', 0)}")
            print(f"  Avg latency: {summary.get('avg_latency_ms', 0):.1f}ms")
            print(f"  Total throughput: {summary.get('total_throughput', 0):.1f} ops/sec")
            print(f"  Avg memory: {summary.get('avg_memory_usage_mb', 0):.1f}MB")
            print(f"  Error rate: {summary.get('overall_error_rate', 0)*100:.1f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Benchmark bilingual models for performance'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Models to benchmark'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=['generation'],
        choices=['generation', 'translation', 'readability', 'safety'],
        help='Tasks to benchmark'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for benchmark results (JSON)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='JSON file with test data for benchmarking'
    )

    args = parser.parse_args()

    # Load test data if provided
    test_data = None
    if args.test_data:
        try:
            with open(args.test_data, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load test data: {e}")

    # Initialize benchmarker
    benchmarker = ModelBenchmark(output_file=args.output)

    # Run benchmarks
    try:
        results = benchmarker.run_benchmarks(args.models, args.tasks, test_data)
        print("\nBenchmarking completed successfully!")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
