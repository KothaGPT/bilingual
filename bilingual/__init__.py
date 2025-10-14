"""
Bilingual: High-quality Bangla and English NLP toolkit.

This package provides production-ready tools for Bangla and English
natural language processing, including tokenization, normalization,
translation, and generation.
"""

__version__ = "0.1.0"
__author__ = "Bilingual Project Contributors"
__license__ = "Apache-2.0"

from bilingual.testing import run_unit_tests, run_integration_tests, run_performance_benchmarks, generate_test_report

__all__ = [
    "bilingual_api",
    "normalize_text",
    "load_tokenizer",
    "BilingualTokenizer",
    "detect_language",
    "is_bengali",
    "is_english",
    "augment_text",
    "process_mixed_text",
    "detect_language_segments",
    "split_mixed_text",
    "evaluate_translation",
    "evaluate_generation",
    "bleu_score",
    "rouge_score",
    "load_model",
    "generate_text",
    "translate_text",
    "summarize_text",
    "zero_shot_classify",
    "multilingual_generate",
    "convert_to_onnx",
    "create_onnx_session",
    "benchmark_onnx_model",
    "optimize_onnx_model",
    "submit_evaluation",
    "calculate_content_safety_score",
    "generate_evaluation_report",
    "create_evaluation_interface",
    "get_settings",
    "run_unit_tests",
    "run_integration_tests",
    "run_performance_benchmarks",
    "generate_test_report",
    "__version__",
]
