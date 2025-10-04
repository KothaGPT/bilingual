"""
Bilingual: High-quality Bangla and English NLP toolkit.

This package provides production-ready tools for Bangla and English
natural language processing, including tokenization, normalization,
translation, and generation.
"""

__version__ = "0.1.0"
__author__ = "Bilingual Project Contributors"
__license__ = "Apache-2.0"

from bilingual import api as bilingual_api
from bilingual.normalize import normalize_text
from bilingual.tokenizer import BilingualTokenizer, load_tokenizer

__all__ = [
    "bilingual_api",
    "normalize_text",
    "load_tokenizer",
    "BilingualTokenizer",
    "__version__",
]
