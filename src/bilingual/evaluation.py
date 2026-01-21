"""
Evaluation metrics and utilities for the Bilingual NLP Toolkit.

Provides standardized metrics for translation (BLEU, METEOR, chrF),
generation (ROUGE, Diversity), and classification.
"""

import math
import re
import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

from bilingual.exceptions import EvaluationError

logger = logging.getLogger(__name__)

# Try internal imports
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Standardizing on fallback tokenization.")

class BilingualEvaluator:
    """
    Consolidated evaluator for bilingual NLP tasks.
    Replaces fragmented and duplicated evaluation logic.
    """
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method4 if NLTK_AVAILABLE else None
        self._ensure_resources()

    def _ensure_resources(self):
        """Prepare NLTK resources silently."""
        if NLTK_AVAILABLE:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

    def tokenize(self, text: str) -> List[str]:
        """Unified tokenization for all metrics."""
        if not text: return []
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception:
                pass
        # Fallback regex tokenization
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    def compute_bleu(self, reference: str, candidate: str) -> float:
        """Compute BLEU score (0-1)."""
        ref_tokens = [self.tokenize(reference)]
        cand_tokens = self.tokenize(candidate)
        
        if not ref_tokens[0] or not cand_tokens:
            return 0.0
            
        try:
            if NLTK_AVAILABLE:
                return float(sentence_bleu(ref_tokens, cand_tokens, smoothing_function=self.smoothing))
            return self._fallback_bleu(ref_tokens[0], cand_tokens)
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")
            return 0.0

    def _fallback_bleu(self, ref_tokens: List[str], cand_tokens: List[str]) -> float:
        """Simplified BLEU for environments without NLTK."""
        overlap = len(set(ref_tokens) & set(cand_tokens))
        precision = overlap / len(cand_tokens) if cand_tokens else 0
        
        # Brevity penalty
        bp = math.exp(1 - len(ref_tokens)/len(cand_tokens)) if len(cand_tokens) < len(ref_tokens) else 1.0
        return bp * precision

    def compute_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute basic ROUGE-like overlap metrics."""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return {"rouge_1": 0.0}
            
        overlap = len(set(ref_tokens) & set(cand_tokens))
        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"rouge_1": f1}

    def evaluate_batch(self, references: List[str], candidates: List[str], task: str = "translation") -> Dict[str, Any]:
        """Perform batch evaluation for a task."""
        if len(references) != len(candidates):
            raise EvaluationError("Batch mismatch: references and candidates must have same length.")
            
        results = {"bleu": [], "rouge": []}
        for ref, cand in zip(references, candidates):
            results["bleu"].append(self.compute_bleu(ref, cand))
            results["rouge"].append(self.compute_rouge(ref, cand)["rouge_1"])
            
        count = len(references)
        summary = {
            "avg_bleu": sum(results["bleu"]) / count if count > 0 else 0,
            "avg_rouge": sum(results["rouge"]) / count if count > 0 else 0,
            "sample_count": count
        }
        return summary

# Global instance
_evaluator = BilingualEvaluator()

def evaluate_translation(references: List[str], candidates: List[str]) -> Dict[str, Any]:
    return _evaluator.evaluate_batch(references, candidates, task="translation")

def evaluate_generation(references: List[str], candidates: List[str]) -> Dict[str, Any]:
    return _evaluator.evaluate_batch(references, candidates, task="generation")
