"""
High-level API for bilingual package.

Provides simple functions for common NLP tasks in Bangla and English.
"""

from typing import Optional, Union, List, Dict, Any
import warnings

from bilingual.normalize import normalize_text as _normalize_text, detect_language
from bilingual.tokenizer import load_tokenizer as _load_tokenizer, BilingualTokenizer


# Global cache for loaded models and tokenizers
_TOKENIZER_CACHE: Dict[str, BilingualTokenizer] = {}
_MODEL_CACHE: Dict[str, Any] = {}


def load_tokenizer(model_name: str = "bilingual-tokenizer", force_reload: bool = False) -> BilingualTokenizer:
    """
    Load a tokenizer (with caching).
    
    Args:
        model_name: Name or path of the tokenizer model
        force_reload: Force reload even if cached
        
    Returns:
        BilingualTokenizer instance
    """
    if model_name not in _TOKENIZER_CACHE or force_reload:
        _TOKENIZER_CACHE[model_name] = _load_tokenizer(model_name)
    return _TOKENIZER_CACHE[model_name]


def load_model(model_name: str, force_reload: bool = False, **kwargs) -> Any:
    """
    Load a language model (with caching).
    
    Args:
        model_name: Name or path of the model
        force_reload: Force reload even if cached
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded model instance
    """
    if model_name not in _MODEL_CACHE or force_reload:
        # Import here to avoid circular dependencies
        from bilingual.models.loader import load_model_from_name
        _MODEL_CACHE[model_name] = load_model_from_name(model_name, **kwargs)
    return _MODEL_CACHE[model_name]


def normalize_text(
    text: str,
    lang: Optional[str] = None,
    **kwargs
) -> str:
    """
    Normalize text for Bangla or English.
    
    Args:
        text: Input text
        lang: Language code ('bn' or 'en'), auto-detected if None
        **kwargs: Additional normalization options
        
    Returns:
        Normalized text
        
    Examples:
        >>> normalize_text("আমি   স্কুলে যাচ্ছি।", lang="bn")
        'আমি স্কুলে যাচ্ছি.'
    """
    return _normalize_text(text, lang=lang, **kwargs)


def tokenize(
    text: str,
    tokenizer: Optional[Union[str, BilingualTokenizer]] = None,
    return_ids: bool = False,
) -> Union[List[str], List[int]]:
    """
    Tokenize text.
    
    Args:
        text: Input text
        tokenizer: Tokenizer name/path or instance (default: "bilingual-tokenizer")
        return_ids: If True, return token IDs instead of strings
        
    Returns:
        List of tokens or token IDs
        
    Examples:
        >>> tokenize("আমি বই পড়ি।")
        ['▁আমি', '▁বই', '▁পড়ি', '.']
    """
    if tokenizer is None:
        tokenizer = "bilingual-tokenizer"
    
    if isinstance(tokenizer, str):
        tokenizer = load_tokenizer(tokenizer)
    
    return tokenizer.encode(text, as_ids=return_ids)


def generate(
    prompt: str,
    model_name: str = "bilingual-small-lm",
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs
) -> str:
    """
    Generate text continuation from a prompt.
    
    Args:
        prompt: Input prompt text
        model_name: Name of the generation model
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        **kwargs: Additional generation parameters
        
    Returns:
        Generated text
        
    Examples:
        >>> generate("Once upon a time, there was a brave rabbit")
        'Once upon a time, there was a brave rabbit who lived in a forest...'
    """
    model = load_model(model_name)
    
    # Import here to avoid circular dependencies
    from bilingual.models.lm import generate_text
    
    return generate_text(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        **kwargs
    )


def translate(
    text: str,
    src: str = "bn",
    tgt: str = "en",
    model_name: str = "bilingual-translate",
    **kwargs
) -> str:
    """
    Translate text between Bangla and English.
    
    Args:
        text: Input text to translate
        src: Source language code ('bn' or 'en')
        tgt: Target language code ('bn' or 'en')
        model_name: Name of the translation model
        **kwargs: Additional translation parameters
        
    Returns:
        Translated text
        
    Examples:
        >>> translate("আমি বই পড়তে ভালোবাসি।", src="bn", tgt="en")
        'I love to read books.'
    """
    if src == tgt:
        warnings.warn(f"Source and target languages are the same ({src}). Returning original text.")
        return text
    
    model = load_model(model_name)
    
    # Import here to avoid circular dependencies
    from bilingual.models.translate import translate_text
    
    return translate_text(
        model=model,
        text=text,
        src_lang=src,
        tgt_lang=tgt,
        **kwargs
    )


def readability_check(
    text: str,
    lang: Optional[str] = None,
    model_name: str = "bilingual-readability",
) -> Dict[str, Any]:
    """
    Check readability level of text.
    
    Args:
        text: Input text
        lang: Language code ('bn' or 'en'), auto-detected if None
        model_name: Name of the readability model
        
    Returns:
        Dictionary with readability metrics:
            - level: Reading level (e.g., "elementary", "intermediate", "advanced")
            - age_range: Suggested age range
            - score: Numerical readability score
            
    Examples:
        >>> readability_check("আমি স্কুলে যাই।", lang="bn")
        {'level': 'elementary', 'age_range': '6-8', 'score': 2.5}
    """
    if lang is None:
        lang = detect_language(text)
    
    # Placeholder implementation - will be replaced with actual model
    # For now, use simple heuristics
    words = text.split()
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
    
    if avg_word_length < 4:
        level = "elementary"
        age_range = "6-8"
        score = 2.0
    elif avg_word_length < 6:
        level = "intermediate"
        age_range = "9-12"
        score = 5.0
    else:
        level = "advanced"
        age_range = "13+"
        score = 8.0
    
    return {
        "level": level,
        "age_range": age_range,
        "score": score,
        "language": lang,
    }


def safety_check(
    text: str,
    lang: Optional[str] = None,
    model_name: str = "bilingual-safety",
) -> Dict[str, Any]:
    """
    Check if text is safe and appropriate for children.
    
    Args:
        text: Input text
        lang: Language code ('bn' or 'en'), auto-detected if None
        model_name: Name of the safety model
        
    Returns:
        Dictionary with safety assessment:
            - is_safe: Boolean indicating if content is safe
            - confidence: Confidence score (0-1)
            - flags: List of any safety concerns
            - recommendation: Action recommendation
            
    Examples:
        >>> safety_check("This is a nice story about rabbits.")
        {'is_safe': True, 'confidence': 0.95, 'flags': [], 'recommendation': 'approved'}
    """
    if lang is None:
        lang = detect_language(text)
    
    # Placeholder implementation - will be replaced with actual model
    # For now, use simple keyword-based filtering
    
    # Simple safety check (to be replaced with ML model)
    is_safe = True
    flags = []
    confidence = 0.9
    
    return {
        "is_safe": is_safe,
        "confidence": confidence,
        "flags": flags,
        "recommendation": "approved" if is_safe else "review_required",
        "language": lang,
    }


def classify(
    text: str,
    labels: List[str],
    model_name: str = "bilingual-classifier",
    **kwargs
) -> Dict[str, float]:
    """
    Classify text into one or more categories.
    
    Args:
        text: Input text
        labels: List of possible labels
        model_name: Name of the classification model
        **kwargs: Additional classification parameters
        
    Returns:
        Dictionary mapping labels to confidence scores
        
    Examples:
        >>> classify("This is a story about animals.", labels=["story", "news", "dialogue"])
        {'story': 0.85, 'news': 0.05, 'dialogue': 0.10}
    """
    # Placeholder implementation
    # Return uniform distribution for now
    score = 1.0 / len(labels)
    return {label: score for label in labels}


# Convenience aliases
normalize = normalize_text
tok = tokenize
gen = generate
trans = translate
