"""Tests for tokenizer utilities."""

import pytest
from bilingual.tokenizer import BilingualTokenizer


class TestBilingualTokenizer:
    def test_tokenizer_initialization(self):
        tokenizer = BilingualTokenizer()
        assert tokenizer.sp is None
        assert tokenizer.model_path is None
    
    def test_tokenizer_without_model_raises_error(self):
        tokenizer = BilingualTokenizer()
        
        with pytest.raises(RuntimeError):
            tokenizer.encode("test")
        
        with pytest.raises(RuntimeError):
            tokenizer.decode([1, 2, 3])
        
        with pytest.raises(RuntimeError):
            tokenizer.get_vocab_size()
    
    def test_load_nonexistent_model_raises_error(self):
        tokenizer = BilingualTokenizer()
        
        with pytest.raises(FileNotFoundError):
            tokenizer.load("nonexistent_model.model")
    
    # Note: Actual tokenization tests require a trained model
    # These would be integration tests run after training
