"""
Central Model Manager for the Bilingual NLP Toolkit.

Provides a singleton interface for loading, caching, and managing 
model lifecycles to prevent OOM and redundant loading.
"""

import logging
import torch
from typing import Dict, Any, Optional, Union
from threading import Lock
from pathlib import Path

from bilingual.exceptions import ModelLoadError, ConfigurationError

logger = logging.getLogger(__name__)

from bilingual.models.registry.registry import model_registry

class ModelManager:
    """
    Singleton Manager for LLM Models and Tokenizers.
    Integrated with ModelRegistry for versioning support.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        """Initialize the manager internal state."""
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelManager initialized on device: {self.device}")

    def load_model(
        self, 
        model_name: str, 
        version: Optional[str] = None,
        model_type: Optional[str] = None, 
        load_in_8bit: bool = True,
        force_reload: bool = False
    ) -> Any:
        """
        Load a model into memory with version support.
        
        Args:
            model_name: Registry name or Path/HF ID
            version: Model version (optional if using Registry)
            model_type: override model type if needed
            load_in_8bit: Whether to use 8-bit quantization
            force_reload: Whether to bypass cache
        """
        # Resolve via Registry if possible
        try:
            entry = model_registry.resolve(model_name, version)
            actual_path = entry.path
            actual_type = model_type or entry.model_type
            actual_version = entry.version
        except Exception:
            # Fallback for direct loading by path/HF ID
            actual_path = model_name
            actual_type = model_type or "auto"
            actual_version = "raw"

        model_key = f"{model_name}_{actual_version}_{actual_type}_{'8bit' if load_in_8bit else 'fp32'}"
        
        with self._lock:
            if model_key in self._models and not force_reload:
                return self._models[model_key]

            logger.info(f"Loading model {actual_path} ({actual_version})...")
            
            try:
                from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
                
                quant_config = None
                if load_in_8bit and self.device == "cuda":
                    from transformers import BitsAndBytesConfig
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)

                if actual_type == "causal":
                    model = AutoModelForCausalLM.from_pretrained(
                        actual_path, quantization_config=quant_config, 
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                elif actual_type == "masked":
                    model = AutoModelForMaskedLM.from_pretrained(
                        actual_path, quantization_config=quant_config,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )
                else:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(
                        actual_path, quantization_config=quant_config,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True
                    )

                if self.device == "cpu":
                    model.to("cpu")

                self._models[model_key] = model
                return model

            except Exception as e:
                logger.error(f"Failed to load model {actual_path}: {e}")
                raise ModelLoadError(f"Could not load model {actual_path}")

    def get_tokenizer(self, model_name: str, force_reload: bool = False) -> Any:
        """Load and cache tokenizer."""
        with self._lock:
            if model_name in self._tokenizers and not force_reload:
                return self._tokenizers[model_name]

            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._tokenizers[model_name] = tokenizer
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {model_name}: {str(e)}")
                raise ModelLoadError(f"Tokenizer fail for {model_name}")

    def warmup(self, model_list: list):
        """Preload specified models to avoid latency on first request."""
        for m in model_list:
            try:
                self.load_model(m)
                self.get_tokenizer(m)
            except Exception as e:
                logger.error(f"Warmup failed for {m}: {e}")

    def clear_cache(self, model_name: Optional[str] = None):
        """Free memory by clearing cached models."""
        with self._lock:
            if model_name:
                keys_to_del = [k for k in self._models if k.startswith(model_name)]
                for k in keys_to_del:
                    del self._models[k]
                    logger.info(f"Evicted {k} from cache.")
            else:
                self._models.clear()
                self._tokenizers.clear()
                logger.info("Cleared all cached models and tokenizers.")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Global access point
model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    return model_manager
