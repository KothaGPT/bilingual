"""
Model loading utilities.

Handles loading of various model types from different sources.
"""

import os
import random
import warnings
from pathlib import Path
from typing import Any, Optional


from typing import Any, Optional
from bilingual.models.manager import model_manager
from bilingual.exceptions import ModelLoadError

def load_model_from_name(
    model_name: str, cache_dir: Optional[str] = None, device: Optional[str] = None, **kwargs
) -> Any:
    """
    Load a model by name using the central ModelManager.

    Args:
        model_name: Name or path of the model
        cache_dir: Directory to cache downloaded models (passed to manager)
        device: Device to load model on ('cpu', 'cuda', 'mps')
        **kwargs: Additional model-specific arguments including 'model_type' and 'load_in_8bit'

    Returns:
        Loaded model instance from manager cache
    """
    try:
        # Map device if provided, else manager handles it
        if device:
            model_manager.device = device
            
        model_type = kwargs.get("model_type", "auto")
        load_in_8bit = kwargs.get("load_in_8bit", True) # Default to production-ready 8-bit

        return model_manager.load_model(
            model_name=model_name,
            model_type=model_type,
            load_in_8bit=load_in_8bit
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to load model {model_name} via manager.", {"error": str(e)})

# Deprecated: PlaceholderModel is now prohibited in production.
# Removed to enforce strictly managed model loading.
