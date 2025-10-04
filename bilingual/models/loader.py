"""
Model loading utilities.

Handles loading of various model types from different sources.
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict
import warnings


def load_model_from_name(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Load a model by name.
    
    Args:
        model_name: Name or path of the model
        cache_dir: Directory to cache downloaded models
        device: Device to load model on ('cpu', 'cuda', 'mps')
        **kwargs: Additional model-specific arguments
        
    Returns:
        Loaded model instance
    """
    # Check if it's a local path
    if os.path.exists(model_name):
        return _load_local_model(model_name, device=device, **kwargs)
    
    # Check in package models directory
    package_dir = Path(__file__).parent.parent
    model_dir = package_dir / "models"
    local_model_path = model_dir / model_name
    
    if local_model_path.exists():
        return _load_local_model(str(local_model_path), device=device, **kwargs)
    
    # Try to load from Hugging Face Hub or other sources
    try:
        return _load_from_hub(model_name, cache_dir=cache_dir, device=device, **kwargs)
    except Exception as e:
        warnings.warn(f"Could not load model from hub: {e}")
    
    # Model not found - return placeholder
    warnings.warn(
        f"Model '{model_name}' not found. Returning placeholder model. "
        "Train and save a model first, or specify a valid model path."
    )
    return PlaceholderModel(model_name)


def _load_local_model(model_path: str, device: Optional[str] = None, **kwargs) -> Any:
    """Load a model from a local path."""
    # Determine model type from path/config
    # For now, return a placeholder
    return PlaceholderModel(model_path)


def _load_from_hub(model_name: str, cache_dir: Optional[str] = None, device: Optional[str] = None, **kwargs) -> Any:
    """Load a model from Hugging Face Hub or other remote source."""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **kwargs
        )
        
        if device:
            model = model.to(device)
        
        return model
    except ImportError:
        raise ImportError(
            "transformers is required to load models from Hugging Face Hub. "
            "Install it with: pip install transformers"
        )


class PlaceholderModel:
    """
    Placeholder model for development and testing.
    
    This is used when the actual model is not available yet.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.config = {"model_type": "placeholder"}
    
    def __call__(self, *args, **kwargs):
        warnings.warn(
            f"Using placeholder model '{self.name}'. "
            "This is for development only. Train a real model for production use."
        )
        return None
    
    def __repr__(self):
        return f"PlaceholderModel(name='{self.name}')"
