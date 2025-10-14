#!/usr/bin/env python3
"""
Script to download and set up a small bilingual language model.
"""

from pathlib import Path


def download_small_model():
    """Download a small multilingual model for bilingual use."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Use a small multilingual model that supports both Bangla and English
        model_name = "microsoft/DialoGPT-small"  # Small conversational model

        print(f"Downloading model: {model_name}")

        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Save to our models directory
        model_dir = Path("/Users/dev/bilingual/models")
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "bilingual-small-lm"
        model_path.mkdir(exist_ok=True)

        print(f"Saving model to: {model_path}")

        # Save the model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # Also save as a general model for other tasks
        general_model_path = model_dir / "bilingual-general"
        general_model_path.mkdir(exist_ok=True)

        model.save_pretrained(general_model_path)
        tokenizer.save_pretrained(general_model_path)

        print("✅ Model downloaded and saved successfully")
        print(f"Model location: {model_path}")

        # Test the model
        print("\n🧪 Testing model...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test generation: {generated}")

        return True

    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


if __name__ == "__main__":
    success = download_small_model()
    if success:
        print("\n🎉 Model setup completed!")
    else:
        print("\n💥 Model setup failed!")
