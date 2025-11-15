#!/usr/bin/env python3
"""
Create Gradio demo app for Hugging Face Spaces.

Usage:
    python scripts/huggingface/create_demo.py --repo your-username/bn-wikipedia-lm --output spaces/bn-wikipedia-demo
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


APP_PY_TEMPLATE = '''import gradio as gr
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch

# Load model
MODEL_ID = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)

# Create pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1)


def predict_mask(text, top_k=5):
    """Fill masked text."""
    if "[MASK]" not in text:
        return "Please include [MASK] token in your text"
    
    try:
        results = fill_mask(text, top_k=top_k)
        
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"{{i}}. {{result['sequence']}} (score: {{result['score']:.4f}})")
        
        return "\\n".join(output)
    
    except Exception as e:
        return f"Error: {{str(e)}}"


def get_embeddings(text):
    """Get sentence embeddings."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]
        
        # Mean pooling
        sentence_embedding = embeddings.mean(dim=1)
        
        return f"Embedding shape: {{sentence_embedding.shape}}\\nFirst 10 values: {{sentence_embedding[0, :10].tolist()}}"
    
    except Exception as e:
        return f"Error: {{str(e)}}"


def compute_similarity(text1, text2):
    """Compute similarity between two texts."""
    try:
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=512)
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs1 = model(**inputs1, output_hidden_states=True)
            outputs2 = model(**inputs2, output_hidden_states=True)
            
            emb1 = outputs1.hidden_states[-1].mean(dim=1)
            emb2 = outputs2.hidden_states[-1].mean(dim=1)
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        
        return f"Similarity: {{similarity:.4f}}"
    
    except Exception as e:
        return f"Error: {{str(e)}}"


# Create Gradio interface
with gr.Blocks(title="Bangla Language Model Demo") as demo:
    gr.Markdown("# 🇧🇩 Bangla Language Model Demo")
    gr.Markdown(f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})")
    
    with gr.Tab("Fill Mask"):
        gr.Markdown("### Fill Masked Text")
        gr.Markdown("Enter text with [MASK] token to predict the masked word.")
        
        with gr.Row():
            with gr.Column():
                mask_input = gr.Textbox(
                    label="Input Text",
                    placeholder="বাংলাদেশের রাজধানী [MASK]",
                    value="বাংলাদেশের রাজধানী [MASK]"
                )
                top_k = gr.Slider(1, 10, value=5, step=1, label="Top K Predictions")
                mask_button = gr.Button("Predict")
            
            with gr.Column():
                mask_output = gr.Textbox(label="Predictions", lines=10)
        
        mask_button.click(predict_mask, inputs=[mask_input, top_k], outputs=mask_output)
        
        gr.Examples(
            examples=[
                ["বাংলাদেশের রাজধানী [MASK]", 5],
                ["আমি [MASK] খাই", 5],
                ["সূর্য [MASK] উঠে", 5],
                ["কবিতার [MASK] সুন্দর", 5],
            ],
            inputs=[mask_input, top_k],
        )
    
    with gr.Tab("Embeddings"):
        gr.Markdown("### Get Text Embeddings")
        gr.Markdown("Get contextualized embeddings for Bangla text.")
        
        with gr.Row():
            with gr.Column():
                emb_input = gr.Textbox(
                    label="Input Text",
                    placeholder="আমি বাংলায় কথা বলি",
                    value="আমি বাংলায় কথা বলি"
                )
                emb_button = gr.Button("Get Embeddings")
            
            with gr.Column():
                emb_output = gr.Textbox(label="Embeddings", lines=10)
        
        emb_button.click(get_embeddings, inputs=emb_input, outputs=emb_output)
    
    with gr.Tab("Similarity"):
        gr.Markdown("### Compute Semantic Similarity")
        gr.Markdown("Compute cosine similarity between two Bangla texts.")
        
        with gr.Row():
            with gr.Column():
                sim_input1 = gr.Textbox(
                    label="Text 1",
                    placeholder="আমি ভাত খাই",
                    value="আমি ভাত খাই"
                )
                sim_input2 = gr.Textbox(
                    label="Text 2",
                    placeholder="আমি খাবার খাই",
                    value="আমি খাবার খাই"
                )
                sim_button = gr.Button("Compute Similarity")
            
            with gr.Column():
                sim_output = gr.Textbox(label="Similarity Score", lines=5)
        
        sim_button.click(compute_similarity, inputs=[sim_input1, sim_input2], outputs=sim_output)
        
        gr.Examples(
            examples=[
                ["আমি ভাত খাই", "আমি খাবার খাই"],
                ["বাংলাদেশ সুন্দর", "বাংলাদেশ একটি দেশ"],
                ["কবিতা লিখি", "গান গাই"],
            ],
            inputs=[sim_input1, sim_input2],
        )
    
    gr.Markdown("---")
    gr.Markdown("### About")
    gr.Markdown("""
    This demo showcases a Bangla language model trained on Wikipedia data.
    
    **Features:**
    - Fill-mask: Predict masked words in Bangla text
    - Embeddings: Get contextualized embeddings
    - Similarity: Compute semantic similarity
    
    **Model:** [{MODEL_ID}](https://huggingface.co/{MODEL_ID})
    
    **GitHub:** [KothaGPT/bilingual](https://github.com/KothaGPT/bilingual)
    """)

if __name__ == "__main__":
    demo.launch()
'''

REQUIREMENTS_TXT = """gradio>=4.0.0
transformers>=4.30.0
torch>=2.0.0
"""

README_MD = """---
title: Bangla Language Model Demo
emoji: 🇧🇩
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Bangla Language Model Demo

Interactive demo for the Bangla language model trained on Wikipedia.

## Features

- **Fill Mask:** Predict masked words in Bangla text
- **Embeddings:** Get contextualized embeddings for Bangla text
- **Similarity:** Compute semantic similarity between texts

## Model

Model: [{repo_id}](https://huggingface.co/{repo_id})

## Usage

Try the demo above or use the model directly:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForMaskedLM.from_pretrained("{repo_id}")
```

## Links

- **Model:** https://huggingface.co/{repo_id}
- **GitHub:** https://github.com/KothaGPT/bilingual
"""


def create_demo(repo_id: str, output_dir: Path) -> bool:
    """Create Gradio demo app."""
    logger.info(f"Creating demo for {repo_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create app.py
    app_py = APP_PY_TEMPLATE.format(repo_id=repo_id)
    with open(output_dir / "app.py", "w", encoding="utf-8") as f:
        f.write(app_py)
    logger.info("✓ Created app.py")

    # Create requirements.txt
    with open(output_dir / "requirements.txt", "w") as f:
        f.write(REQUIREMENTS_TXT)
    logger.info("✓ Created requirements.txt")

    # Create README.md
    readme = README_MD.format(repo_id=repo_id)
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    logger.info("✓ Created README.md")

    logger.info(f"✓ Demo created at {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create Gradio demo for Hugging Face Spaces")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face model repository ID")
    parser.add_argument(
        "--output", type=str, default="spaces/demo", help="Output directory for demo files"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    success = create_demo(args.repo, output_dir)

    if not success:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Gradio Demo Created!")
    print("=" * 60)
    print(f"Location: {output_dir}")
    print("\nFiles created:")
    print("  - app.py (Gradio app)")
    print("  - requirements.txt (Dependencies)")
    print("  - README.md (Space description)")
    print("\nNext steps:")
    print("1. Test locally:")
    print(f"   cd {output_dir}")
    print("   pip install -r requirements.txt")
    print("   python app.py")
    print("\n2. Create Hugging Face Space:")
    print("   - Go to https://huggingface.co/spaces")
    print("   - Click 'Create new Space'")
    print("   - Choose Gradio SDK")
    print(f"   - Upload files from {output_dir}")
    print("\n3. Or use git:")
    print("   git clone https://huggingface.co/spaces/your-username/space-name")
    print(f"   cp {output_dir}/* space-name/")
    print("   cd space-name && git add . && git commit -m 'Add demo' && git push")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
