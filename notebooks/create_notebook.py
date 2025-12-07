
import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "view-in-github",
    "colab_type": "text"
   },
   "source": [
    "<a href=\"https://colab.research.google.com/github/khulnasoft-bot/bilingual/blob/main/notebooks/Auto_Training.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Bilingual Model Auto-Training\n",
    "\n",
    "This notebook provides an automated pipeline for training the Bilingual English-Bangla model using Google Colab.\n",
    "\n",
    "**Steps:**\n",
    "1. Setup Environment\n",
    "2. Collect/Prepare Data\n",
    "3. Process Data\n",
    "4. Train Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone repository (if not already present)\n",
    "import os\n",
    "if not os.path.exists(\"bilingual-1\"):\n",
    "    !git clone https://github.com/khulnasoft-bot/bilingual.git bilingual-1\n",
    "    %cd bilingual-1\n",
    "else:\n",
    "    %cd bilingual-1\n",
    "    !git pull"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
    "!pip install -r requirements.txt\n",
    "!pip install pandas pyarrow sentencepiece"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install local package\n",
    "!pip install -e ."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Collect Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Collect sample data (replace with 'wikipedia' or 'web-scrape' for more data)\n",
    "!python scripts/collect_data.py --source sample --output data/raw"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Process Data\n",
    "Convert the JSONL data to Parquet format required by the training script."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "# Output directory for processed data\n",
    "processed_dir = Path(\"data/processed/colab\")\n",
    "processed_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "def convert_jsonl_to_parquet(input_file, output_file):\n",
    "    data = []\n",
    "    with open(input_file, 'r', encoding='utf-8') as f:\n",
    "        for line in f:\n",
    "            if line.strip():\n",
    "                record = json.loads(line)\n",
    "                # Map keys to expected format if needed, or assume data is clean\n",
    "                # Trainer expects: en_ids, bn_ids (after tokenization) OR raw text to be tokenized\n",
    "                # Wait, scripts/train.py expects 'en_ids' and 'bn_ids' in the parquet!\n",
    "                # We need to Tokenize first!\n",
    "                data.append(record)\n",
    "    \n",
    "    if not data:\n",
    "        print(f\"No data found in {input_file}\")\n",
    "        return\n",
    "\n",
    "    df = pd.DataFrame(data)\n",
    "    # Provide text columns\n",
    "    if 'en' in df.columns: df['en_text'] = df['en']\n",
    "    if 'bn' in df.columns: df['bn_text'] = df['bn']\n",
    "\n",
    "    # We need to tokenize here because train.py expects IDs\n",
    "    # Let's use the BilingualTokenizer\n",
    "    from bilingual.tokenizer import BilingualTokenizer, train_tokenizer\n",
    "    \n",
    "    # Check for tokenizer model\n",
    "    model_path = \"bilingual_tokenizer.model\"\n",
    "    if not os.path.exists(model_path):\n",
    "        print(\"Training Tokenizer...\")\n",
    "        # Save temp files for training\n",
    "        with open(\"temp_train.txt\", \"w\", encoding=\"utf-8\") as f:\n",
    "            for item in data:\n",
    "                 f.write(item.get('en', '') + '\\n')\n",
    "                 f.write(item.get('bn', '') + '\\n')\n",
    "        train_tokenizer([\"temp_train.txt\"], \"bilingual_tokenizer\", vocab_size=1000)\n",
    "    \n",
    "    tokenizer = BilingualTokenizer(model_path)\n",
    "    \n",
    "    print(\"Tokenizing...\")\n",
    "    df['en_ids'] = df['en'].apply(lambda x: tokenizer.encode(x, as_ids=True))\n",
    "    df['bn_ids'] = df['bn'].apply(lambda x: tokenizer.encode(x, as_ids=True))\n",
    "    \n",
    "    # Split into train/val\n",
    "    train_df = df.sample(frac=0.8, random_state=42)\n",
    "    val_df = df.drop(train_df.index)\n",
    "    \n",
    "    train_df.to_parquet(processed_dir / \"train.parquet\")\n",
    "    val_df.to_parquet(processed_dir / \"val.parquet\")\n",
    "    print(f\"Saved processing buffers to {processed_dir}\")\n",
    "\n",
    "\n",
    "convert_jsonl_to_parquet(\"data/raw/parallel_corpus.jsonl\", processed_dir)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Train Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a temporary training config for Colab\n",
    "import yaml\n",
    "\n",
    "config = {\n",
    "    \"model\": {\n",
    "        \"src_vocab_size\": 1000, # Matching our small sample tokenizer\n",
    "        \"tgt_vocab_size\": 1000,\n",
    "        \"d_model\": 128,\n",
    "        \"nhead\": 4,\n",
    "        \"num_encoder_layers\": 2,\n",
    "        \"num_decoder_layers\": 2,\n",
    "        \"dim_feedforward\": 512,\n",
    "        \"dropout\": 0.1,\n",
    "        \"max_seq_length\": 64\n",
    "    },\n",
    "    \"data\": {\n",
    "        \"train_data\": \"data/processed/colab/train.parquet\",\n",
    "        \"val_data\": \"data/processed/colab/val.parquet\"\n",
    "    },\n",
    "    \"training\": {\n",
    "        \"batch_size\": 8,\n",
    "        \"num_epochs\": 5,\n",
    "        \"learning_rate\": 0.001,\n",
    "        \"output_dir\": \"models/colab_model\",\n",
    "        \"log_interval\": 1,\n",
    "        \"num_workers\": 0 # often better for colab interactions\n",
    "    }\n",
    "}\n",
    "\n",
    "with open(\"config/colab_train.yaml\", \"w\") as f:\n",
    "    yaml.dump(config, f)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run Training\n",
    "!python scripts/train.py --config config/colab_train.yaml"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("notebooks/Auto_Training.ipynb", "w") as f:
    json.dump(notebook_content, f, indent=2)

print("Notebook created successfully.")
