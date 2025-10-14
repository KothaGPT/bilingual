#!/usr/bin/env python3
"""
Classification Model Training Script.

This script trains classification models for:
- Readability assessment (age-appropriate content)
- Safety classification (child-safe content)
- Language identification
- Domain classification

Usage:
    python scripts/train_classifier.py \
        --task readability \
        --data datasets/processed/ \
        --model bert-base-multilingual-cased \
        --output models/readability-classifier/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bilingual.data_utils import BilingualDataset  # noqa: E402


class ClassificationTrainer:
    """Train classification models for bilingual text."""

    def __init__(
        self,
        task: str,
        data_dir: str,
        base_model: str,
        output_dir: str,
        max_length: int = 512
    ):
        """
        Initialize trainer.

        Args:
            task: Classification task (readability, safety, language, domain)
            data_dir: Directory containing training data
            base_model: Base model to fine-tune
            output_dir: Output directory for trained model
            max_length: Maximum sequence length
        """
        self.task = task
        self.data_dir = Path(data_dir)
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.max_length = max_length

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Task-specific configurations
        self.task_configs = {
            'readability': {
                'labels': ['6-8', '9-10', '11-12', 'general'],
                'label_field': 'age_range',
                'description': 'Age-appropriate readability classification'
            },
            'safety': {
                'labels': ['safe', 'unsafe'],
                'label_field': 'safety_label',
                'description': 'Child-safety content classification'
            },
            'language': {
                'labels': ['bn', 'en', 'mixed'],
                'label_field': 'language',
                'description': 'Language identification'
            },
            'domain': {
                'labels': ['story', 'education', 'dialogue', 'description', 'instruction'],
                'label_field': 'domain',
                'description': 'Content domain classification'
            }
        }

        if task not in self.task_configs:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.task_configs.keys())}")

        self.config = self.task_configs[task]
        self.num_labels = len(self.config['labels'])
        self.label_to_id = {label: i for i, label in enumerate(self.config['labels'])}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

    def load_and_prepare_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load and prepare training data.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        print(f"Loading data for task: {self.task}")
        print(f"Expected labels: {self.config['labels']}")
        print()

        # Load datasets
        train_file = self.data_dir / "train.jsonl"
        val_file = self.data_dir / "val.jsonl"
        test_file = self.data_dir / "test.jsonl"

        datasets = {}
        for name, file_path in [("train", train_file), ("val", val_file), ("test", test_file)]:
            if file_path.exists():
                dataset = BilingualDataset(file_path=str(file_path))
                datasets[name] = list(dataset)
                print(f"Loaded {len(datasets[name])} {name} samples")
            else:
                print(f"Warning: {file_path} not found")
                datasets[name] = []

        # Prepare data for classification
        prepared_datasets = {}
        for name, samples in datasets.items():
            prepared = self.prepare_samples_for_task(samples)
            prepared_datasets[name] = prepared
            print(f"Prepared {len(prepared)} {name} samples for {self.task}")

        print()
        return prepared_datasets.get('train', []), prepared_datasets.get('val', []), prepared_datasets.get('test', [])

    def prepare_samples_for_task(self, samples: List[Dict]) -> List[Dict]:
        """
        Prepare samples for specific classification task.

        Args:
            samples: Raw samples

        Returns:
            Prepared samples with labels
        """
        prepared = []
        label_field = self.config['label_field']

        for sample in samples:
            text = sample.get('text', '')
            if not text:
                continue

            # Get label based on task
            if self.task == 'readability':
                label = self.prepare_readability_label(sample)
            elif self.task == 'safety':
                label = self.prepare_safety_label(sample)
            elif self.task == 'language':
                label = sample.get('language', 'mixed')
            elif self.task == 'domain':
                label = sample.get('domain', 'general')
            else:
                continue

            # Skip if label not in expected labels
            if label not in self.label_to_id:
                continue

            prepared.append({
                'text': text,
                'label': label,
                'label_id': self.label_to_id[label]
            })

        return prepared

    def prepare_readability_label(self, sample: Dict) -> str:
        """Prepare readability label from sample."""
        # Use existing age_range if available
        if 'age_range' in sample:
            return sample['age_range']

        # Generate readability label based on text features
        text = sample.get('text', '')
        words = text.split()
        
        if not words:
            return 'general'

        avg_word_length = sum(len(w) for w in words) / len(words)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = len(words) / max(sentence_count, 1)

        # Simple heuristic for age classification
        if avg_word_length < 4 and avg_sentence_length < 8:
            return '6-8'
        elif avg_word_length < 6 and avg_sentence_length < 12:
            return '9-10'
        elif avg_word_length < 8 and avg_sentence_length < 16:
            return '11-12'
        else:
            return 'general'

    def prepare_safety_label(self, sample: Dict) -> str:
        """Prepare safety label from sample."""
        # Use existing safety label if available
        if 'safety_label' in sample:
            return sample['safety_label']

        # Simple keyword-based safety classification
        text = sample.get('text', '').lower()
        
        unsafe_keywords = [
            'violence', 'weapon', 'scary', 'horror', 'death', 'kill',
            'সহিংসতা', 'অস্ত্র', 'ভয়ানক', 'মৃত্যু', 'হত্যা'
        ]

        for keyword in unsafe_keywords:
            if keyword in text:
                return 'unsafe'

        return 'safe'

    def create_synthetic_labels(self, samples: List[Dict]) -> List[Dict]:
        """
        Create synthetic labels for demonstration.
        In production, use real labeled data.

        Args:
            samples: Unlabeled samples

        Returns:
            Samples with synthetic labels
        """
        print("Creating synthetic labels for demonstration...")
        
        labeled_samples = []
        for sample in samples:
            text = sample.get('text', '')
            if not text:
                continue

            # Create synthetic label based on text characteristics
            if self.task == 'readability':
                label = self.prepare_readability_label(sample)
            elif self.task == 'safety':
                label = self.prepare_safety_label(sample)
            elif self.task == 'language':
                # Simple language detection based on script
                bn_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
                en_chars = sum(1 for c in text if c.isascii() and c.isalpha())
                
                if bn_chars > en_chars * 2:
                    label = 'bn'
                elif en_chars > bn_chars * 2:
                    label = 'en'
                else:
                    label = 'mixed'
            elif self.task == 'domain':
                # Simple domain classification based on keywords
                if any(word in text.lower() for word in ['story', 'once', 'গল্প']):
                    label = 'story'
                elif any(word in text.lower() for word in ['learn', 'teach', 'শিক্ষা']):
                    label = 'education'
                elif any(word in text.lower() for word in ['said', 'asked', 'বলল']):
                    label = 'dialogue'
                else:
                    label = 'description'
            else:
                continue

            labeled_samples.append({
                'text': text,
                'label': label,
                'label_id': self.label_to_id.get(label, 0)
            })

        return labeled_samples

    def train_model(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        """
        Train classification model.

        Args:
            train_data: Training samples
            val_data: Validation samples

        Returns:
            Path to trained model
        """
        print("=" * 60)
        print(f"TRAINING {self.task.upper()} CLASSIFIER")
        print("=" * 60)
        print()

        # Check if transformers is available
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                DataCollatorWithPadding,
                Trainer,
                TrainingArguments,
            )
            from datasets import Dataset
        except ImportError:
            print("Error: transformers and datasets libraries required")
            print("Install with: pip install transformers datasets")
            return self._create_placeholder_model()

        # Load tokenizer and model
        print(f"Loading base model: {self.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=self.num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )

        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model loaded with {self.num_labels} labels")
        print()

        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.max_length,
                padding=True
            )

        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        print("Starting training...")
        trainer.train()

        # Save model
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        # Save task configuration
        config_file = self.output_dir / "task_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'task': self.task,
                'labels': self.config['labels'],
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label,
                'description': self.config['description']
            }, f, indent=2)

        print(f"Model saved to: {self.output_dir}")
        return str(self.output_dir)

    def _create_placeholder_model(self) -> str:
        """Create placeholder model info when transformers not available."""
        print("Creating placeholder model configuration...")
        
        config_file = self.output_dir / "task_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'task': self.task,
                'labels': self.config['labels'],
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label,
                'description': self.config['description'],
                'status': 'placeholder'
            }, f, indent=2)

        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(f"""# {self.task.title()} Classifier

## Task: {self.config['description']}

### Labels
{chr(10).join(f'- {label}' for label in self.config['labels'])}

### Training
To train this model, install transformers and run:

```bash
pip install transformers datasets
python scripts/train_classifier.py --task {self.task} --data datasets/processed/
```

### Usage
```python
from bilingual import bilingual_api as bb

# Use the classifier
result = bb.{self.task}_check("Your text here")
print(result)
```
""")

        print(f"Placeholder configuration saved to: {self.output_dir}")
        return str(self.output_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Train classification models for bilingual text'
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['readability', 'safety', 'language', 'domain'],
        help='Classification task to train'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-multilingual-cased',
        help='Base model to fine-tune'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--synthetic-labels',
        action='store_true',
        help='Create synthetic labels for demonstration'
    )

    args = parser.parse_args()

    # Set default output directory
    if not args.output:
        args.output = f"models/{args.task}-classifier/"

    # Initialize trainer
    trainer = ClassificationTrainer(
        task=args.task,
        data_dir=args.data,
        base_model=args.model,
        output_dir=args.output,
        max_length=args.max_length
    )

    try:
        # Load data
        train_data, val_data, test_data = trainer.load_and_prepare_data()

        # Create synthetic labels if requested or no labeled data
        if args.synthetic_labels or not train_data:
            print("Using synthetic labels...")
            # Load raw data and create synthetic labels
            raw_train = BilingualDataset(file_path=str(Path(args.data) / "train.jsonl"))
            train_data = trainer.create_synthetic_labels(list(raw_train)[:1000])  # Limit for demo
            
            raw_val = BilingualDataset(file_path=str(Path(args.data) / "val.jsonl"))
            val_data = trainer.create_synthetic_labels(list(raw_val)[:200])

        if not train_data:
            print("Error: No training data available")
            sys.exit(1)

        print(f"Training with {len(train_data)} samples")
        print(f"Validation with {len(val_data)} samples")
        print()

        # Train model
        model_path = trainer.train_model(train_data, val_data)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Task: {args.task}")
        print(f"Model saved to: {model_path}")
        print(f"Labels: {trainer.config['labels']}")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
