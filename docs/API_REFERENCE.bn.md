---
original: /docs/API_REFERENCE.md
translated: 2025-10-24
---

> **বিঃদ্রঃ** এটি একটি স্বয়ংক্রিয়ভাবে অনুবাদকৃত নথি। মূল ইংরেজি সংস্করণের জন্য [এখানে ক্লিক করুন](/{rel_path}) করুন।

---

# Bilingual# Bilingual API Reference

## Table of Contents
- [Core Modules](#core-modules)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Utilities](#utilities)
- [Examples](#examples)
- [Contributing](#contributing)

## Core Modules

### `bilingual.config`
Configuration management for the Bilingual project.

#### `ModelConfig`
```python
class ModelConfig:
    """Configuration for model architecture and training.
    
    Args:
        vocab_size (int): Size of the vocabulary
        hidden_size (int): Dimensionality of the model
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
        max_length (int): Maximum sequence length (default: 512)
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_length = max_length
```

### `bilingual.models`
Base model implementations and architectures.

#### `BilingualLM`
```python
class BilingualLM(nn.Module):
    """Bilingual Language Model for Bengali and English.
    
    Args:
        config (ModelConfig): Model configuration
        tokenizer (Tokenizer): Tokenizer instance
    """
    def __init__(self, config: ModelConfig, tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        # Model implementation...
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            labels (torch.Tensor, optional): Target token IDs
            
        Returns:
            dict: Model outputs including loss and logits
        """
        # Implementation...
```

## Data Processing

### `bilingual.data_utils`

#### `load_dataset`
```python
def load_dataset(
    file_path: str,
    tokenizer: Tokenizer,
    max_length: int = 512,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Load and preprocess dataset.
    
    Args:
        file_path (str): Path to dataset file
        tokenizer (Tokenizer): Tokenizer instance
        max_length (int): Maximum sequence length
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader instance
    """
    # Implementation...
```

## Model Training

### `bilingual.train`

#### `train_model`
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, List[float]]:
    """Train a model with the given configuration.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (TrainingConfig): Training configuration
        device (str): Device to train on ('cuda' or 'cpu')
        
    Returns:
        dict: Training history with loss and metrics
    """
    # Implementation...
```

## Interactive Examples

### Style Transfer
```python
from bilingual import pipeline

# Initialize the style transfer model
style_transfer = pipeline(
    "text2text-generation",
    model="KothaGPT/style-transfer-gpt"
)

# Transfer from formal to informal
result = style_transfer("<informal> আমি ভাত খাই")
print(f"Informal: {result[0]['generated_text']}")
```

### Sentiment Analysis
```python
from bilingual import pipeline

# Initialize the sentiment classifier
classifier = pipeline(
    "text-classification",
    model="KothaGPT/sentiment-tone-classifier"
)

# Analyze sentiment
result = classifier("আমি খুব খুশি আজ")
print(f"Sentiment: {result[0]['label']} (Score: {result[0]['score']:.2f})")
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
