# Rotary Positional Embedding (RoPE)

## Overview
Rotary Positional Embedding (RoPE) is a type of position encoding that encodes absolute positional information with rotation matrices. It's particularly effective in transformer architectures for sequence modeling tasks.

## Implementation

### Initialization

```python
class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) from RoFormer."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim  # Dimension of the embeddings (must be even)
        self.max_seq_len = max_seq_len  # Maximum sequence length
        self.base = base  # Base value for frequency calculation
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Initialize cache
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.max_seq_len_cached = 0
```

### Cache Management

```python
def _update_cos_sin_cache(self, seq_len: int, device=None):
    # Only update cache if needed
    if seq_len <= self.max_seq_len_cached and self.cos_cached is not None and self.cos_cached.device == device:
        return
        
    self.max_seq_len_cached = seq_len
    position = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # Compute frequencies
    freqs = torch.einsum('i,j->ij', position, self.inv_freq.to(device))
    emb = torch.cat([freqs, freqs], dim=-1)  # Duplicate for sin and cos
    
    # Cache the computed values
    self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
    self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
```

### Forward Pass

```python
def forward(self, x: Tensor, seq_dim: int = -2) -> Tensor:
    """
    Applies rotary embeddings to input tensor.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim] or [seq_len, batch_size, num_heads, head_dim]
        seq_dim: Dimension containing the sequence length (default: -2)
        
    Returns:
        Tensor with rotary position embeddings applied
    """
    seq_len = x.size(seq_dim)
    self._update_cos_sin_cache(seq_len, device=x.device)
    
    # Handle different input shapes
    if seq_dim != -2:
        x = x.transpose(1, seq_dim)
    
    # Apply rotation to the first 'dim' dimensions
    x_rot = x[..., :self.dim]
    x_pass = x[..., self.dim:]
    
    # Apply rotary embeddings
    x_rot = (x_rot * self.cos_cached[:, :, :seq_len] + 
             self._rotate_half(x_rot) * self.sin_cached[:, :, :seq_len])
    
    # Combine rotated and passed-through dimensions
    result = torch.cat([x_rot, x_pass], dim=-1)
    
    # Restore original shape if needed
    if seq_dim != -2:
        result = result.transpose(1, seq_dim)
    
    return result
```

### Helper Method

```python
def _rotate_half(self, x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return torch.cat([-x2, x1], dim=-1)
```

## Key Features

1. **Memory Efficiency**
   - Only stores base frequencies, not full matrices
   - Lazy initialization of caches
   - On-demand computation of positional encodings

2. **Numerical Stability**
   - Uses `float32` for frequency calculations
   - Proper handling of device placement
   - Careful management of tensor shapes

3. **Performance**
   - Caches computed values
   - Uses efficient tensor operations
   - Minimizes memory allocations

4. **Flexibility**
   - Handles different input shapes
   - Works on both CPU and CUDA
   - Supports variable sequence lengths

## Usage Example

```python
import torch
from bilingual.models.transformer_enhanced import RotaryPositionalEmbedding

# Initialize RoPE
rope = RotaryPositionalEmbedding(dim=64)

# Create random input tensor
x = torch.randn(2, 16, 8, 64)  # [batch_size, seq_len, num_heads, head_dim]

# Apply rotary positional embeddings
output = rope(x)

# Verify output shape
assert output.shape == x.shape
```

## Testing

Run the test suite with:

```bash
pytest tests/test_rope.py -v
```

Test coverage includes:
- Initialization and parameter validation
- Cache behavior
- Device handling
- Gradient computation
- Sequence length handling
- Numerical properties

## References
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
