import pytest
import torch
import torch.nn as nn

from src.bilingual.models.transformer_enhanced import RotaryPositionalEmbedding


class TestRotaryPositionalEmbedding:
    @pytest.fixture
    def rope(self):
        return RotaryPositionalEmbedding(dim=64, max_seq_len=4096)

    def test_initialization(self, rope):
        assert rope.dim == 64
        assert rope.max_seq_len == 4096
        assert rope.base == 10000.0
        assert rope.inv_freq.shape == (32,)  # dim // 2
        assert rope.cos_cached is None
        assert rope.sin_cached is None
        assert rope.max_seq_len_cached == 0

    def test_rotate_half(self, rope):
        x = torch.arange(10, dtype=torch.float32).view(1, 1, 1, 10)
        rotated = rope._rotate_half(x)
        assert rotated.shape == x.shape

        # For even dimensions, first half should be negated second half
        # and second half should be first half
        half = x.shape[-1] // 2
        assert torch.allclose(rotated[..., :half], -x[..., half:], atol=1e-6)
        assert torch.allclose(rotated[..., half:], x[..., :half], atol=1e-6)

    def test_cache_update(self, rope):
        # First call should update cache
        seq_len = 128
        device = torch.device("cpu")

        rope._update_cos_sin_cache(seq_len, device)

        assert rope.cos_cached is not None
        assert rope.sin_cached is not None
        assert rope.max_seq_len_cached == seq_len
        assert rope.cos_cached.shape == (1, 1, seq_len, 64)  # [1, 1, seq_len, dim]

        # Second call with same or smaller seq_len should not update
        prev_cos = rope.cos_cached.clone()
        rope._update_cos_sin_cache(seq_len // 2, device)
        assert torch.equal(rope.cos_cached, prev_cos)

        # Call with larger seq_len should update
        new_seq_len = 256
        rope._update_cos_sin_cache(new_seq_len, device)
        assert rope.max_seq_len_cached == new_seq_len
        assert rope.cos_cached.shape == (1, 1, new_seq_len, 64)

    def test_forward_shape(self, rope):
        batch_size = 4
        seq_len = 32
        num_heads = 8
        head_dim = 64

        # Test with different input shapes
        input_shapes = [
            (batch_size, seq_len, num_heads, head_dim),  # [batch, seq_len, heads, dim]
            (seq_len, batch_size, num_heads, head_dim),  # [seq_len, batch, heads, dim]
        ]

        for shape in input_shapes:
            x = torch.randn(shape)
            output = rope(x, seq_dim=-3 if shape[0] == seq_len else -2)
            assert output.shape == x.shape

    def test_rotary_properties(self, rope):
        # Test that rotating back and forth returns the original
        x = torch.randn(2, 16, 8, 64)  # [batch, seq, heads, dim]

        # Apply rotary embedding
        y = rope(x)

        # Apply inverse rotation (negative angles)
        rope.inv_freq = -rope.inv_freq
        z = rope(y)

        # Should be close to original (within numerical precision)
        assert torch.allclose(x, z, atol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_device_handling(self, rope, device):
        device = torch.device(device)
        rope = rope.to(device)

        x = torch.randn(2, 16, 8, 64, device=device)
        output = rope(x)

        assert output.device == device
        if rope.cos_cached is not None:
            assert rope.cos_cached.device == device

    def test_gradients(self, rope):
        # Test that gradients flow through the rotary embedding
        x = torch.randn(2, 16, 8, 64, requires_grad=True)
        output = rope(x)

        # Create dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_long_sequences(self):
        # Test with sequence length much larger than initial max_seq_len
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=1024)
        x = torch.randn(1, 2048, 1, 64)  # Longer sequence

        # Should automatically handle longer sequences
        output = rope(x)
        assert output.shape == x.shape
        assert rope.max_seq_len_cached == 2048


if __name__ == "__main__":
    pytest.main([__file__])
