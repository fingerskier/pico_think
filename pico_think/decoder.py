"""Decoder: 128D → vocab logits with weight tying to encoder."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, d_model: int = 128, vocab_size: int = 4096):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def tie_weights(self, encoder: nn.Module):
        """Tie projection weights to encoder token embeddings."""
        self.proj.weight = encoder.token_embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        x = self.norm(x)
        return self.proj(x)
