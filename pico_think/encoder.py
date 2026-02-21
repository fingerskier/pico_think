"""Encoder: token embeddings + positional embeddings → 128D."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, d_model: int = 128,
                 max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) long tensor
        Returns:
            (batch, seq_len, d_model) float tensor
        """
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.norm(x)
        x = self.dropout(x)
        return x
