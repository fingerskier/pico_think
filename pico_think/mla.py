"""Multi-Head Latent Attention with KV compression and expert gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LatentAttentionLayer(nn.Module):
    """MHA with latent KV compression: K,V projected to latent_dim then back."""

    def __init__(self, d_model: int, n_heads: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        # KV compression: d_model → latent_dim → d_model
        self.kv_down = nn.Linear(d_model, latent_dim * 2)  # compress K and V together
        self.kv_up = nn.Linear(latent_dim, d_model * 2)    # expand back to K and V

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Compress KV to latent space
        kv_latent = self.kv_down(x)  # (B, T, latent_dim*2)
        kv_compressed = F.gelu(kv_latent)
        kv = self.kv_up(kv_compressed.chunk(2, dim=-1)[0] + kv_compressed.chunk(2, dim=-1)[1])
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention (causal)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.out_proj(out))


class MLABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LatentAttentionLayer(d_model, n_heads, latent_dim, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MLA(nn.Module):
    """Multi-Head Latent Attention with expert gating head."""

    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 latent_dim: int = 64, n_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLABlock(d_model, n_heads, latent_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Expert gating: produces per-position expert weights
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_experts),
        )
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, seq_len, d_model) — encoder output + retrieved vectors
        Returns:
            hidden: (batch, seq_len, d_model) — MLA output
            gate_weights: (batch, seq_len, n_experts) — softmax expert weights
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        gate_logits = self.gate(x)  # (B, T, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)

        return x, gate_weights

    @staticmethod
    def balance_loss(gate_weights: torch.Tensor) -> torch.Tensor:
        """Load-balancing loss to prevent expert collapse.
        Penalizes deviation from uniform distribution."""
        # Average gate weights across batch and sequence
        avg = gate_weights.mean(dim=(0, 1))  # (n_experts,)
        n = gate_weights.shape[-1]
        uniform = torch.ones_like(avg) / n
        return F.mse_loss(avg, uniform)
