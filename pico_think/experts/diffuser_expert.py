"""Diffuser Expert: 128D denoising diffusion with AdaLN and DDIM sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) integer timesteps → (batch, d_model)."""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class AdaLNBlock(nn.Module):
    """Denoiser block with Adaptive Layer Normalization."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        # AdaLN modulation: scale and shift for both norms
        self.adaln = nn.Linear(d_model, d_model * 4)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            t_emb: (batch, d_model) timestep embedding
        """
        # Compute modulation params
        mod = self.adaln(t_emb).unsqueeze(1)  # (B, 1, 4*D)
        scale1, shift1, scale2, shift2 = mod.chunk(4, dim=-1)

        # First sub-layer
        h = self.norm1(x) * (1 + scale1) + shift1
        h = self.dropout(self.linear(h))
        x = x + h

        # Second sub-layer (FFN)
        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.dropout(self.ffn(h))
        x = x + h
        return x


class DiffuserExpert(nn.Module):
    def __init__(self, d_model: int = 128, n_blocks: int = 3,
                 n_steps: int = 50, sample_steps: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps
        self.sample_steps = sample_steps

        self.time_embed = TimestepEmbedding(d_model)
        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, dropout) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

        # Pre-compute noise schedule (linear beta schedule)
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> tuple:
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        acp = self.alphas_cumprod[t]  # (B,)
        # Reshape for broadcasting over seq_len and d_model
        acp = acp.view(-1, 1, 1)
        xt = torch.sqrt(acp) * x0 + torch.sqrt(1.0 - acp) * noise
        return xt, noise

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise in xt at timestep t."""
        t_emb = self.time_embed(t)
        h = xt
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(h)

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute MSE noise prediction loss."""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = self.predict_noise(xt, t)
        return F.mse_loss(pred_noise, noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        DDIM sampling: refine from input x (not pure noise).
        Used during inference in the full model.
        """
        # Use a subset of timesteps for DDIM
        step_size = max(1, self.n_steps // self.sample_steps)
        timesteps = list(range(self.n_steps - 1, -1, -step_size))

        xt = x
        for i, t_val in enumerate(timesteps):
            t = torch.full((x.shape[0],), t_val, device=x.device, dtype=torch.long)
            pred_noise = self.predict_noise(xt, t)

            acp_t = self.alphas_cumprod[t_val]
            # Predict x0
            x0_pred = (xt - torch.sqrt(1 - acp_t) * pred_noise) / torch.sqrt(acp_t)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                acp_prev = self.alphas_cumprod[t_prev]
                # DDIM deterministic step
                xt = torch.sqrt(acp_prev) * x0_pred + \
                     torch.sqrt(1 - acp_prev) * pred_noise
            else:
                xt = x0_pred

        return xt
