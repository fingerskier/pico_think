"""State-Space Expert: Mamba-style selective scan with HiPPO initialization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def hippo_init(state_dim: int) -> torch.Tensor:
    """HiPPO-LegS initialization matrix."""
    A = torch.zeros(state_dim, state_dim)
    for n in range(state_dim):
        for k in range(state_dim):
            if n > k:
                A[n, k] = math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = n + 1
    return -A


class MambaBlock(nn.Module):
    """Simplified Mamba block with selective scan."""

    def __init__(self, d_model: int, state_dim: int = 64,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.state_dim = state_dim

        self.norm = nn.LayerNorm(d_model)
        # Input projection: split into main path and gate
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # SSM parameters (selective, input-dependent)
        self.s_B = nn.Linear(d_inner, state_dim, bias=False)
        self.s_C = nn.Linear(d_inner, state_dim, bias=False)
        self.s_dt = nn.Linear(d_inner, d_inner, bias=True)

        # Discretized A (log-space for stability)
        A = hippo_init(state_dim)
        # Use diagonal approximation for efficiency
        A_diag = torch.diagonal(A)
        self.log_A = nn.Parameter(torch.log(-A_diag).unsqueeze(0).expand(d_inner, -1))

        # 1D conv for local context
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        # Project and split
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Conv (causal: trim future)
        x_conv = x_main.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv(x_conv)[:, :, :T]  # trim to causal
        x_main = x_conv.transpose(1, 2)
        x_main = F.silu(x_main)

        # Selective scan
        out = self._selective_scan(x_main)

        # Gate and project
        out = out * F.silu(z)
        out = self.dropout(self.out_proj(out))
        return residual + out

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Input-dependent discrete SSM scan."""
        B, T, d_inner = x.shape

        # Compute input-dependent parameters
        B_t = self.s_B(x)  # (B, T, state_dim)
        C_t = self.s_C(x)  # (B, T, state_dim)
        dt = F.softplus(self.s_dt(x))  # (B, T, d_inner)

        # Discretize A
        A = -torch.exp(self.log_A)  # (d_inner, state_dim)

        # Sequential scan
        h = torch.zeros(B, self.d_inner, self.state_dim, device=x.device)
        outputs = []

        for t in range(T):
            # dt_t: (B, d_inner)
            dt_t = dt[:, t]
            # Discretize: dA = exp(A * dt), dB = dt * B
            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, d_inner, state_dim)
            dB = dt_t.unsqueeze(-1) * B_t[:, t].unsqueeze(1)  # (B, d_inner, state_dim)
            # State update: h = dA * h + dB * x
            h = dA * h + dB * x[:, t].unsqueeze(-1)
            # Output: y = C * h
            y = (h * C_t[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y)

        return torch.stack(outputs, dim=1)  # (B, T, d_inner)


class StateSpaceExpert(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 4,
                 state_dim: int = 64, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, state_dim, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
