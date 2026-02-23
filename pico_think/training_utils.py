"""Shared training utilities for GPU-optimized training loops."""

import math
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler


def get_cosine_lr(step, warmup, total, lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(dataset, cfg):
    """Create a DataLoader with GPU-friendly settings."""
    use_workers = cfg.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.get_device() == "cuda"),
        persistent_workers=use_workers,
    )


def setup_amp(device):
    """Return (GradScaler, autocast_context) for AMP. No-op on CPU/MPS."""
    if device == "cuda":
        scaler = GradScaler("cuda")
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
        return scaler, autocast_ctx
    # CPU/MPS: no-op scaler and context
    scaler = GradScaler(enabled=False)
    return scaler, nullcontext()


def maybe_compile(model, cfg):
    """Wrap model in torch.compile() if enabled and available."""
    if not cfg.use_compile:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile not available, skipping")
        return model
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile failed ({e}), continuing without compilation")
    return model
