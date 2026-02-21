"""Pre-train diffuser expert: noise prediction MSE loss with frozen encoder."""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from pico_think.encoder import Encoder
from pico_think.experts.diffuser_expert import DiffuserExpert
from pico_think.dataset import PicoDataset


def get_cosine_lr(step, warmup, total, lr):
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    cfg = Config()
    cfg.ensure_dirs()
    device = cfg.get_device()
    print(f"Device: {device}")

    # Load pre-trained encoder (frozen)
    ckpt = torch.load(
        Path(cfg.checkpoint_dir) / "pretrain_transformer.pt",
        map_location=device, weights_only=True,
    )
    encoder = Encoder(cfg.vocab_size, cfg.d_model, cfg.seq_len, cfg.embed_dropout).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("Loaded frozen encoder")

    print("Loading dataset...")
    dataset = PicoDataset(
        cfg.data_dir, cfg.tokenizer_path,
        seq_len=cfg.seq_len, stride=cfg.chunk_stride, min_len=cfg.min_chunk_len,
        bos_id=cfg.bos_id, eos_id=cfg.eos_id,
    )
    print(f"  {len(dataset)} chunks")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    diffuser = DiffuserExpert(
        cfg.d_model, cfg.diff_n_blocks, cfg.diff_n_steps,
        cfg.diff_sample_steps, cfg.diff_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        diffuser.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.pretrain_epochs * len(loader)
    step = 0

    for epoch in range(1, cfg.pretrain_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.pretrain_epochs}")

        for batch in pbar:
            batch = batch.to(device)
            with torch.no_grad():
                embeds = encoder(batch)

            loss = diffuser.training_loss(embeds)

            lr = get_cosine_lr(step, cfg.warmup_steps, total_steps, cfg.pretrain_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffuser.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

    save_path = Path(cfg.checkpoint_dir) / "pretrain_diffuser.pt"
    torch.save({"diffuser": diffuser.state_dict(), "config": vars(cfg)}, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
