"""Pre-train encoder + decoder + transformer expert via next-token prediction."""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from pico_think.encoder import Encoder
from pico_think.decoder import Decoder
from pico_think.experts.transformer_expert import TransformerExpert
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

    print("Loading dataset...")
    dataset = PicoDataset(
        cfg.data_dir, cfg.tokenizer_path,
        seq_len=cfg.seq_len, stride=cfg.chunk_stride, min_len=cfg.min_chunk_len,
        bos_id=cfg.bos_id, eos_id=cfg.eos_id,
    )
    print(f"  {len(dataset)} chunks")

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    encoder = Encoder(cfg.vocab_size, cfg.d_model, cfg.seq_len, cfg.embed_dropout).to(device)
    decoder = Decoder(cfg.d_model, cfg.vocab_size).to(device)
    decoder.tie_weights(encoder)
    transformer = TransformerExpert(
        cfg.d_model, cfg.tf_n_layers, cfg.tf_n_heads, cfg.tf_ffn_dim, cfg.tf_dropout
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(transformer.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.pretrain_epochs * len(loader)
    step = 0

    for epoch in range(1, cfg.pretrain_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.pretrain_epochs}")

        for batch in pbar:
            batch = batch.to(device)
            # Input: all tokens except last; Target: all tokens except first
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # Forward
            embeds = encoder(input_ids)
            hidden = transformer(embeds)
            logits = decoder(hidden)

            # Loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                target_ids.reshape(-1),
                ignore_index=cfg.pad_id,
            )

            # Cosine LR
            lr = get_cosine_lr(step, cfg.warmup_steps, total_steps, cfg.pretrain_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

    # Save
    ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "transformer": transformer.state_dict(),
        "config": vars(cfg),
    }
    save_path = Path(cfg.checkpoint_dir) / "pretrain_transformer.pt"
    torch.save(ckpt, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
