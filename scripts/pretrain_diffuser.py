"""Pre-train diffuser expert: noise prediction MSE loss with frozen encoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from config import Config
from pico_think.encoder import Encoder
from pico_think.experts.diffuser_expert import DiffuserExpert
from pico_think.dataset import PicoDataset
from pico_think.training_utils import get_cosine_lr, make_loader, setup_amp, maybe_compile


def main():
    cfg = Config()
    cfg.ensure_dirs()
    cfg.setup_backends()
    device = cfg.get_device()
    print(f"Device: {device}")

    scaler, autocast_ctx = setup_amp(device if cfg.use_amp else "cpu")
    print(f"Using AMP: {cfg.use_amp and device == 'cuda'}")

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
    loader = make_loader(dataset, cfg)

    diffuser = DiffuserExpert(
        cfg.d_model, cfg.diff_n_blocks, cfg.diff_n_steps,
        cfg.diff_sample_steps, cfg.diff_dropout,
    ).to(device)

    diffuser = maybe_compile(diffuser, cfg)

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
            batch = batch.to(device, non_blocking=True)
            with torch.no_grad():
                embeds = encoder(batch)

            with autocast_ctx:
                loss = diffuser.training_loss(embeds)

            lr = get_cosine_lr(step, cfg.warmup_steps, total_steps, cfg.pretrain_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaled_loss = loss / cfg.grad_accum_steps
            scaler.scale(scaled_loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffuser.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

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
