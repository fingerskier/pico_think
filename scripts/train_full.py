"""Train MLA + gating with frozen experts. Loss = CE + balance_loss."""

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pico_think.config import Config
from pico_think.model import PicoThink
from pico_think.mla import MLA
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

    print("Loading dataset...")
    dataset = PicoDataset(
        cfg.data_dir, cfg.tokenizer_path,
        seq_len=cfg.seq_len, stride=cfg.chunk_stride, min_len=cfg.min_chunk_len,
        bos_id=cfg.bos_id, eos_id=cfg.eos_id,
    )
    print(f"  {len(dataset)} chunks")
    loader = make_loader(dataset, cfg)

    # Build full model and load pre-trained weights
    model = PicoThink(cfg).to(device)
    model.load_pretrained(cfg.checkpoint_dir, device)
    model.vector_store.to(device)
    print("Loaded pre-trained weights")

    # Print param counts
    counts = model.count_params()
    for k, v in counts.items():
        print(f"  {k}: {v:,}")

    # Freeze experts, train MLA + gate + encoder + decoder
    model.freeze_experts()

    model = maybe_compile(model, cfg)

    # Trainable params: MLA + encoder + decoder
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.full_train_lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.full_train_epochs * len(loader)
    step = 0

    for epoch in range(1, cfg.full_train_epochs + 1):
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_bal = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.full_train_epochs}")

        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            with autocast_ctx:
                out = model(input_ids, use_store=False)
                logits = out["logits"]
                gate_weights = out["gate_weights"]

                ce_loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size),
                    target_ids.reshape(-1),
                    ignore_index=cfg.pad_id,
                )
                bal_loss = MLA.balance_loss(gate_weights)
                loss = ce_loss + cfg.balance_loss_weight * bal_loss

            lr = get_cosine_lr(step, cfg.warmup_steps, total_steps, cfg.full_train_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaled_loss = loss / cfg.grad_accum_steps
            scaler.scale(scaled_loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_bal += bal_loss.item()
            n_batches += 1
            step += 1

            avg_gate = gate_weights.mean(dim=(0, 1)).tolist()
            gate_str = "/".join(f"{g:.2f}" for g in avg_gate)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{ce_loss.item():.4f}",
                gate=gate_str,
            )

        avg = epoch_loss / max(n_batches, 1)
        avg_ce = epoch_ce / max(n_batches, 1)
        avg_bal = epoch_bal / max(n_batches, 1)
        print(f"  Epoch {epoch} — loss: {avg:.4f}, CE: {avg_ce:.4f}, balance: {avg_bal:.4f}")

    # Save full model
    save_path = Path(cfg.checkpoint_dir) / "full_model.pt"
    torch.save({
        "model": model.state_dict(),
        "config": vars(cfg),
    }, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
