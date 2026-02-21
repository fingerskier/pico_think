# TODO

## Periodic Checkpoint Saving

Currently all training scripts only save a checkpoint at the end of training. Add periodic saving to guard against losing progress during long runs.

**Scripts to update:**
- `scripts/pretrain_transformer.py`
- `scripts/pretrain_diffuser.py`
- `scripts/pretrain_state_space.py`
- `scripts/train_full.py`

**Requirements:**
- Save every N epochs (configurable via `checkpoint_interval` in config)
- Use timestamped or epoch-numbered filenames (e.g. `pretrain_transformer_epoch_50.pt`)
- Keep a `latest.pt` symlink or overwrite for easy resume
- Add `--resume` flag to load from a periodic checkpoint and continue training
- Optionally limit max checkpoints kept on disk to avoid filling storage
