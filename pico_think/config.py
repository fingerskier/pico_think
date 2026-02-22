"""PicoThink hyperparameters and configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    # Tokenizer / vocab
    vocab_size: int = 4096
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3

    # Model dimensions
    d_model: int = 128
    seq_len: int = 256

    # Encoder / Decoder
    embed_dropout: float = 0.1

    # Transformer expert
    tf_n_layers: int = 4
    tf_n_heads: int = 4
    tf_ffn_dim: int = 256
    tf_dropout: float = 0.1

    # Diffuser expert
    diff_n_blocks: int = 3
    diff_n_steps: int = 50          # training diffusion steps
    diff_sample_steps: int = 8      # DDIM sampling steps
    diff_dropout: float = 0.1

    # State-Space expert (Mamba-style)
    ssm_n_layers: int = 4
    ssm_state_dim: int = 16
    ssm_expand: int = 2
    ssm_dropout: float = 0.1

    # MLA (Multi-Head Latent Attention)
    mla_n_layers: int = 2
    mla_n_heads: int = 4
    mla_latent_dim: int = 64        # KV compression dim
    mla_dropout: float = 0.1
    n_experts: int = 3
    balance_loss_weight: float = 0.01

    # Vector store
    vs_max_vectors: int = 100_000
    vs_top_k: int = 4

    # Training
    batch_size: int = 32
    pretrain_epochs: int = 1
    full_train_epochs: int = 1
    pretrain_lr: float = 3e-4
    full_train_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    chunk_stride: int = 128
    min_chunk_len: int = 16

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # GPU optimization
    use_amp: bool = True              # automatic mixed precision (fp16)
    num_workers: int = 2              # DataLoader workers (0 = main thread)
    use_compile: bool = False         # torch.compile() — opt-in, can be flaky on Windows
    grad_accum_steps: int = 1         # gradient accumulation (effective batch = batch_size * this)

    def get_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def setup_backends(self):
        """Enable GPU-specific backend optimizations when running on CUDA."""
        if self.get_device() == "cuda":
            import torch
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def ensure_dirs(self):
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
