"""PicoThink: full model orchestration.

Forward pass: encode → retrieve → MLA → experts → combine → store → decode
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .mla import MLA
from .vector_store import VectorStore
from .experts.transformer_expert import TransformerExpert
from .experts.diffuser_expert import DiffuserExpert
from .experts.state_space_expert import StateSpaceExpert


class PicoThink(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(cfg.vocab_size, cfg.d_model, cfg.seq_len, cfg.embed_dropout)
        self.decoder = Decoder(cfg.d_model, cfg.vocab_size)
        self.decoder.tie_weights(self.encoder)

        self.mla = MLA(
            cfg.d_model, cfg.mla_n_layers, cfg.mla_n_heads,
            cfg.mla_latent_dim, cfg.n_experts, cfg.mla_dropout,
        )

        self.transformer = TransformerExpert(
            cfg.d_model, cfg.tf_n_layers, cfg.tf_n_heads,
            cfg.tf_ffn_dim, cfg.tf_dropout,
        )
        self.diffuser = DiffuserExpert(
            cfg.d_model, cfg.diff_n_blocks, cfg.diff_n_steps,
            cfg.diff_sample_steps, cfg.diff_dropout,
        )
        self.ssm = StateSpaceExpert(
            cfg.d_model, cfg.ssm_n_layers, cfg.ssm_state_dim,
            cfg.ssm_expand, cfg.ssm_dropout,
        )

        self.vector_store = VectorStore(cfg.d_model, cfg.vs_max_vectors, cfg.vs_top_k)

    def forward(self, token_ids: torch.Tensor, use_store: bool = True) -> dict:
        """
        Full forward pass.
        Args:
            token_ids: (batch, seq_len) long tensor
            use_store: whether to use vector store retrieval
        Returns:
            dict with keys: logits, gate_weights, hidden
        """
        # 1. Encode
        embeds = self.encoder(token_ids)  # (B, T, D)

        # 2. Retrieve from vector store and prepend
        if use_store and self.vector_store.count > 0:
            # Use mean of last position as query
            query = embeds[:, -1].mean(dim=0)  # (D,)
            retrieved = self.vector_store.search(query)  # (top_k, D)
            if retrieved.shape[0] > 0:
                # Prepend retrieved vectors as prefix context
                prefix = retrieved.unsqueeze(0).expand(embeds.shape[0], -1, -1)
                embeds = torch.cat([prefix, embeds], dim=1)  # (B, top_k+T, D)

        # 3. MLA: attend and get expert gate weights
        mla_out, gate_weights = self.mla(embeds)  # (B, T', D), (B, T', 3)

        # 4. Expert processing
        tf_out = self.transformer(mla_out)
        diff_out = self.diffuser(mla_out)
        ssm_out = self.ssm(mla_out)

        # 5. Weighted combination
        w = gate_weights  # (B, T', 3)
        combined = (
            w[:, :, 0:1] * tf_out +
            w[:, :, 1:2] * diff_out +
            w[:, :, 2:3] * ssm_out
        )

        # If we prepended retrieved vectors, strip them
        if use_store and self.vector_store.count > 0 and retrieved.shape[0] > 0:
            n_prefix = retrieved.shape[0]
            combined = combined[:, n_prefix:]
            gate_weights = gate_weights[:, n_prefix:]

        # 6. Store the mean representation
        if use_store:
            store_vec = combined.detach().mean(dim=(0, 1))  # (D,)
            self.vector_store.add(store_vec.unsqueeze(0))

        # 7. Decode
        logits = self.decoder(combined)  # (B, T, vocab)

        return {
            "logits": logits,
            "gate_weights": gate_weights,
            "hidden": combined,
        }

    def freeze_experts(self):
        """Freeze all expert parameters for full-model training."""
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.diffuser.parameters():
            p.requires_grad = False
        for p in self.ssm.parameters():
            p.requires_grad = False
        self.transformer.eval()
        self.diffuser.eval()
        self.ssm.eval()

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def load_pretrained(self, checkpoint_dir: str, device: str = "cpu"):
        """Load all pre-trained component weights."""
        from pathlib import Path
        ckpt_dir = Path(checkpoint_dir)

        # Transformer pre-training (includes encoder + decoder)
        tf_ckpt = torch.load(ckpt_dir / "pretrain_transformer.pt",
                             map_location=device, weights_only=True)
        self.encoder.load_state_dict(tf_ckpt["encoder"])
        self.decoder.load_state_dict(tf_ckpt["decoder"])
        self.transformer.load_state_dict(tf_ckpt["transformer"])

        # Diffuser
        diff_ckpt = torch.load(ckpt_dir / "pretrain_diffuser.pt",
                               map_location=device, weights_only=True)
        self.diffuser.load_state_dict(diff_ckpt["diffuser"])

        # SSM
        ssm_ckpt = torch.load(ckpt_dir / "pretrain_ssm.pt",
                              map_location=device, weights_only=True)
        self.ssm.load_state_dict(ssm_ckpt["ssm"])

    def count_params(self) -> dict:
        """Count parameters by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "encoder": _count(self.encoder),
            "decoder": sum(p.numel() for p in self.decoder.parameters()
                          if p.data_ptr() != self.encoder.token_embed.weight.data_ptr()),
            "mla": _count(self.mla),
            "transformer": _count(self.transformer),
            "diffuser": _count(self.diffuser),
            "ssm": _count(self.ssm),
            "total": _count(self),
        }

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new: int = 128,
                 temperature: float = 0.8, top_k: int = 40) -> list:
        """Autoregressive generation."""
        self.eval()
        B = token_ids.shape[0]
        generated = token_ids.tolist()

        for _ in range(max_new):
            # Truncate to seq_len
            ctx = token_ids[:, -self.cfg.seq_len:]
            out = self.forward(ctx, use_store=True)
            logits = out["logits"][:, -1] / temperature  # (B, vocab)

            # Top-k filtering
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)  # (B, 1)
            token_ids = torch.cat([token_ids, next_tok], dim=1)

            tok_id = next_tok[0].item()
            if tok_id == self.cfg.eos_id:
                break
            generated[0].append(tok_id)

        return generated
