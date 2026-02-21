"""Vector store: cosine-similarity tensor store with FIFO replacement."""

import torch
import torch.nn.functional as F
from pathlib import Path


class VectorStore:
    def __init__(self, d_model: int = 128, max_vectors: int = 100_000, top_k: int = 4):
        self.d_model = d_model
        self.max_vectors = max_vectors
        self.top_k = top_k
        self.vectors = torch.zeros(0, d_model)
        self.write_idx = 0
        self._norms_dirty = True
        self._normalized: torch.Tensor | None = None

    def to(self, device):
        self.vectors = self.vectors.to(device)
        self._norms_dirty = True
        self._normalized = None
        return self

    @property
    def device(self):
        return self.vectors.device

    @property
    def count(self):
        return self.vectors.shape[0]

    def add(self, vecs: torch.Tensor):
        """Add vectors to the store. vecs: (N, d_model)."""
        vecs = vecs.detach().to(self.vectors.device)
        if vecs.dim() == 1:
            vecs = vecs.unsqueeze(0)

        N = vecs.shape[0]
        self._norms_dirty = True

        if self.count < self.max_vectors:
            space = self.max_vectors - self.count
            if N <= space:
                self.vectors = torch.cat([self.vectors, vecs], dim=0)
            else:
                self.vectors = torch.cat([self.vectors, vecs[:space]], dim=0)
                # FIFO overwrite from beginning (vectorized)
                remaining = N - space
                indices = torch.arange(remaining, device=self.vectors.device)
                write_positions = (self.write_idx + indices) % self.max_vectors
                self.vectors[write_positions] = vecs[space:]
                self.write_idx = (self.write_idx + remaining) % self.max_vectors
        else:
            # FIFO overwrite (vectorized)
            indices = torch.arange(N, device=self.vectors.device)
            write_positions = (self.write_idx + indices) % self.max_vectors
            self.vectors[write_positions] = vecs
            self.write_idx = (self.write_idx + N) % self.max_vectors

    def _ensure_normalized(self):
        """Lazily compute and cache normalized vectors."""
        if self._norms_dirty or self._normalized is None:
            self._normalized = F.normalize(self.vectors, dim=-1)
            self._norms_dirty = False

    def search(self, query: torch.Tensor, top_k: int = None) -> torch.Tensor:
        """
        Search for top-k most similar vectors.
        Args:
            query: (d_model,) or (1, d_model)
        Returns:
            (top_k, d_model) tensor of retrieved vectors, or empty if store is empty
        """
        if self.count == 0:
            return torch.zeros(0, self.d_model, device=self.device)

        top_k = top_k or self.top_k
        top_k = min(top_k, self.count)

        if query.dim() == 2:
            query = query.squeeze(0)

        # Cosine similarity with cached norms
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        self._ensure_normalized()
        sims = (query_norm @ self._normalized.T).squeeze(0)

        _, indices = sims.topk(top_k)
        return self.vectors[indices]

    def random_sample(self, n: int) -> tuple:
        """Sample n random vectors, returning (vectors, indices)."""
        if self.count == 0:
            return torch.zeros(0, self.d_model, device=self.device), torch.zeros(0, dtype=torch.long)
        n = min(n, self.count)
        indices = torch.randperm(self.count, device=self.device)[:n]
        return self.vectors[indices], indices

    def replace(self, indices: torch.Tensor, new_vecs: torch.Tensor):
        """Replace vectors at given indices."""
        self.vectors[indices] = new_vecs.detach()
        self._norms_dirty = True

    def delete(self, indices: torch.Tensor):
        """Delete vectors at indices by moving last vectors into their slots."""
        mask = torch.ones(self.count, dtype=torch.bool, device=self.device)
        mask[indices] = False
        self.vectors = self.vectors[mask]
        self._norms_dirty = True

    def save(self, path: str):
        torch.save({
            "vectors": self.vectors.cpu(),
            "write_idx": self.write_idx,
            "d_model": self.d_model,
            "max_vectors": self.max_vectors,
            "top_k": self.top_k,
        }, path)

    def load(self, path: str, device=None):
        data = torch.load(path, map_location=device or "cpu", weights_only=True)
        self.vectors = data["vectors"]
        self.write_idx = data["write_idx"]
        self.d_model = data["d_model"]
        self.max_vectors = data["max_vectors"]
        self.top_k = data["top_k"]
        self._norms_dirty = True
        self._normalized = None
        if device:
            self.vectors = self.vectors.to(device)
