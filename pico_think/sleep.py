"""Sleep consolidation: Grafting and Dreaming routines for vector store."""

import torch
import torch.nn.functional as F


def graft(vector_store, n_samples: int = 100, similarity_threshold: float = 0.95):
    """
    Grafting: merge highly similar vectors.
    - Sample random vectors
    - Find very similar pairs
    - Fuse them (average) into new vectors
    - Discard originals
    Returns number of merges performed.
    """
    if vector_store.count < 2:
        return 0

    n_samples = min(n_samples, vector_store.count)
    sampled, indices = vector_store.random_sample(n_samples)

    # Compute pairwise cosine similarities
    normed = F.normalize(sampled, dim=-1)
    sims = normed @ normed.T

    # Mask diagonal
    sims.fill_diagonal_(-1.0)

    merges = 0
    merged_set = set()
    to_delete = []
    to_add = []

    for i in range(n_samples):
        if i in merged_set:
            continue
        for j in range(i + 1, n_samples):
            if j in merged_set:
                continue
            if sims[i, j] > similarity_threshold:
                # Fuse: average the two vectors
                fused = (sampled[i] + sampled[j]) / 2.0
                to_add.append(fused)
                to_delete.extend([indices[i].item(), indices[j].item()])
                merged_set.add(i)
                merged_set.add(j)
                merges += 1
                break

    if to_delete:
        delete_indices = torch.tensor(to_delete, device=vector_store.device, dtype=torch.long)
        vector_store.delete(delete_indices)
    if to_add:
        new_vecs = torch.stack(to_add)
        vector_store.add(new_vecs)

    return merges


def dream(vector_store, n_samples: int = 50, dissimilarity_threshold: float = 0.1):
    """
    Dreaming: fuse dissimilar vectors to create novel combinations.
    - Sample random vectors
    - Find dissimilar pairs
    - Fuse them into new vectors (weighted average biased toward first)
    - Retain originals
    Returns number of new vectors created.
    """
    if vector_store.count < 2:
        return 0

    n_samples = min(n_samples, vector_store.count)
    sampled, _ = vector_store.random_sample(n_samples)

    normed = F.normalize(sampled, dim=-1)
    sims = normed @ normed.T
    sims.fill_diagonal_(1.0)  # ignore self

    created = 0
    used = set()
    to_add = []

    for i in range(n_samples):
        if i in used:
            continue
        for j in range(i + 1, n_samples):
            if j in used:
                continue
            if sims[i, j] < dissimilarity_threshold:
                # Fuse with bias toward first vector (0.7/0.3)
                fused = 0.7 * sampled[i] + 0.3 * sampled[j]
                to_add.append(fused)
                used.add(i)
                used.add(j)
                created += 1
                break

    if to_add:
        new_vecs = torch.stack(to_add)
        vector_store.add(new_vecs)

    return created


def sleep_cycle(vector_store, graft_rounds: int = 3, dream_rounds: int = 2,
                graft_samples: int = 200, dream_samples: int = 100,
                sim_threshold: float = 0.95, dissim_threshold: float = 0.1):
    """Run a full sleep cycle: multiple rounds of grafting then dreaming."""
    stats = {"grafts": 0, "dreams": 0, "initial_count": vector_store.count}

    for _ in range(graft_rounds):
        m = graft(vector_store, graft_samples, sim_threshold)
        stats["grafts"] += m

    for _ in range(dream_rounds):
        c = dream(vector_store, dream_samples, dissim_threshold)
        stats["dreams"] += c

    stats["final_count"] = vector_store.count
    return stats
