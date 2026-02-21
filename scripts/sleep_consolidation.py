"""CLI for running sleep consolidation on the vector store."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from config import Config
from pico_think.vector_store import VectorStore
from pico_think.sleep import sleep_cycle


def main():
    parser = argparse.ArgumentParser(description="Run sleep consolidation")
    parser.add_argument("--store", default="checkpoints/vector_store.pt",
                        help="Path to vector store checkpoint")
    parser.add_argument("--graft-rounds", type=int, default=3)
    parser.add_argument("--dream-rounds", type=int, default=2)
    parser.add_argument("--graft-samples", type=int, default=200)
    parser.add_argument("--dream-samples", type=int, default=100)
    parser.add_argument("--sim-threshold", type=float, default=0.95)
    parser.add_argument("--dissim-threshold", type=float, default=0.1)
    args = parser.parse_args()

    cfg = Config()
    device = cfg.get_device()

    store = VectorStore(cfg.d_model, cfg.vs_max_vectors, cfg.vs_top_k)
    store_path = Path(args.store)
    if store_path.exists():
        store.load(str(store_path), device=device)
        print(f"Loaded vector store: {store.count} vectors")
    else:
        print(f"No vector store found at {store_path}")
        return

    print("Running sleep cycle...")
    stats = sleep_cycle(
        store,
        graft_rounds=args.graft_rounds,
        dream_rounds=args.dream_rounds,
        graft_samples=args.graft_samples,
        dream_samples=args.dream_samples,
        sim_threshold=args.sim_threshold,
        dissim_threshold=args.dissim_threshold,
    )

    print(f"  Initial vectors: {stats['initial_count']}")
    print(f"  Grafts (merges):  {stats['grafts']}")
    print(f"  Dreams (created): {stats['dreams']}")
    print(f"  Final vectors:   {stats['final_count']}")

    store.save(str(store_path))
    print(f"Saved to {store_path}")


if __name__ == "__main__":
    main()
