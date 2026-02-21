"""Interactive CLI chat interface for PicoThink."""

import argparse

import torch
from tokenizers import Tokenizer

from pico_think.config import Config
from pico_think.model import PicoThink
from pico_think.sleep import sleep_cycle


def main():
    parser = argparse.ArgumentParser(description="PicoThink Chat")
    parser.add_argument("--checkpoint", default="checkpoints/full_model.pt",
                        help="Path to full model checkpoint")
    parser.add_argument("--store", default="checkpoints/vector_store.pt",
                        help="Path to vector store")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()

    cfg = Config()
    device = cfg.get_device()
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    print(f"Tokenizer loaded (vocab: {tokenizer.get_vocab_size()})")

    # Load model
    model = PicoThink(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Model loaded")

    # Load vector store
    store_path = Path(args.store)
    if store_path.exists():
        model.vector_store.load(str(store_path), device=device)
        print(f"Vector store loaded: {model.vector_store.count} vectors")
    else:
        model.vector_store.to(device)
        print("Starting with empty vector store")

    # Print stats
    counts = model.count_params()
    total = counts["total"]
    print(f"Total params: {total:,}")
    print()
    print("Commands: /quit  /sleep  /stats  /save  /clear")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            print("Bye!")
            break

        if user_input.lower() == "/sleep":
            print("Running sleep consolidation...")
            stats = sleep_cycle(model.vector_store)
            print(f"  Grafts: {stats['grafts']}, Dreams: {stats['dreams']}")
            print(f"  Vectors: {stats['initial_count']} → {stats['final_count']}")
            continue

        if user_input.lower() == "/stats":
            print(f"  Vector store: {model.vector_store.count} vectors")
            counts = model.count_params()
            for k, v in counts.items():
                print(f"  {k}: {v:,}")
            continue

        if user_input.lower() == "/save":
            model.vector_store.save(str(store_path))
            print(f"  Vector store saved ({model.vector_store.count} vectors)")
            continue

        if user_input.lower() == "/clear":
            model.vector_store = type(model.vector_store)(
                cfg.d_model, cfg.vs_max_vectors, cfg.vs_top_k
            ).to(device)
            print("  Vector store cleared")
            continue

        # Encode input
        encoded = tokenizer.encode(user_input)
        token_ids = [cfg.bos_id] + encoded.ids
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_tensor,
                max_new=args.max_new,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        # Decode (skip special tokens)
        gen_ids = output_ids[0]
        # Remove BOS if present at start
        if gen_ids and gen_ids[0] == cfg.bos_id:
            gen_ids = gen_ids[1:]
        # Remove input tokens to get just the generation
        gen_ids = gen_ids[len(encoded.ids):]
        response = tokenizer.decode(gen_ids)
        print(f"\nPicoThink: {response}")


if __name__ == "__main__":
    main()
