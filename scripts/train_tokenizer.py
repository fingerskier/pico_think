"""Train a BPE tokenizer on all JSONL data files."""

import json
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from pico_think.config import Config


def iter_texts(data_dir: str):
    """Yield text strings from all JSONL files."""
    data_path = Path(data_dir)
    for jsonl_file in sorted(data_path.rglob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    yield text


def main():
    cfg = Config()
    cfg.ensure_dirs()

    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
        min_frequency=2,
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab_size={cfg.vocab_size})...")
    texts = list(iter_texts(cfg.data_dir))
    print(f"  Collected {len(texts)} texts")

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.save(cfg.tokenizer_path)
    print(f"Tokenizer saved to {cfg.tokenizer_path}")

    # Verify
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    print(f"  Vocab size: {tok.get_vocab_size()}")
    sample = "The quick brown fox jumps over the lazy dog."
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded.ids)
    print(f"  Sample encode: {sample!r}")
    print(f"  Token IDs: {encoded.ids[:20]}...")
    print(f"  Decoded:   {decoded!r}")


if __name__ == "__main__":
    main()
