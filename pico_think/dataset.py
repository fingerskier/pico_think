"""JSONL dataset loading, tokenization, and chunking."""

import json
from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class PicoDataset(Dataset):
    """Loads JSONL files, tokenizes text, and produces fixed-length chunks."""

    def __init__(self, data_dir: str, tokenizer_path: str, seq_len: int = 256,
                 stride: int = 128, min_len: int = 16,
                 bos_id: int = 1, eos_id: int = 2):
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.chunks = []

        data_path = Path(data_dir)
        for jsonl_file in sorted(data_path.rglob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text:
                        continue
                    token_ids = self.tokenizer.encode(text).ids
                    # Wrap with BOS/EOS
                    token_ids = [bos_id] + token_ids + [eos_id]
                    # Chunk with stride
                    if len(token_ids) < min_len:
                        continue
                    for start in range(0, max(1, len(token_ids) - seq_len + 1), stride):
                        chunk = token_ids[start:start + seq_len]
                        if len(chunk) < min_len:
                            continue
                        self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        import torch
        chunk = self.chunks[idx]
        # Pad to seq_len
        padded = chunk + [0] * (self.seq_len - len(chunk))
        return torch.tensor(padded, dtype=torch.long)
