"""Create a deterministic CPU fixture for the install and serving demo."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer, models, pre_tokenizers

from loopinfer.models import Ouro


def make_checkpoint(folder: Path):
    config = {
        "model_type": "ouro", "hidden_size": 24, "intermediate_size": 32, "head_dim": 8,
        "num_hidden_layers": 2, "num_attention_heads": 3, "num_key_value_heads": 1,
        "max_position_embeddings": 128, "vocab_size": 10, "total_ut_steps": 4,
        "rope_theta": 10000, "bos_token_id": 1, "eos_token_id": 2,
    }
    vocab = ["[UNK]", "<s>", "</s>", "the", "cat", "sat", "on", "a", "mat", "."]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocab)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<s>", "</s>"])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    tokenizer.save(str(folder / "tokenizer.json"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        save_model(Ouro(config), folder / "model.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    make_checkpoint(parser.parse_args().output)
