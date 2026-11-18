import json

import pytest
import torch
from safetensors.torch import save_model
from tokenizers import Tokenizer, models, pre_tokenizers

from loopinfer.models import DenseToLoop, Ouro


@pytest.fixture(autouse=True, scope="session")
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["ouro", "dense-to-loop"])
def checkpoint(tmp_path, request):
    kind = request.param
    config = {
        "model_type": kind, "hidden_size": 24, "intermediate_size": 32, "head_dim": 8,
        "num_hidden_layers": 2, "num_attention_heads": 3, "num_key_value_heads": 1,
        "max_position_embeddings": 128, "vocab_size": 12, "total_ut_steps": 4,
        "rope_theta": 10000.0, "bos_token_id": 1, "eos_token_id": 2,
        "tie_word_embeddings": kind == "dense-to-loop",
    }
    torch.manual_seed(123)
    model = (Ouro if kind == "ouro" else DenseToLoop)(config)
    folder = tmp_path / kind
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tokenizer = Tokenizer(models.WordLevel(
        vocab={word: i for i, word in enumerate(["[UNK]", "<s>", "</s>", "a", "b", "c", "d", "e", "f", "g"])},
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<s>", "</s>"])
    tokenizer.save(str(folder / "tokenizer.json"))
    save_model(model, folder / "model.safetensors")
    return folder
