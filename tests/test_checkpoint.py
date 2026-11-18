import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from loopinfer import load_model
from loopinfer.models import DenseToLoop
from tools.export_dense_to_loop import export


def test_sharded_weights(checkpoint):
    path = checkpoint / "model.safetensors"
    state = load_file(path)
    path.unlink()
    weight_map = {}
    for index in range(2):
        name = f"model-{index:05d}-of-00002.safetensors"
        shard = dict(list(state.items())[index::2])
        save_file(shard, checkpoint / name)
        weight_map.update({key: name for key in shard})
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    actual = load_model(str(checkpoint), device="cpu").model.state_dict()
    for key, value in state.items():
        torch.testing.assert_close(actual[key], value)
    weight_map["missing.weight"] = name
    index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    with pytest.raises(ValueError, match="index references missing"):
        load_model(str(checkpoint), device="cpu")


def test_export_dense_to_loop(checkpoint):
    base = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    base["model_type"] = "qwen3"
    base["tie_word_embeddings"] = True
    (checkpoint / "config.json").write_text(json.dumps(base), encoding="utf-8")
    expected = DenseToLoop(base)
    source = checkpoint / "last.pt"
    torch.save({"config": {"student": str(checkpoint), "loop": True}, "model": expected.state_dict()}, source)
    exported = checkpoint / "exported"
    export(source, exported)
    actual = load_model(str(exported), device="cpu").model
    assert actual.backbone.lm_head.weight is actual.backbone.model.embed_tokens.weight
    torch.testing.assert_close(actual.state_dict(), expected.state_dict())
