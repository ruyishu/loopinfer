import json
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tokenizers import Tokenizer

from .models import DenseToLoop, Ouro

BUILTINS = {key: model for model in (Ouro, DenseToLoop) for key in model.supported_types}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: Tokenizer
    name: str
    bos: int | None
    eos: frozenset[int]
    device: torch.device
    dtype: torch.dtype


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("The Torch backend supports CPU and CUDA")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but this PyTorch installation cannot access it")
    return resolved


def model_adapter(name):
    if not isinstance(name, str) or not name:
        raise ValueError("Checkpoint config must specify model_type")
    if name in BUILTINS:
        return BUILTINS[name]
    matches = list(entry_points(group="loopinfer.models", name=name))
    if len(matches) != 1:
        raise ValueError(f"Expected one adapter for model_type={name!r}; built-ins: {', '.join(BUILTINS)}")
    return matches[0].load()


def load_weights(model, folder: Path, device, dtype):
    weight_map = folder / "model.safetensors.index.json"
    if weight_map.is_file():
        index = json.loads(weight_map.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(index.values()))
    else:
        shards = ["model.safetensors"]
        index = None
    expected = dict(model.named_parameters(remove_duplicate=False))
    seen = set()
    loaded = {}
    for name in shards:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Invalid safetensors shard name {name!r}")
        with safe_open(folder / name, framework="pt", device="cpu") as file:
            for key in file.keys():
                if key not in expected or key in seen or (index is not None and index.get(key) != name):
                    raise ValueError(f"Unexpected, duplicate, or incorrectly indexed tensor {key!r}")
                source = file.get_tensor(key)
                target = expected[key]
                if source.shape != target.shape or not source.is_floating_point():
                    raise ValueError(f"Incompatible tensor {key}: got {source.shape} {source.dtype}, expected {target.shape}")
                alias = id(target)
                if alias in loaded:
                    if not torch.equal(source.to(device=device, dtype=dtype), loaded[alias]):
                        raise ValueError(f"Tied weights disagree for {key}")
                else:
                    loaded[alias] = torch.nn.Parameter(source.to(device=device, dtype=dtype), requires_grad=False)
                seen.add(key)
    for key, target in expected.items():
        if id(target) not in loaded:
            raise ValueError(f"Checkpoint is missing tensor {key}")
        parent, _, name = key.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, name, loaded[id(target)])
    if index is not None and seen != set(index):
        raise ValueError("Safetensors index references missing tensors")


def load_model(name: str, *, revision=None, loops=None, device="auto", dtype="auto") -> LoadedModel:
    folder = Path(name).expanduser()
    if not folder.is_dir():
        folder = Path(snapshot_download(
            name, revision=revision,
            allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "*.safetensors", "*.safetensors.index.json"],
        ))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    target = resolve_device(device)
    if dtype == "auto":
        dtype = "bfloat16" if target.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "float16" if target.type == "cuda" else "float32"
        )
    precision = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype)
    if precision is None:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    adapter = model_adapter(config.get("model_type"))
    with torch.device("meta"):
        model = adapter(config, loops)
    load_weights(model, folder, target, precision)
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    token_ids = tokenizer.get_vocab().values()
    if max(token_ids, default=-1) >= model.vocab_size:
        raise ValueError("Tokenizer contains IDs outside model vocabulary")
    if hasattr(model, "set_tokenizer"):
        model.set_tokenizer(tokenizer)
    generation_file = folder / "generation_config.json"
    generation = json.loads(generation_file.read_text(encoding="utf-8")) if generation_file.exists() else {}
    eos = generation.get("eos_token_id")
    if eos is None:
        eos = config.get("eos_token_id")
    bos = generation.get("bos_token_id")
    if bos is None:
        bos = config.get("bos_token_id")
    ends = frozenset(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    if any(not isinstance(token, int) or not 0 <= token < model.vocab_size for token in ends):
        raise ValueError("Invalid EOS token ID")
    if bos is not None and (not isinstance(bos, int) or not 0 <= bos < model.vocab_size):
        raise ValueError("Invalid BOS token ID")
    return LoadedModel(model.eval(), tokenizer, str(name), bos, ends, target, precision)
