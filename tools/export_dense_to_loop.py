"""Convert a dense-to-loop last.pt into a weights-only LoopInfer directory."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from tokenizers import Tokenizer


def export(checkpoint: Path, output: Path, base_override: str | None = None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    meta = payload["config"]
    if not meta.get("loop") or "model" not in payload:
        raise ValueError("Expected a trained or zero-update dense-to-loop last.pt containing full model weights")
    base = Path(base_override or meta["student"]).expanduser()
    if not base.is_dir():
        base = Path(snapshot_download(
            base_override or meta["student"], revision=meta.get("student_revision"),
            allow_patterns=["config.json", "tokenizer.json", "generation_config.json"],
        ))
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("Base config must come from Qwen3")
    config["model_type"] = "dense-to-loop"
    config.pop("auto_map", None)
    config.pop("architectures", None)
    state = dict(payload["model"])
    if config.get("tie_word_embeddings"):
        key = "backbone.lm_head.weight"
        other = "backbone.model.embed_tokens.weight"
        if key in state and other in state:
            if not torch.equal(state[key], state[other]):
                raise ValueError("Expected tied embedding and LM head weights")
            del state[key]
    tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
    generation = base / "generation_config.json"
    generation_config = json.loads(generation.read_text(encoding="utf-8")) if generation.is_file() else {}
    eos = generation_config.get("eos_token_id", config.get("eos_token_id"))
    eos_ids = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    eos_ids.update(
        token_id for name in ("<|im_end|>", "<|endoftext|>")
        if (token_id := tokenizer.token_to_id(name)) is not None
    )
    generation_config["eos_token_id"] = sorted(eos_ids)

    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (output / "generation_config.json").write_text(json.dumps(generation_config) + "\n", encoding="utf-8")
        shutil.copyfile(base / "tokenizer.json", output / "tokenizer.json")
        save_file({name: tensor.contiguous() for name, tensor in state.items()}, output / "model.safetensors")
    except Exception:
        for filename in ("config.json", "generation_config.json", "tokenizer.json", "model.safetensors"):
            (output / filename).unlink(missing_ok=True)
        output.rmdir()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", help="Matching local Qwen3 checkpoint directory or Hub model ID")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.base)
