"""Compare native paged execution with the official Transformers Ouro reference."""

import argparse
import json
from pathlib import Path

import torch

from loopinfer import load_model
from loopinfer.cache import PagePool

PROMPTS = (
    "The capital of France is", "Compute 19 + 27.", "A short Python function:",
    "Today the weather", "A triangle has", "The first prime number is",
    "She picked up the book and", "翻译：我今天很高兴。",
)


@torch.inference_mode()
def verify(args):
    from transformers import AutoModelForCausalLM

    native = load_model(args.model, revision=args.revision, loops=args.loops, device=args.device, dtype=args.dtype)
    official = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, torch_dtype=native.dtype,
        attn_implementation="sdpa",
    ).to(native.device).eval()
    differences, selected_errors, hidden_errors, matches = [], [], [], []
    for prompt in PROMPTS:
        ids = native.tokenizer.encode(prompt).ids
        if not ids:
            continue
        table, past = [], None
        pool = PagePool(native.model.cache_spec, (len(ids) + args.steps + 15) // 16, 16, native.device, native.dtype)
        # Forward hooks observe round states without changing official logits or cache.
        reference_hidden = []
        hook = official.model.norm.register_forward_hook(lambda _m, _i, h: reference_hidden.append(h[:, -1].float()))
        try:
            offset = 0
            chunk = torch.tensor([ids], device=native.device)
            for _ in range(args.steps):
                reference_hidden.clear()
                out = official(
                    chunk, past_key_values=past, use_cache=True, exit_at_step=args.loops - 1,
                )
                past = out.past_key_values
                length = chunk.size(1)
                pool.extend(0, table, offset + length)
                batch = pool.prepare([table], [offset], [length], [0])
                states = []
                got = native.model(chunk, batch, pool, trace=states)
                reference = out.logits[:, -1].float()
                delta = (reference.log_softmax(-1) - got.log_softmax(-1)).abs()
                expected_token = reference.argmax(-1)
                selected_errors.append(delta.gather(-1, expected_token[:, None]).item())
                differences.append(delta.max().item())
                matches.append((expected_token == got.argmax(-1)).item())
                hidden_errors.append(max(
                    (lhs.float() - rhs).abs().max().item() for lhs, rhs in zip(states, reference_hidden)
                ))
                offset += length
                # Force the same reference token so later comparisons survive a divergence.
                chunk = expected_token[:, None]
        finally:
            hook.remove()
    return {
        "model": args.model, "revision": args.revision, "device": str(native.device),
        "dtype": str(native.dtype), "loops": args.loops, "compared_positions": len(matches),
        "argmax_matches": sum(matches), "max_all_logprobs_error": max(differences),
        "max_selected_token_logprob_error": max(selected_errors), "max_round_hidden_error": max(hidden_errors),
        "torch": torch.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B")
    parser.add_argument("--revision", required=True, help="A reviewed official model commit for the remote-code reference")
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-selected-error", type=float, help="Optional regression threshold")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    report = verify(args)
    content = json.dumps(report, indent=2)
    print(content)
    if args.output:
        args.output.write_text(content + "\n", encoding="utf-8")
    raise SystemExit(int(
        args.max_selected_error is not None and report["max_selected_token_logprob_error"] > args.max_selected_error
    ))
