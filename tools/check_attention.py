"""Compare paged Triton decode attention against the gather + SDPA path."""

import argparse
import json

from loopinfer import Engine, Sampling

PROMPTS = (
    "The capital of France is",
    "Compute 19 + 27.",
    "A short Python function:",
    "Today the weather",
    "A triangle has",
    "The first prime number is",
    "She picked up the book and",
    "翻译：我今天很高兴。",
)


def generate(args, attention):
    with Engine(
        args.model, device=args.device, loops=args.loops, context=args.context, dtype=args.dtype,
        max_batch=len(PROMPTS), kv_mib=args.kv_mib, cuda_graph=args.cuda_graph, attention=attention,
    ) as engine:
        results = engine.generate(PROMPTS, Sampling(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True))
        return [list(result.token_ids) for result in results], engine.stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--loops", type=int, default=None)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--kv-mib", type=float, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cuda-graph", dest="cuda_graph", action="store_true", default="auto")
    parser.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    args = parser.parse_args()

    reference, _ = generate(args, "sdpa")
    candidate, stats = generate(args, "triton")
    matches = sum(left == right for left, right in zip(reference, candidate))
    report = {
        "prompts": len(PROMPTS), "identical": matches,
        "kernel_active": stats["attention_kernel"],
        "compared_tokens": sum(len(row) for row in reference),
        "mismatched": [index for index, (left, right) in enumerate(zip(reference, candidate)) if left != right],
        "step_ms": {key: stats[f"step_ms_{key}"] for key in ("prepare", "forward", "sample", "step")},
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if matches == len(PROMPTS) and report["kernel_active"] else 1)


if __name__ == "__main__":
    main()
