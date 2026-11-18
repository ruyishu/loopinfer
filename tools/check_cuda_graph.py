"""Compare greedy generation with and without the CUDA graph decode path."""

import argparse
import json

import torch

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


def generate(model, prompts, graph, args):
    with Engine(
        model, device=args.device, loops=args.loops, context=args.context, dtype=args.dtype,
        max_batch=len(prompts), kv_mib=args.kv_mib, cuda_graph=graph,
    ) as engine:
        results = engine.generate(prompts, Sampling(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True))
        return [list(result.token_ids) for result in results], engine.stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--loops", type=int, default=None)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--kv-mib", type=float, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    eager, eager_stats = generate(args.model, PROMPTS, False, args)
    graphed, graph_stats = generate(args.model, PROMPTS, True, args)
    matches = sum(left == right for left, right in zip(eager, graphed))
    report = {
        "prompts": len(PROMPTS), "identical": matches,
        "cuda_graph_captures": graph_stats["cuda_graph_captures"],
        "cuda_graph_replays": graph_stats["cuda_graph_replays"],
        "cuda_graph_active": graph_stats["cuda_graph"],
        "eager_stats": eager_stats, "graph_stats": graph_stats,
        "mismatched": [index for index, (left, right) in enumerate(zip(eager, graphed)) if left != right],
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if matches == len(PROMPTS) and report["cuda_graph_active"] else 1)


if __name__ == "__main__":
    main()
