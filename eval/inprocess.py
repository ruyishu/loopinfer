"""Measure LoopInfer throughput in-process, without the HTTP layer.

Same workload shape as ``perf.py --engine loopinfer`` (concurrent clients issuing
back-to-back requests), so the two numbers answer "what does the server add?".
"""

import argparse
import json
import threading
import time
from pathlib import Path

from common import MODEL, RESULTS, env, write_json

env()


def counter(stats, name):
    return float(stats[f"step_ms_{name}"]) * stats["steps"]


def one_client(engine, prompt, sample, count, tokens, latencies, ttfts):
    for _ in range(count):
        started = time.perf_counter()
        result = engine.submit(prompt, sample).wait()
        latencies.append(time.perf_counter() - started)
        tokens.append(len(result.token_ids))
        ttfts.append(result.first_token_seconds)


def percentile(values, fraction):
    return round(sorted(values)[round((len(values) - 1) * fraction)] * 1000, 1) if values else None


def run(args):
    from loopinfer import Engine, Sampling

    prompt = [1] * args.input_tokens
    sample = Sampling(max_tokens=args.output_tokens, temperature=0.0, ignore_eos=True)
    rows = []
    with Engine(
        str(MODEL), device="cuda", loops=args.loops, context=args.context,
        max_batch=max(args.concurrency), kv_mib=args.kv_mib,
    ) as engine:
        for concurrency in args.concurrency:
            for _ in range(args.warmup):
                engine.submit(prompt, sample).wait()
            before = engine.stats()
            tokens, latencies, ttfts = [], [], []
            per_client = args.per_client
            clients = [
                threading.Thread(target=one_client, args=(engine, prompt, sample, per_client, tokens, latencies,
                                                          ttfts))
                for _ in range(concurrency)
            ]
            started = time.perf_counter()
            for client in clients:
                client.start()
            for client in clients:
                client.join()
            elapsed = time.perf_counter() - started
            after = engine.stats()
            steps = max(1, after["steps"] - before["steps"])
            phases = {
                name: round((counter(after, name) - counter(before, name)) / steps, 3)
                for name in ("prepare", "forward", "sample", "commit", "step")
            }
            rows.append({
                "concurrency": concurrency,
                "requests": len(tokens),
                "output_tokens": sum(tokens),
                "elapsed_seconds": round(elapsed, 3),
                "output_tokens_per_second": round(sum(tokens) / elapsed, 2),
                "requests_per_second": round(len(tokens) / elapsed, 2),
                "request_p50_ms": percentile(latencies, 0.50),
                "request_p95_ms": percentile(latencies, 0.95),
                "ttft_p50_ms": percentile(ttfts, 0.50),
                "inter_token_p50_ms": round(
                    sum(latency - ttft for latency, ttft in zip(latencies, ttfts))
                    / max(1, sum(tokens) - len(tokens)) * 1000, 2
                ),
                "engine": {**phases, "preemptions": after["preemptions"] - before["preemptions"],
                           "graph_captures": after["cuda_graph_captures"], "graph_replays": after["cuda_graph_replays"]},
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return {"engine": "loopinfer-inprocess", "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--per-client", type=int, default=2, help="Requests each client issues, as in perf.py")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--kv-mib", type=float, default=3072)
    parser.add_argument("--out", type=Path, default=RESULTS / "perf")
    args = parser.parse_args()
    data = run(args)
    data.update({"input_tokens": args.input_tokens, "output_tokens": args.output_tokens, "loops": args.loops})
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "perf_inprocess.json", data)
    print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
