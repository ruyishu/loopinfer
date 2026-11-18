"""Replay a fixed-size concurrent streaming workload and measure client-side latency."""

import concurrent.futures
import json
import platform
import time
import urllib.request


def percentile(values, p):
    if not values:
        return None
    return sorted(values)[round((len(values) - 1) * p)] * 1000


def run(url, model, input_tokens, output_tokens, concurrency, requests, token_id=1, warmup=2, timeout=180):
    if min(input_tokens, output_tokens, concurrency, requests) < 1 or min(warmup, token_id) < 0:
        raise ValueError("Benchmark lengths, concurrency and requests must be positive")
    root = url.rstrip("/")
    if model is None:
        with urllib.request.urlopen(root + "/v1/models", timeout=timeout) as response:
            model = json.load(response)["data"][0]["id"]
    payload = json.dumps({
        "model": model, "prompt": [token_id] * input_tokens, "max_tokens": output_tokens,
        "temperature": 0, "ignore_eos": True, "stream": True,
    }).encode()

    def request(_):
        api = urllib.request.Request(root + "/v1/completions", payload, {"Content-Type": "application/json"})
        started = time.perf_counter()
        tokens = []
        reason = None
        with urllib.request.urlopen(api, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                value = line[6:].strip()
                if value == b"[DONE]":
                    break
                event = json.loads(value)
                if "error" in event:
                    raise RuntimeError(event["error"]["message"])
                choice = event["choices"][0]
                tokens.append(time.perf_counter())
                reason = choice["finish_reason"]
        if len(tokens) != output_tokens or reason != "length":
            raise RuntimeError(f"Expected {output_tokens} tokens and length stop, got {len(tokens)}, {reason!r}")
        return tokens[0] - started, [b - a for a, b in zip(tokens, tokens[1:])], tokens[-1] - started

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(request, range(warmup)))
        started = time.perf_counter()
        futures = [pool.submit(request, i) for i in range(requests)]
        rows, errors = [], []
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(str(exc))
        elapsed = time.perf_counter() - started
    ttft = [row[0] for row in rows]
    itl = [interval for row in rows for interval in row[1]]
    report = {
        "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "concurrency": concurrency, "requests": requests, "warmup_requests": warmup,
        "completed": len(rows), "errors": errors, "elapsed_seconds": elapsed,
        "output_tokens_per_second": len(rows) * output_tokens / elapsed,
        "requests_per_second": len(rows) / elapsed,
        "ttft_p50_ms": percentile(ttft, .50), "ttft_p95_ms": percentile(ttft, .95),
        "inter_token_p50_ms": percentile(itl, .50), "inter_token_p95_ms": percentile(itl, .95),
        "request_p95_ms": percentile([r[2] for r in rows], .95),
        "client_platform": platform.platform(),
    }
    return report
