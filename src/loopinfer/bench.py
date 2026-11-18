"""Deterministic workload generation and client-observed SSE benchmarks."""

import concurrent.futures
import json
import math
import platform
import random
import time
import urllib.error
import urllib.request

PROFILES = {
    "short-decode": ((128, 256),),
    "long-prefill": ((2048, 1),),
    "long-decode": ((2048, 128),),
    "mixed": ((128, 128), (2048, 64)),
    "mixed-depth": ((128, 128),),
}


def percentile(values, p):
    if not values:
        return None
    return sorted(values)[round((len(values) - 1) * p)] * 1000


def make_trace(profile, requests, *, input_tokens=None, output_tokens=None, token_id=1,
               vocab_size=None, seed=2026, shared_prefix=0, request_rate=None, short_depth_fraction=0.5):
    if requests < 1 or min(token_id, shared_prefix) < 0 or (vocab_size is not None and vocab_size <= 3):
        raise ValueError("Invalid trace generation limits")
    if request_rate is not None and (not math.isfinite(request_rate) or request_rate <= 0):
        raise ValueError("Request rate must be positive and finite")
    if not 0 <= short_depth_fraction <= 1:
        raise ValueError("Short-depth fraction must be in [0, 1]")
    arrivals = random.Random(seed)
    prompts = random.Random(seed + 1)
    mix = random.Random(seed + 2)
    cases = PROFILES[profile]
    prefix = prompts.choices(range(3, vocab_size), k=shared_prefix) if vocab_size is not None else []
    if shared_prefix and not prefix:
        raise ValueError("Shared-prefix workloads require --vocab-size")
    arrival = 0.0
    trace = []
    for index in range(requests):
        category = ("long" if mix.random() < 0.3 else "short") if len(cases) == 2 else profile
        default_in, default_out = cases[-1 if category == "long" else 0]
        length = default_in if input_tokens is None else input_tokens
        output = default_out if output_tokens is None else output_tokens
        if min(length, output) < 1 or len(prefix) >= length:
            raise ValueError("Token lengths must be positive and leave at least one non-shared token")
        prompt = (prefix + prompts.choices(range(3, vocab_size), k=length - len(prefix))
                  if vocab_size is not None else [token_id] * length)
        row = {"id": index, "prompt": prompt, "output": output, "category": category}
        if profile == "mixed-depth":
            row["loop_depth"] = 2 if mix.random() < short_depth_fraction else 4
            row["category"] = f"depth-{row['loop_depth']}"
        if request_rate is not None:
            if index:
                arrival += arrivals.expovariate(request_rate)
            row["arrival_s"] = arrival
        trace.append(row)
    return trace


def write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_trace(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _measure(response, protocol):
    stamps, counts = [], []
    reason, usage, complete = None, None, False
    for line in response:
        if not line.startswith(b"data:"):
            continue
        stamp = time.perf_counter()
        value = line[5:].strip()
        if value == b"[DONE]":
            complete = True
            break
        event = json.loads(value)
        if "error" in event:
            raise RuntimeError(event["error"]["message"])
        if event.get("usage"):
            usage = event["usage"]["completion_tokens"]
        for choice in event.get("choices", []):
            ids = event.get("token_ids") or choice.get("token_ids")
            legacy_token = protocol == "loopinfer-v03" and reason is None
            if ids or legacy_token or (protocol == "openai" and choice.get("text")):
                stamps.append(stamp)
                counts.append(len(ids) if ids else (1 if legacy_token else None))
            reason = choice.get("finish_reason") or reason
    if not complete:
        raise RuntimeError("Stream closed without [DONE]")
    if all(count is not None for count in counts) and stamps:
        usage = sum(counts)
    if usage is None:
        raise RuntimeError("Server did not supply completion token counts")
    if not stamps:
        raise RuntimeError("Stream produced no timed token or text events")
    return stamps, counts, usage, reason


def replay(url, model, trace, concurrency, *, warmup=2, timeout=180, protocol="loopinfer",
           ttft_slo=1000, tpot_slo=50):
    if not trace or concurrency < 1 or warmup < 0 or min(timeout, ttft_slo, tpot_slo) <= 0:
        raise ValueError("Benchmark limits must be positive")
    if protocol not in {"loopinfer", "loopinfer-v03", "openai"}:
        raise ValueError("Unknown SSE protocol")
    open_loop = "arrival_s" in trace[0]
    arrival = 0.0
    for row in trace:
        if type(row.get("output")) is not int or row["output"] < 1:
            raise ValueError("Each trace row requires a positive output limit")
        if "arrival_s" in row:
            offset = row["arrival_s"]
            if not math.isfinite(offset) or offset < arrival:
                raise ValueError("Arrival offsets must be finite and nondecreasing")
            arrival = offset
        if ("arrival_s" in row) != open_loop:
            raise ValueError("The trace must specify arrival offsets for all or no requests")
    root = url.rstrip("/")
    if model is None:
        with urllib.request.urlopen(root + "/v1/models", timeout=timeout) as response:
            model = json.load(response)["data"][0]["id"]

    def request(row, scheduled=None):
        json_body = {
            "model": model, "prompt": row["prompt"], "max_tokens": row["output"],
            "temperature": 0, "ignore_eos": row.get("ignore_eos", True), "stream": True,
        }
        if row.get("loop_depth") is not None:
            json_body["loop_depth"] = row["loop_depth"]
        if protocol == "openai":
            json_body["stream_options"] = {"include_usage": True}
        api = urllib.request.Request(
            root + "/v1/completions", json.dumps(json_body).encode(),
            {"Content-Type": "application/json", "X-LoopInfer-Token-Ids": "1"} if protocol == "loopinfer"
            else {"Content-Type": "application/json"},
        )
        sent = time.perf_counter()
        scheduled = sent if scheduled is None else scheduled
        event = {"id": row.get("id"), "category": row.get("category", "all"),
                 "client_queue_s": max(0.0, sent - scheduled)}
        try:
            with urllib.request.urlopen(api, timeout=timeout) as response:
                stamps, counts, tokens, reason = _measure(response, protocol)
            if (json_body["ignore_eos"] and (tokens != row["output"] or reason != "length")
                    or not 1 <= tokens <= row["output"]):
                raise RuntimeError(f"Unexpected stream length {tokens} or finish reason {reason!r}")
            event.update(ttft_s=stamps[0] - scheduled, end_to_end_s=stamps[-1] - scheduled,
                         tpot_s=(stamps[-1] - stamps[0]) / (tokens - 1) if tokens > 1 else None,
                         events_s=[stamp - scheduled for stamp in stamps], event_tokens=counts,
                         output_tokens=tokens, finish_reason=reason)
        except Exception as exc:
            event["error"] = str(exc)
            event["status"] = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        return event

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        # Reorder the prompt tokens during warmup so cold-prefix traces do not become cache-hit traces.
        def warm(index):
            row = dict(trace[index % len(trace)])
            if isinstance(row["prompt"], list) and len(row["prompt"]) > 1:
                first = row["prompt"][0]
                other = next((token for token in row["prompt"] if token != first), None)
                if other is not None:
                    row["prompt"] = [other, *row["prompt"][1:]]
            return request(row)

        warm_results = list(pool.map(warm, range(warmup)))
        started = time.perf_counter()
        futures = []
        for row in trace:
            scheduled = started + row["arrival_s"] if open_loop else None
            if scheduled is not None:
                time.sleep(max(0, scheduled - time.perf_counter()))
            futures.append(pool.submit(request, row, scheduled))
        events = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
    report = summarize(events, elapsed, ttft_slo, tpot_slo)
    report.update(
        model=model, concurrency=concurrency, warmup_requests=warmup,
        warmup_errors=[row["error"] for row in warm_results if "error" in row],
        arrival_model="open" if open_loop else "closed", protocol=protocol,
        ttft_slo_ms=ttft_slo, tpot_slo_ms=tpot_slo, client_platform=platform.platform(),
    )
    report["categories"] = {
        category: summarize([e for e in events if e["category"] == category], elapsed, ttft_slo, tpot_slo)
        for category in sorted({e["category"] for e in events})
    }
    return report, events


def summarize(events, elapsed, ttft_slo, tpot_slo):
    rows = [event for event in events if "error" not in event]
    ttft = [row["ttft_s"] for row in rows]
    tpot = [row["tpot_s"] for row in rows if row["tpot_s"] is not None]
    itl = [b - a for row in rows for a, b in zip(row["events_s"], row["events_s"][1:])]
    good = sum(row["ttft_s"] * 1000 <= ttft_slo and (
        row["tpot_s"] is None or row["tpot_s"] * 1000 <= tpot_slo
    ) for row in rows)
    metrics = {
        "ttft": ttft, "tpot": tpot, "inter_token": itl,
        "request": [row["end_to_end_s"] for row in rows],
        "client_queue": [row["client_queue_s"] for row in events],
    }
    return {
        "requests": len(events), "completed": len(rows), "errors": [e["error"] for e in events if "error" in e],
        "rejected": sum(row.get("status") in {429, 503} for row in events),
        "elapsed_seconds": elapsed, "goodput_requests_per_second": good / elapsed,
        "output_tokens_per_second": sum(row["output_tokens"] for row in rows) / elapsed,
        "requests_per_second": len(rows) / elapsed,
        **{f"{name}_p{p}_ms": percentile(values, p / 100)
           for name, values in metrics.items() for p in (50, 95)},
    }


def run(url, model, input_tokens, output_tokens, concurrency, requests, token_id=1, warmup=2, timeout=180):
    trace = make_trace("short-decode", requests, input_tokens=input_tokens, output_tokens=output_tokens,
                       token_id=token_id)
    report, _ = replay(url, model, trace, concurrency, warmup=warmup, timeout=timeout)
    return {**report, "input_tokens": input_tokens, "output_tokens": output_tokens}
