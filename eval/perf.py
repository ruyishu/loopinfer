"""Measure serving latency and throughput for LoopInfer or the HF baseline."""

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from common import BASE, MODEL, RESULTS, env, write_json

env()
PY = sys.executable


def gpu_memory():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return 0


def run_loopinfer(args):
    port = args.port
    log = (BASE / "logs" / "serve_perf.log").open("w")
    server = subprocess.Popen(
        [PY, "-m", "loopinfer", "serve", str(args.model), "--device", "cuda", "--loops", str(args.loops),
         "--port", str(port), "--context", str(args.input_tokens + args.output_tokens - 1), "--max-batch", str(max(args.concurrency)),
         "--kv-mib", "3072"],
        cwd=BASE, env=env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    peak = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_memory())
            stop.wait(1)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2) as response:
                    if json.load(response).get("ready"):
                        break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server did not become ready")
        peak[0] = max(peak[0], gpu_memory())

        rows = []
        for concurrency in args.concurrency:
            report = subprocess.run(
                [PY, "-m", "loopinfer", "bench", "--url", f"http://127.0.0.1:{port}",
                 "--input-tokens", str(args.input_tokens), "--output-tokens", str(args.output_tokens),
                 "--concurrency", str(concurrency), "--requests", str(concurrency * args.requests_per_worker),
                 "--warmup", "1"],
                cwd=BASE, env=env(), capture_output=True, text=True, timeout=1800,
            )
            data = json.loads(report.stdout)
            data["gpu_peak_mib"] = peak[0]
            rows.append(data)
            print(json.dumps(data), flush=True)
        return {"engine": "loopinfer", "runs": rows, "gpu_peak_mib": peak[0]}
    finally:
        stop.set()
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()
        log.close()


def run_hf(args):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    rows = []
    for batch in args.concurrency:
        prompt = torch.full((batch, args.input_tokens), 1, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            for _ in range(args.repeats):
                model(prompt, use_cache=True, exit_at_step=args.loops - 1)  # warmup
            prefill = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                model(prompt, use_cache=True, exit_at_step=args.loops - 1)
                torch.cuda.synchronize()
                prefill.append(time.perf_counter() - started)
            generate, generated = [], []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = model.generate(
                    prompt, max_new_tokens=args.output_tokens, do_sample=False,
                    pad_token_id=0, eos_token_id=None, use_cache=True, exit_at_step=args.loops - 1,
                )
                torch.cuda.synchronize()
                generate.append(time.perf_counter() - started)
                generated.append(batch * (result.shape[-1] - args.input_tokens))
        prefill.sort()
        generate.sort()
        median_generate = generate[len(generate) // 2]
        median_prefill = prefill[len(prefill) // 2]
        rows.append({
            "concurrency": batch,
            "prefill_p50_ms": round(median_prefill * 1000, 1),
            "generate_p50_ms": round(median_generate * 1000, 1),
            "inter_token_p50_ms": round((median_generate - median_prefill) / max(1, args.output_tokens - 1) * 1000, 1),
            "output_tokens_per_second": round(sum(generated) / sum(generate), 2),
            "gpu_peak_mib": gpu_memory(),
        })
        print(json.dumps(rows[-1]), flush=True)
    return {"engine": "hf", "runs": rows, "gpu_peak_mib": gpu_memory()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL), help="Local checkpoint or Hub model ID")
    parser.add_argument("--engine", choices=["loopinfer", "hf"], required=True)
    parser.add_argument("--input-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--requests-per-worker", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--out", type=Path, default=RESULTS / "perf")
    args = parser.parse_args()

    data = run_loopinfer(args) if args.engine == "loopinfer" else run_hf(args)
    data.update({
        "input_tokens": args.input_tokens, "output_tokens": args.output_tokens, "loops": args.loops,
    })
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / f"perf_{args.engine}.json", data)
    print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
