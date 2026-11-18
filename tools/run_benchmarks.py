"""Replay matched workloads against isolated engine processes on a developer GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request

import torch

from loopinfer.bench import load_trace, make_trace, replay, write_trace


def command_output(command):
    try:
        return subprocess.run(command, text=True, capture_output=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def wait_ready(url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Engine stopped during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as response:
                if json.load(response).get("data"):
                    return
        except (OSError, ValueError):
            time.sleep(0.5)
    raise TimeoutError(f"Engine did not start at {url}")


def metrics(url, path):
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
            path.write_bytes(response.read())
    except (OSError, urllib.error.URLError):
        return


def validate_name(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError(f"Invalid case or engine name {value!r}")
    return value


def summarize(reports):
    grouped = {}
    for report in reports:
        key = (report["engine"], report["case"])
        grouped.setdefault(key, []).append(report)
    fields = ("ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "goodput_requests_per_second",
              "output_tokens_per_second")
    rows = []
    for (engine, case), runs in grouped.items():
        row = {"engine": engine, "case": case, "runs": len(runs),
               "failed_requests": sum(len(run["errors"]) for run in runs)}
        for field in fields:
            samples = [run[field] for run in runs if run[field] is not None]
            row[field] = {"median": statistics.median(samples), "runs": samples} if samples else None
        rows.append(row)
    return rows


def run_suite(config, output):
    output.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 2026)
    repetitions = config.get("repetitions", 3)
    engines = config["engines"]
    cases = config["cases"]
    if repetitions < 1 or not engines or not cases:
        raise ValueError("Configure at least one repetition, engine and case")
    for items in (engines, cases):
        names = [validate_name(item["name"]) for item in items]
        if len(set(names)) != len(names):
            raise ValueError("Engine and case names must be unique within their list")
    rng = random.Random(seed)
    traces = {}
    for case in cases:
        name = case["name"]
        traces[name] = load_trace(Path(case["trace"])) if "trace" in case else make_trace(
            case.get("profile", "short-decode"), case.get("requests", 100), seed=seed,
            **case.get("workload", {}),
        )
        write_trace(output / f"{name}.trace.jsonl", traces[name])
    save(output / "manifest.json", {
        "config": config, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "runner_revision": command_output(["git", "rev-parse", "HEAD"]),
        "gpus": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,power.limit",
                                "--format=csv,noheader"]),
        "trace_sha256": {name: hashlib.sha256((output / f"{name}.trace.jsonl").read_bytes()).hexdigest()
                         for name in traces},
    })
    reports, errors = [], []
    for repeat in range(repetitions):
        for engine in rng.sample(engines, len(engines)):
            selected = [case for case in cases if case["name"] in engine.get("cases", traces)]
            rng.shuffle(selected)
            for case in selected:
                name, label = case["name"], engine["name"]
                location = output / label / name / str(repeat + 1)
                location.mkdir(parents=True, exist_ok=True)
                root = engine["url"].rstrip("/")
                command = engine["command"]
                env = {**os.environ, **engine.get("env", {})}
                with (location / "server.log").open("wb") as log:
                    process = None
                    try:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   cwd=engine.get("cwd"), env=env)
                        wait_ready(root, process, engine.get("startup_timeout", 300))
                        metrics(root, location / "metrics-before.txt")
                        stats, events = replay(
                            root, engine.get("model"), traces[name], case.get("concurrency", 8),
                            protocol=engine.get("protocol", "openai"), warmup=case.get("warmup", 8),
                            ttft_slo=case.get("ttft_slo_ms", 1000), tpot_slo=case.get("tpot_slo_ms", 50),
                            timeout=case.get("timeout", 180),
                        )
                        stats.update(engine=label, case=name, repeat=repeat + 1)
                        save(location / "summary.json", stats)
                        write_trace(location / "events.jsonl", events)
                        metrics(root, location / "metrics-after.txt")
                        reports.append(stats)
                    except (RuntimeError, OSError, ValueError) as exc:
                        error = {"engine": label, "case": name, "repeat": repeat + 1, "error": str(exc)}
                        save(location / "error.json", error)
                        errors.append(error)
                    finally:
                        if process is not None and process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                print(f"{label}/{name}/{repeat + 1}", flush=True)
    save(output / "summary.json", {"results": summarize(reports), "errors": errors})
    return int(bool(errors or any(run["errors"] for run in reports)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="JSON benchmark suite configuration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_suite(json.loads(args.spec.read_text(encoding="utf-8")), args.output))
