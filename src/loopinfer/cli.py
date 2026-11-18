import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from . import __version__
from .bench import PROFILES, load_trace, make_trace, replay, write_trace
from .chat import DEFAULT_TEMPLATES
from .engine import Engine, Sampling
from .loading import BUILTINS, resolve_device


def model_options(parser):
    parser.add_argument("model")
    parser.add_argument("--revision", help="Pin the model to a Hugging Face revision")
    parser.add_argument("--loops", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-mib", type=float)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--cuda-graph", dest="cuda_graph", action="store_true", default="auto")
    parser.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    parser.add_argument("--graph-budget", type=int, default=32, help="Maximum captured CUDA graphs kept cached")
    parser.add_argument("--attention", choices=["auto", "sdpa", "triton", "flashinfer"], default="auto",
                        help="Attention backend; auto selects PyTorch SDPA")
    parser.add_argument("--scheduler", choices=["auto", "bucket", "round"], default="auto")
    parser.add_argument("--prefix-cache-pages", type=int, default=0)
    parser.add_argument("--graph-mib", type=float, help="Budget for estimated live graph allocations")
    parser.add_argument("--fuse-projections", action="store_true", help="Pack Ouro QKV and gate/up projections")
    parser.add_argument("--speculate-tokens", type=int, default=0, help="Single-row greedy Ouro verification width")
    parser.add_argument("--profile-gpu", action="store_true", help="Record CUDA events and NVTX ranges")


def parser():
    root = argparse.ArgumentParser(prog="loopinfer")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check available devices and model adapters")
    doctor.add_argument("--device", default="auto")
    generate = sub.add_parser("generate", help="Run offline generation")
    model_options(generate)
    generate.add_argument("--prompt", action="append", required=True)
    generate.add_argument("--max-tokens", type=int, default=32)
    generate.add_argument("--temperature", type=float, default=0)
    generate.add_argument("--top-p", type=float, default=1)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--ignore-eos", action="store_true")
    generate.add_argument("--loop-depth", type=int)
    generate.add_argument("--exit-threshold", type=float)
    serve = sub.add_parser("serve", help="Start an OpenAI-compatible completion server")
    model_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--timeout", type=float, default=120)
    serve.add_argument("--queue-size", type=int, default=128)
    serve.add_argument("--stream-buffer", type=int, default=64)
    serve.add_argument("--chat-template", type=Path, help="Jinja template file for /v1/chat/completions")
    serve.add_argument("--default-chat-template", choices=sorted(DEFAULT_TEMPLATES),
                       help="Built-in template for /v1/chat/completions when the checkpoint ships none")
    benchmark = sub.add_parser("bench", help="Run a concurrent streaming benchmark")
    benchmark.add_argument("--url", default="http://localhost:8000")
    benchmark.add_argument("--model")
    benchmark.add_argument("--profile", choices=sorted(PROFILES), default="short-decode")
    benchmark.add_argument("--trace-in", type=Path)
    benchmark.add_argument("--trace-out", type=Path)
    benchmark.add_argument("--events-out", type=Path)
    benchmark.add_argument("--seed", type=int, default=2026)
    benchmark.add_argument("--vocab-size", type=int, help="Generate distinct token IDs from [3, vocab-size)")
    benchmark.add_argument("--shared-prefix", type=int, default=0, help="Shared leading token count")
    benchmark.add_argument("--short-depth-fraction", type=float, default=0.5)
    benchmark.add_argument("--request-rate", type=float, help="Poisson arrivals per second")
    benchmark.add_argument("--ttft-slo-ms", type=float, default=1000)
    benchmark.add_argument("--tpot-slo-ms", type=float, default=50)
    benchmark.add_argument("--timeout", type=float, default=180)
    benchmark.add_argument("--protocol", choices=["loopinfer", "loopinfer-v03", "openai"], default="loopinfer")
    benchmark.add_argument("--input-tokens", type=int)
    benchmark.add_argument("--output-tokens", type=int)
    benchmark.add_argument("--concurrency", type=int, default=4)
    benchmark.add_argument("--requests", type=int, default=16)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--token-id", type=int, default=1)
    benchmark.add_argument("--output", type=Path, help="Write the JSON report to this path")
    return root


def main(argv=None):
    options = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if options.command == "doctor":
            device = resolve_device(options.device)
            gpu = {
                "name": torch.cuda.get_device_name(device), "capability": torch.cuda.get_device_capability(device),
                "free_bytes": torch.cuda.mem_get_info(device)[0], "total_bytes": torch.cuda.mem_get_info(device)[1],
            } if device.type == "cuda" else None
            from .paged import unavailable

            flash_reason = unavailable(device)
            print(json.dumps({
                "loopinfer": __version__, "platform": platform.platform(), "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "device": str(device), "gpu": gpu,
                "attention_backend": "torch_sdpa", "adapters": list(BUILTINS),
                "flashinfer": {"available": flash_reason is None, "reason": flash_reason},
                "ouro": {"fixed_depth": True, "mixed_depth": True, "gate_readout_full_kv": True,
                         "prefix_cache": True},
            }, indent=2))
            return 0
        if options.command == "bench":
            trace = load_trace(options.trace_in) if options.trace_in else make_trace(
                options.profile, options.requests, input_tokens=options.input_tokens,
                output_tokens=options.output_tokens, token_id=options.token_id, vocab_size=options.vocab_size,
                seed=options.seed, shared_prefix=options.shared_prefix, request_rate=options.request_rate,
                short_depth_fraction=options.short_depth_fraction,
            )
            if options.trace_out:
                write_trace(options.trace_out, trace)
            report, events = replay(
                options.url, options.model, trace, options.concurrency, warmup=options.warmup,
                timeout=options.timeout, protocol=options.protocol, ttft_slo=options.ttft_slo_ms,
                tpot_slo=options.tpot_slo_ms,
            )
            if options.events_out:
                write_trace(options.events_out, events)
            data = json.dumps(report, indent=2)
            if options.output:
                options.output.parent.mkdir(parents=True, exist_ok=True)
                options.output.write_text(data + "\n", encoding="utf-8")
            print(data)
            return int(bool(report["errors"]))
        engine_params = {
            key: getattr(options, key) for key in (
                "revision", "loops", "device", "dtype", "context", "page_size", "kv_mib", "max_batch",
                "prefill_chunk", "cuda_graph", "graph_budget", "attention", "scheduler", "prefix_cache_pages",
                "profile_gpu", "graph_mib", "fuse_projections", "speculate_tokens",
            )
        }
        if options.command == "serve":
            engine_params.update(queue_size=options.queue_size, stream_buffer=options.stream_buffer)
        with Engine(options.model, **engine_params) as engine:
            if options.command == "generate":
                sample = Sampling(
                    max_tokens=options.max_tokens, temperature=options.temperature, top_p=options.top_p,
                    top_k=options.top_k, seed=options.seed, ignore_eos=options.ignore_eos,
                    loop_depth=options.loop_depth, exit_threshold=options.exit_threshold,
                )
                print(json.dumps([asdict(row) for row in engine.generate(options.prompt, sample)], ensure_ascii=False))
            else:
                from .server import make_server
                template = options.chat_template.read_text(encoding="utf-8") if options.chat_template else None
                with make_server(
                    engine, options.host, options.port, timeout=options.timeout, chat_template=template,
                    default_chat_template=options.default_chat_template,
                ) as server:
                    logging.info("Serving %s at http://%s:%d", options.model, *server.server_address)
                    server.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"loopinfer: {exc}", file=sys.stderr)
        return 1
