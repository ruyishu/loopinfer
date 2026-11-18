# CUDA validation and benchmark guide

Use a Linux Python environment with a PyTorch CUDA build that matches the target driver. Pin the reviewed Ouro checkpoint revision for published comparisons. Default CPU and CUDA serving uses SDPA. Runtime dependencies exclude competitor serving engines.

~~~bash
python -m pip install '.[test,reference]'
loopinfer doctor --device cuda
python -m pytest -q
python -m pip wheel --no-deps . --wheel-dir dist
~~~

Install the FlashInfer extra in a CUDA environment with toolchain support for JIT kernels before selecting that backend.

~~~bash
python -m pip install '.[flashinfer]'
python tools/verify_ouro.py --revision <commit> --device cuda --dtype float32 --steps 8 \
  --output benchmark-results/parity/fp32.json
python tools/verify_ouro.py --revision <commit> --device cuda --dtype bfloat16 --steps 8 \
  --attention flashinfer --output benchmark-results/parity/flashinfer.json
~~~

The parity tool teacher-forces reference tokens and records every round's hidden-state error, chosen-token error, top-1 margin at argmax divergences, and the compared positions. Run the BF16 reference with SDPA before comparing FlashInfer or packed projections. Pass `--fuse-projections` to test packed linear layers or `--exit-threshold` to check the full-KV gate path. The available gate readout is not compute-skipping early exit. Run the existing `eval/accuracy.py` GSM8K test on enough questions for a paired quality assessment. Use the same task prompts, policy and checkpoint for both engines.

## Same-trace online benchmarks

Copy `eval/benchmark_suite.json` and edit GPU allocation, revision flags, output rate and optional case selection. The checked-in suite starts separate server processes sequentially on one GPU and randomizes the order across independent repetitions. It includes short decode, long prefill, long-context decode, mixed arrivals, three fixed-depth blends, and shared KV prefixes. Each report folder stores client event JSONL, metrics snapshots, server log, suite manifest, trace SHA256 and repeat-level summaries.

~~~bash
CUDA_VISIBLE_DEVICES=0 python tools/run_benchmarks.py eval/benchmark_suite.json \
  --output benchmark-results/l20
~~~

Use `--profile-gpu` in a separate profiling pass. Read the `gpu_ms_prefill`, `gpu_ms_decode`, `gpu_ms_round` and `gpu_ms_speculate` event metrics alongside the existing host phase times. Use Nsight Systems for the matching NVTX ranges and Nsight Compute if HBM traffic or individual kernels need inspection. Run the formal speed comparison without profiling. Record cold captures, rejected requests and effective backend selection from metrics and server logs.

For the v0.3 baseline, launch its CLI in an isolated checkout. It exports the legacy text SSE stream and takes fixed depth at server startup. Set this engine's protocol to `loopinfer-v03` to timestamp legacy token events through the first terminal event. Compare it on fixed-depth cases. For example, install the separate worktree into its own virtual environment and point the suite's command to that environment's absolute `loopinfer` path. vLLM-RLT and Loop-SGLang can be included as independent commands with the suite's `openai` event protocol when they supply completion counts via streaming usage. Confirm the checkpoint, token IDs, KV semantics, readout, dtype, resource cap and candidate's modes before ranking a scenario. A competitor that cannot accept the trace's fixed per-request depth field needs a workload adapter using its documented API. Record that adapter and any semantic difference in the benchmark manifest.

To compare common OpenAI text SSE, use `--protocol openai` for both ends. Native `--protocol loopinfer` timestamps token IDs even when the decoded text is empty. OpenAI protocol timestamps text chunks if an engine doesn't expose token IDs. For multi-token SSE frames, ITL measures frame gaps and TPOT divides the first-to-last interval by generated tokens minus one. Goodput counts successful requests that meet TTFT and TPOT SLOs. Open-loop TTFT includes client queueing if the dispatch pool saturates. Use enough client workers to expose server queuing and check the report's client queue metric.

## Release checks

Build a GPU image with `docker build -t loopinfer:0.4.0 .`. The image installs the pinned PyTorch wheel and the SDPA backend. Override `TORCH_INDEX` and `TORCH_VERSION` to match your device and driver. A CUDA-enabled host with the NVIDIA container runtime can launch the server directly.

~~~bash
docker run --gpus '"device=0"' --rm -p 8000:8000 -v /models:/models:ro loopinfer:0.4.0 \
  serve /models/Ouro-1.4B --device cuda --host 0.0.0.0 --context 2304
~~~

Run the same suite independently on L20 and a second device. For overnight load tests, use a trace with arrivals spanning the test period and repeat cold and warm workloads. Inspect page high-water marks, replays, failures, and stable RSS and GPU memory at rest. The package exposes `Engine.close(drain=True, timeout=30)` for embedding services that need to quiesce workers before shutdown. Promote opt-in CUDA settings once matched parity, speed and stability results are available.
