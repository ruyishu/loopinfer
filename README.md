# LoopInfer

LoopInfer is a standalone inference engine for recurrent-depth language models. It owns its model execution, scheduler, KV pages, and serving API. The runtime uses PyTorch SDPA and has no dependency on vLLM, SGLang, or DeepInfer.

## Install and run

Python 3.10 or newer. Install a PyTorch build for your GPU and driver, then install LoopInfer from this repository.

~~~bash
git clone https://github.com/ruyishu/loopinfer.git
cd loopinfer
python -m pip install .
loopinfer doctor
loopinfer generate ByteDance/Ouro-1.4B --device cuda --loops 4 \
  --prompt "The capital of France is" --max-tokens 16
~~~

Install without a checkout using `pip install 'loopinfer @ git+https://github.com/ruyishu/loopinfer.git'`. The package does not bundle model weights. Use `--revision` to pin a Hub model or pass a local checkpoint directory. The native path loads safetensors and tokenizer files without importing model code from the Hub.

CPU is useful for smaller checkpoints and tests. Use `--device cpu --dtype float32 --kv-mib 64` with a short prompt if trying the real Ouro checkpoint without a GPU. Local smoke tests run on CPU using small generated checkpoints. Linux NVIDIA L20 and A800 are the target GPUs. Real-checkpoint accuracy, memory and performance on them require the validation runs below.

Try the installed CLI without downloading weights. This fixture uses random weights and is for runtime verification.

~~~bash
python examples/make_tiny_checkpoint.py ./tiny-ouro
loopinfer generate ./tiny-ouro --device cpu --prompt "the cat sat" --max-tokens 4
loopinfer serve ./tiny-ouro --device cpu --port 8000
~~~

## Serve

~~~bash
loopinfer serve ByteDance/Ouro-1.4B --device cuda --loops 4 --port 8000
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ByteDance/Ouro-1.4B","prompt":"A triangle has","max_tokens":24,"stream":true}'
~~~

The same endpoint works with the OpenAI Python client.
~~~python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
for chunk in client.completions.create(
    model="ByteDance/Ouro-1.4B", prompt="A triangle has", max_tokens=24, stream=True
):
    print(chunk.choices[0].text, end="", flush=True)
~~~

| Endpoint | Interface |
| --- | --- |
| `POST /v1/completions` | String prompt or token IDs, `n` choices, greedy or sampled, `stop`, `logprobs`, streaming SSE and non-streaming JSON |
| `POST /v1/chat/completions` | OpenAI chat messages rendered through the checkpoint's `chat_template` (or `--chat-template`/`--default-chat-template chatml`), with `logprobs` and `top_logprobs` |
| `GET /v1/models` | Model currently loaded by this instance |
| `GET /health`, `GET /ready` | HTTP liveness and inference-worker readiness |
| `GET /metrics` | Queue, completions, KV page occupancy, preemptions, CUDA graph counters and per-phase step timing |

Completion parameters include `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream`, `n`, `stop`, `logprobs` and `ignore_eos`. Chat adds `messages` and `top_logprobs`. Unsupported parameters return HTTP 400. Queue overload returns HTTP 503. The server binds loopback by default. Tool calls, TLS and request authentication are outside this server's interface.

`GET /metrics` reports `step_ms_prepare`, `step_ms_forward`, `step_ms_sample`, `step_ms_commit` and `step_ms_step` alongside the counters. `sample` includes the device-to-host read of the logits, so it absorbs the GPU time that a launch-only timer would miss.

## CUDA graphs

Decode steps are captured as CUDA graphs keyed by `(batch size, history bucket)`, so one replay runs every layer of every round with a single launch. This removes the per-token Python dispatch that dominates small-batch decoding: on Ouro-1.4B over an NVIDIA H20 the served decode path went from 13.3 to 48.4 output tokens/s at concurrency 1 (3.6x) with byte-identical greedy outputs. Graphs are enabled by default on CUDA; pass `--no-cuda-graph` or `Engine(..., cuda_graph=False)` to measure the eager path, and read `cuda_graph_captures` / `cuda_graph_replays` from `stats()`. Capture happens lazily on first use, so the first request for a new shape pays a short one-off cost. `--graph-budget` bounds how many captured graphs stay cached; eviction is FIFO and counts in `cuda_graph_evictions`.

A step is split into a prefill sub-batch and a decode sub-batch. Previously any request starting prefill sent the whole batch down the eager path, so every decoder in the batch lost its graph replay for that step.

For offline batched use, all inputs share a scheduler.

~~~python
from loopinfer import Engine, Sampling

with Engine("ByteDance/Ouro-1.4B", device="cuda", loops=4) as engine:
    results = engine.generate(["A triangle has", "Shanghai is in"], Sampling(max_tokens=24))
    for result in results:
        print(result.text, result.token_ids, result.first_token_seconds)
~~~

Use `engine.submit(token_ids, sampling, stream=True)` to receive token events, await the final result, or cancel. Seeded sampling uses independent per-request RNG. Different kernel and batch shapes can still change floating-point calculations.

## Models and KV

| Model type | Loops | KV layout and output |
| --- | --- | --- |
| `ouro` | Fixed, up to checkpoint depth | One cache domain per executed round. Normalize after every round and read out the final executed round. Ouro early exit and gate-weighted output require a separate execution path |
| `dense-to-loop` | Fixed, one or two | One persistent round-one domain. On round two, each query reads causal round-one history and its own current round-two KV. Average the two masked softmax distributions before sampling |

The Ouro profile is for full attention and default RoPE. The native loader validates tensor shapes and tied aliases. Dense-to-loop accepts the original full training checkpoint after export, using its matching Qwen3 configuration and tokenizer.

~~~bash
python tools/export_dense_to_loop.py /path/to/run/last.pt ./loop-checkpoint
loopinfer serve ./loop-checkpoint --device cuda --loops 2
~~~

The scheduler packs prefill and decode into separate sub-batches of the same step. Pages are handed out as a request's cached length grows; when the pool runs dry the newest restarted-friendly request is preempted, put back at the head of the queue and re-run from its prompt, and admission stays closed until some request finishes so the engine cannot thrash. A single worker owns device execution and page reclamation. Attention gathers referenced KV pages into a batched SDPA call by default; `--attention triton` selects the experimental paged-attention kernel, which currently matches SDPA numerically but runs slower, so `auto` keeps the reference path. For Ouro-1.4B BF16 with four loops, the raw KV cost is about 0.75 MiB per cached input position. `--kv-mib`, `--context`, `--max-batch` and `--prefill-chunk` tune the capacity and transient workspace.

Per-step page tables and the causal mask are built on the device from a padded table, so only per-row scalars cross PCIe. Building them on the host costs O(batch x history) Python work per step and dominates at long contexts.

## Verify and benchmark

~~~bash
python -m pip install '.[test]'
python -m pytest -q
loopinfer bench --url http://127.0.0.1:8000 --input-tokens 512 \
  --output-tokens 128 --concurrency 8 --requests 96 --output l20.json
~~~

The synthetic benchmark forces a fixed output length and reports successful throughput, P50/P95 client-observed time to first token and inter-token delay, plus failures. `eval/inprocess.py` replays the same request mix through `Engine.submit` without HTTP, so the difference between the two answers what the serving layer costs; on H20 at 32 input and 32 output tokens that difference is 0% at concurrency 1 and 6-8% at concurrency 4-8, and each step reports its prepare/forward/sample/commit split. Keep checkpoint revision, loop count, readout, dtype, hardware, initial context length and output budget fixed across engines.

Use the official Ouro forward implementation as an independent real-checkpoint reference in a separate environment or with the reference extra. This verifier loads reviewed official model code, forces the same reference token at each step, and reports logprob differences and greedy token agreement.

~~~bash
python -m pip install '.[reference]'
python tools/verify_ouro.py --model ByteDance/Ouro-1.4B \
  --revision <reviewed-model-commit> --device cuda --loops 4 --steps 8
~~~

## Extending the engine

The scheduler calls an adapter with token IDs, a batch of positions and page-table indices, and a page pool. An adapter provides `vocab_size`, `max_positions`, and `CacheSpec`, and returns one row of logits or log probabilities per emitting sequence. Adapter factories take `(config, loops)`. Weights are loaded from safetensors by parameter name. Register an external factory with the `loopinfer.models` Python package entry-point group and set the same key as `model_type` in the checkpoint. See the built-in Ouro and dense-to-loop adapters for two concrete layouts.

The current scheduler uses continuous batching at the token boundary. DeepInfer's CPU/device overlap and validation methodology inform the design. The runtime keeps a synchronous execution path as the baseline for future loop-level scheduling, direct paged attention kernels and CUDA Graph experiments.

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- [Ouro checkpoint and reference implementation](https://huggingface.co/ByteDance/Ouro-1.4B)
- [Continuous Depth Batching](https://arxiv.org/abs/2608.09444)
