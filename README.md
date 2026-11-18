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
| `POST /v1/completions` | One string prompt or one array of token IDs, `n=1`, greedy or sampled generation, streaming SSE and non-streaming JSON |
| `GET /v1/models` | Model currently loaded by this instance |
| `GET /health`, `GET /ready` | HTTP liveness and inference-worker readiness |
| `GET /metrics` | Queue, completions, generated tokens and KV page occupancy |

Completion parameters include `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream` and `ignore_eos`. Unsupported parameters return HTTP 400. Queue overload returns HTTP 503. The server binds loopback by default. Chat templating, tool calls, TLS and request authentication are outside this server's interface.

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

The scheduler packs visible decode and prefill work while reserving enough pages for admitted requests to finish. A single worker owns device execution and page reclamation. Attention gathers referenced KV pages into a batched SDPA call. For Ouro-1.4B BF16 with four loops, the raw KV cost is about 0.75 MiB per cached input position. `--kv-mib`, `--context`, `--max-batch` and `--prefill-chunk` tune the capacity and transient workspace.

## Verify and benchmark

~~~bash
python -m pip install '.[test]'
python -m pytest -q
loopinfer bench --url http://127.0.0.1:8000 --input-tokens 512 \
  --output-tokens 128 --concurrency 8 --requests 96 --output l20.json
~~~

The synthetic benchmark forces a fixed output length and reports successful throughput, P50/P95 client-observed time to first token and inter-token delay, plus failures. Keep checkpoint revision, loop count, readout, dtype, hardware, initial context length and output budget fixed across engines.

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
