"""Single-owner GPU worker: admission, continuous batching, split prefill/decode steps."""

import logging
import math
import queue
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .cache import PagePool, PoolExhausted
from .loading import LoadedModel, load_model
from .models.native import Ouro
from .prefix import PrefixCache

LOGGER = logging.getLogger(__name__)
HISTORY_GRAIN = 64


class Overloaded(RuntimeError):
    pass


class _PoolFull(Exception):
    """KV growth failed for one request; the step is retried after preemption."""

    def __init__(self, request):
        super().__init__(f"KV pool exhausted while growing request {request.handle.id}")
        self.request = request


@dataclass(frozen=True)
class Sampling:
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()
    logprobs: int | None = None
    loop_depth: int | None = None
    exit_threshold: float | None = None

    def __post_init__(self):
        if self.exit_threshold is not None and not 0 < self.exit_threshold <= 1:
            raise ValueError("exit_threshold must be in (0, 1]")
        if self.exit_threshold is not None and self.loop_depth is not None:
            raise ValueError("Specify loop_depth or exit_threshold")
        if not isinstance(self.max_tokens, int) or not 1 <= self.max_tokens:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < self.top_p <= 1 or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_p must be in (0, 1] and top_k must be a nonnegative integer")
        if self.seed is not None and (not isinstance(self.seed, int) or not 0 <= self.seed < 2**64):
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not isinstance(self.stop, tuple) or not all(isinstance(text, str) and text for text in self.stop):
            raise ValueError("stop must be a tuple of non-empty strings")
        if self.logprobs is not None and (not isinstance(self.logprobs, int) or self.logprobs < 0):
            raise ValueError("logprobs must be a nonnegative integer or None")


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    finish_reason: str
    first_token_seconds: float
    elapsed_seconds: float
    logprobs: tuple = ()


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    finish_reason: str | None
    logprob: float | None = None
    top: tuple = ()


@dataclass(frozen=True)
class Pick:
    token: int
    logprob: float | None = None
    top: tuple = ()


class Handle:
    def __init__(self, request_id, stream, event_capacity):
        self.id = request_id
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.events = queue.Queue(maxsize=event_capacity) if stream else None
        self.result = None
        self.error = None

    def cancel(self):
        self.cancelled.set()

    def wait(self, timeout=None) -> Generation:
        if not self.done.wait(timeout):
            self.cancel()
            raise TimeoutError("Generation timed out")
        if self.error:
            raise self.error
        return self.result

    def consume(self, timeout=None):
        if self.events is None:
            raise ValueError("Request was submitted without streaming")
        deadline = time.perf_counter() + timeout if timeout is not None else math.inf
        try:
            while not self.done.is_set() or not self.events.empty():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("Streaming request timed out")
                try:
                    yield self.events.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
            self.wait()
        finally:
            if not self.done.is_set():
                self.cancel()


@dataclass
class Request:
    handle: Handle
    prompt: list[int]
    sampling: Sampling
    created: float
    first: float = 0.0
    cursor: int = 0
    pages: list[int] = field(default_factory=list)
    output: list[int] = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    generator: torch.Generator | None = None
    stopped_text: str | None = None
    depth: int | None = None
    round_index: int = 0
    latent: torch.Tensor | None = None
    prefix_keys: tuple[bytes, ...] = ()
    replay: tuple[int, ...] = ()


def _extras(row, token, count):
    scores = torch.log_softmax(row.float(), -1)
    chosen = float(scores[token])
    if count < 1:
        return chosen, ()
    values, ids = torch.topk(scores, min(count, scores.numel()))
    return chosen, tuple(zip(ids.tolist(), values.tolist()))


def select_tokens(logits, requests, device) -> list[Pick]:
    if all(r.sampling.temperature == 0 and r.sampling.logprobs is None for r in requests):
        return [Pick(token) for token in logits.argmax(dim=-1).tolist()]
    greedy = logits.argmax(dim=-1).tolist() if any(r.sampling.temperature == 0 for r in requests) else ()
    picks = []
    for row, request in enumerate(requests):
        sampling = request.sampling
        if sampling.temperature == 0:
            token = greedy[row]
        else:
            if request.generator is None:
                seed = sampling.seed if sampling.seed is not None else secrets.randbits(64)
                request.generator = torch.Generator(device=device).manual_seed(seed)
            scores = logits[row]
            scaled = (scores - scores.max()) / max(sampling.temperature, torch.finfo(scores.dtype).tiny)
            if sampling.top_k or sampling.top_p < 1:
                rank = min(sampling.top_k or scores.numel(), scores.numel())
                scaled, order = torch.topk(scaled, rank)
            else:
                order = None
            if sampling.top_p < 1:
                p = scaled.softmax(0)
                scaled = scaled.masked_fill(p.cumsum(0) - p >= sampling.top_p, -torch.inf)
            drawn = torch.multinomial(scaled.softmax(0), 1, generator=request.generator).squeeze(0)
            token = int(order.gather(0, drawn[None])[0]) if order is not None else int(drawn)
        if sampling.logprobs is None:
            picks.append(Pick(token))
        else:
            logprob, top = _extras(logits[row], token, sampling.logprobs)
            picks.append(Pick(token, logprob, top))
    return picks


class Engine:
    """Use as a context manager. Concurrent submissions share one KV pool."""

    def __init__(
        self, model: str | LoadedModel, *, revision=None, loops=None, device="auto", dtype="auto",
        context=2048, page_size=16, kv_mib=None, max_batch=8, prefill_chunk=128,
        queue_size=128, stream_buffer=64, cuda_graph="auto", graph_budget=32, attention="auto",
        scheduler="auto", prefix_cache_pages=0, profile_gpu=False, graph_mib=None, fuse_projections=False, speculate_tokens=0,
    ):
        if min(context, page_size, max_batch, prefill_chunk, queue_size, stream_buffer, graph_budget) < 1:
            raise ValueError("Engine limits must be positive")
        if attention not in {"auto", "sdpa", "triton", "flashinfer"}:
            raise ValueError("attention must be one of auto, sdpa, triton, flashinfer")
        if scheduler not in {"auto", "bucket", "round"} or prefix_cache_pages < 0:
            raise ValueError("Invalid scheduler or prefix cache capacity")
        if graph_mib is not None and (not math.isfinite(graph_mib) or graph_mib <= 0):
            raise ValueError("graph_mib must be finite and positive")
        if kv_mib is not None and (not math.isfinite(kv_mib) or kv_mib <= 0):
            raise ValueError("kv_mib must be finite and positive")
        self.loaded = model if isinstance(model, LoadedModel) else load_model(
            model, revision=revision, loops=loops, device=device, dtype=dtype,
        )
        self.model, self.tokenizer = self.loaded.model, self.loaded.tokenizer
        self.ouro = isinstance(self.model, Ouro)
        if not self.ouro and (scheduler == "round" or attention == "flashinfer" or prefix_cache_pages):
            raise ValueError("Round scheduling, prefix reuse and FlashInfer currently require Ouro")
        if fuse_projections and not self.ouro:
            raise ValueError("Projection fusion currently requires Ouro")
        if fuse_projections:
            if isinstance(model, LoadedModel):
                raise ValueError("Pass a checkpoint path to use projection fusion")
            self.model.pack_projections()
        if type(speculate_tokens) is not int or not 0 <= speculate_tokens <= 16:
            raise ValueError("speculate_tokens must be between 0 and 16")
        if speculate_tokens and (not self.ouro or self.model.loops < 2 or attention not in {"sdpa", "auto"}):
            raise ValueError("Speculative decoding requires Ouro with at least two loops and SDPA")
        self.speculate_tokens = speculate_tokens
        self._draft_tokens = self._verified_tokens = self._speculative_steps = 0
        self.scheduler = scheduler
        self.profile_gpu = profile_gpu and self.loaded.device.type == "cuda"
        self._gpu_events = deque()
        self._gpu_timings = defaultdict(float)
        self._gpu_counts = defaultdict(int)
        self._round_rows = self._mixed_round_batches = self._replayed_tokens = 0
        self._rejected = self._queue_samples = 0
        self._queue_seconds = 0.0
        self._prefill_buffers = None
        self._graph_bytes = 0
        self._graph_limit = int(graph_mib * 1024**2) if graph_mib is not None else None
        self.max_context = min(context, self.model.max_positions)
        self.max_batch, self.prefill_chunk, self.queue_size = max_batch, prefill_chunk, queue_size
        self.stream_buffer, self.graph_budget = stream_buffer, graph_budget
        page_bytes = page_size * self.model.cache_spec.token_bytes(self.loaded.dtype)
        if kv_mib is None:
            memory = int(torch.cuda.mem_get_info(self.loaded.device)[0] * 0.65) if (
                self.loaded.device.type == "cuda"
            ) else 256 * 1024**2
        else:
            memory = int(kv_mib * 1024**2)
        pages = min(max_batch * math.ceil(self.max_context / page_size), memory // page_bytes)
        if pages < 1:
            raise ValueError(f"KV budget does not fit one page ({page_bytes / 1024**2:.2f} MiB)")
        self.pool = PagePool(self.model.cache_spec, pages, page_size, self.loaded.device, self.loaded.dtype)
        self.use_graphs = (
            cuda_graph is True or (cuda_graph == "auto" and self.loaded.device.type == "cuda")
        )
        from .kernels import available as kernel_available

        self.prefix = PrefixCache(self.pool, min(prefix_cache_pages, pages))
        if attention == "flashinfer":
            from .paged import PagedAttention, unavailable

            reason = unavailable(self.loaded.device, self.loaded.dtype)
            if reason:
                raise ValueError(f"FlashInfer {reason}")
            self.pool.flash = PagedAttention(self.pool, self.model.heads)
            self.use_graphs = False
        self.attention = attention
        # The Triton kernel is experimental: it matches SDPA numerically but is
        # slower than gather + SDPA at the context lengths we serve, so "auto"
        # keeps the reference path.
        self.pool.use_kernel = attention == "triton" and kernel_available(self.loaded.device)
        # The standalone Triton path reads transient page tables rather than
        # the reusable slot and mask buffers owned by the SDPA graphs.
        if self.pool.use_kernel:
            self.use_graphs = False
        self._graphs: dict[tuple[int, int, int | None], tuple] = {}
        self._graph_replays = self._graph_captures = self._graph_evictions = 0
        self._condition = threading.Condition()
        self._waiting = deque()
        self._active: list[Request] = []
        self._next = self._preemptions = self._completed = self._failed = self._generated = 0
        self._memory_blocked = False
        self._steps = 0
        self._timing = dict.fromkeys(("prepare", "forward", "sample", "commit", "step"), 0.0)
        self._closing = False
        self._draining = False
        self._failure = None
        self._worker = threading.Thread(target=self._run, name="loopinfer-worker", daemon=True)
        self._worker.start()

    @property
    def ready(self):
        return self._worker.is_alive() and not (self._closing or self._draining)

    def stats(self):
        with self._condition:
            steps = max(1, self._steps)
            timings = {name: round(1000 * value / steps, 3) for name, value in self._timing.items()}
            return {
                "active": len(self._active), "waiting": len(self._waiting),
                "max_batch": self.max_batch, "prefill_chunk": self.prefill_chunk,
                "kv_pages_total": self.pool.pages, "kv_pages_used": self.pool.used,
                "kv_pages_free": self.pool.free_pages, "kv_pages_peak": self.pool.peak,
                "kv_bytes_per_token": self.pool.spec.token_bytes(self.loaded.dtype),
                "completed": self._completed, "failed": self._failed, "generated_tokens": self._generated,
                "preemptions": self._preemptions, "steps": self._steps,
                **{f"step_ms_{name}": value for name, value in timings.items()},
                "cuda_graph": int(self.use_graphs), "cuda_graph_captures": self._graph_captures,
                "cuda_graph_replays": self._graph_replays, "cuda_graph_evictions": self._graph_evictions,
                "attention_kernel": int(self.pool.use_kernel),
                "attention_flashinfer": int(self.pool.flash is not None),
                "kv_capacity_bytes": self.pool.tensor.numel() * self.pool.tensor.element_size(),
                "prefix_cached_pages": len(self.prefix.entries),
                "prefix_hit_tokens": self.prefix.hit_tokens, "prefix_evictions": self.prefix.evictions,
                "replayed_tokens": self._replayed_tokens, "rejected": self._rejected,
                "queue_ms_mean": round(1000 * self._queue_seconds / max(1, self._queue_samples), 3),
                "decode_round_rows": self._round_rows, "mixed_round_batches": self._mixed_round_batches,
                "cuda_graph_estimated_bytes": self._graph_bytes,
                "speculative_draft_tokens": self._draft_tokens, "speculative_verified_tokens": self._verified_tokens,
                "speculative_steps": self._speculative_steps,
                **{f"gpu_ms_{name}": round(value / self._gpu_counts[name], 3)
                   for name, value in self._gpu_timings.items()},
            }

    def submit(self, prompt: str | list[int], sampling=None, *, stream=False, cache_namespace="") -> Handle:
        sampling = Sampling() if sampling is None else sampling
        depth = getattr(self.model, "loops", None) if sampling.loop_depth is None else sampling.loop_depth
        if sampling.loop_depth is not None and (
            not self.ouro or type(depth) is not int or not 1 <= depth <= self.model.loops
        ):
            raise ValueError("loop_depth requires Ouro and must be between 1 and the server loop limit")
        if sampling.exit_threshold is not None and not self.ouro:
            raise ValueError("exit_threshold requires the Ouro adapter")
        if not isinstance(cache_namespace, str) or len(cache_namespace) > 256:
            raise ValueError("cache_namespace must be a string of at most 256 characters")
        ids = self.tokenizer.encode(prompt).ids if isinstance(prompt, str) else list(prompt)
        if not ids:
            if self.loaded.bos is None:
                raise ValueError("Empty prompt and no configured BOS token")
            ids = [self.loaded.bos]
        if any(not isinstance(token, int) or not 0 <= token < self.model.vocab_size for token in ids):
            raise ValueError("Prompt contains invalid token IDs")
        capacity = len(ids) + sampling.max_tokens - 1
        if capacity > self.max_context:
            raise ValueError(f"Request needs {capacity} cached positions, context limit is {self.max_context}")
        if math.ceil(capacity / self.pool.page_size) > self.pool.pages:
            limit = self.pool.pages * self.pool.page_size
            raise ValueError(f"Request exceeds KV pool capacity of {limit} cached positions")
        with self._condition:
            if self._closing or self._draining:
                raise Overloaded("Inference worker is unavailable")
            if len(self._waiting) >= self.queue_size:
                self._rejected += 1
                raise Overloaded("Inference queue is full")
            handle = Handle(self._next, stream, self.stream_buffer)
            self._next += 1
            request = Request(handle, ids, sampling, time.perf_counter(), depth=depth)
            request.prefix_keys = self.prefix.keys(ids, depth, cache_namespace)
            self._waiting.append(request)
            self._condition.notify()
            return handle

    def generate(self, prompts, sampling=None, *, timeout=None):
        single = isinstance(prompts, str)
        items = [prompts] if single else prompts
        handles = []
        try:
            for prompt in items:
                handles.append(self.submit(prompt, sampling))
            deadline = time.perf_counter() + timeout if timeout is not None else math.inf
            results = [
                handle.wait(max(0, deadline - time.perf_counter()) if timeout is not None else None)
                for handle in handles
            ]
        finally:
            for handle in handles:
                if not handle.done.is_set():
                    handle.cancel()
        return results[0] if single else results

    def _finish(self, request, *, finish_reason=None, error=None):
        handle = request.handle
        self.pool.release(handle.id, request.pages)
        if request in self._active:
            self._active.remove(request)
        self._memory_blocked = False
        if error is None:
            ended = time.perf_counter()
            text = self.tokenizer.decode(request.output) if request.stopped_text is None else request.stopped_text
            result = Generation(
                text, tuple(request.output), len(request.prompt), finish_reason,
                request.first - request.created, ended - request.created, tuple(request.logprobs),
            )
            handle.result = result
            self._completed += 1
        else:
            handle.error = error
            self._failed += 1
        handle.done.set()
        if self._draining:
            self._condition.notify_all()

    def _take_batch(self):
        for request in self._active[:]:
            if request.handle.cancelled.is_set():
                self._finish(request, error=RuntimeError("Request cancelled"))
        for request in list(self._waiting):
            if request.handle.cancelled.is_set():
                self._waiting.remove(request)
                self._finish(request, error=RuntimeError("Request cancelled"))
        while self._waiting and len(self._active) < self.max_batch:
            request = self._waiting[0]
            # After a preemption the queue stays shut until some request finishes,
            # otherwise the preempted request is immediately re-admitted and we thrash.
            if self._memory_blocked and self._active:
                break
            if self.pool.free_pages < self._first_step_pages(request) and self._active:
                break
            self._waiting.popleft()
            self.prefix.attach(request)
            if not request.replay:
                self._queue_samples += 1
                self._queue_seconds += time.perf_counter() - request.created
            self._active.append(request)
        return self._active

    def _first_step_pages(self, request):
        return -(-min(len(request.prompt), self.prefill_chunk) // self.pool.page_size)

    def _plan(self, active):
        plan = []
        for request in active:
            length = len(request.replay) if request.replay else len(request.prompt)
            if request.cursor < length:
                advance = min(self.prefill_chunk, length - request.cursor)
                plan.append((request, advance, "prefill"))
            else:
                plan.append((request, 1, "decode"))
        return plan

    def _reserve(self, plan):
        for request, advance, _ in plan:
            while True:
                try:
                    self.pool.extend(request.handle.id, request.pages, request.cursor + advance)
                    break
                except PoolExhausted:
                    if not self.prefix.evict():
                        self._memory_blocked = True
                        raise _PoolFull(request) from None

    def _preempt(self, active, protected=None):
        """Rebuild KV from committed tokens without resampling or re-emitting."""
        for request in reversed(active):
            if not request.pages or request is protected:
                continue
            self.pool.release(request.handle.id, request.pages)
            request.cursor = request.round_index = 0
            request.latent = None
            request.replay = (*request.prompt, *request.output[:-1]) if request.output else ()
            active.remove(request)
            self._waiting.appendleft(request)
            self._preemptions += 1
            return True
        return False

    def _forward_depth(self, ids, batch, depth, threshold=None):
        if self.ouro and (depth != self.model.loops or threshold is not None):
            return self.model.forward_depth(ids, batch, self.pool, depth, threshold=threshold)
        return self.model(ids, batch, self.pool)

    @contextmanager
    def _measure_gpu(self, name):
        if not self.profile_gpu:
            yield
            return
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        with torch.cuda.nvtx.range(f"loopinfer/{name}"):
            yield
        end.record()
        self._gpu_events.append((name, start, end))

    def _collect_gpu(self, *, drain=False):
        while self._gpu_events:
            name, start, end = self._gpu_events[0]
            if drain:
                end.synchronize()
            elif not end.query():
                break
            self._gpu_events.popleft()
            self._gpu_timings[name] += start.elapsed_time(end)
            self._gpu_counts[name] += 1

    def _history_bucket(self, length):
        return min(self.max_context, -(-length // HISTORY_GRAIN) * HISTORY_GRAIN)

    def _capture(self, buffers, batch, depth):
        device = self.loaded.device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward_depth(buffers.ids, batch, depth)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self._forward_depth(buffers.ids, batch, depth)
        return graph, logits

    def _trim_graphs(self):
        while len(self._graphs) > self.graph_budget or (
            self._graph_limit is not None and self._graph_bytes > self._graph_limit
        ):
            key = next(iter(self._graphs))
            entry = self._graphs.pop(key)
            torch.cuda.synchronize(self.loaded.device)
            self._graph_bytes -= entry[3]
            del entry
            self._graph_evictions += 1

    def _graph_decode(self, rows, ids, starts):
        count = len(rows)
        history = self._history_bucket(max(starts) + 1)
        depth = rows[0][0].depth
        key = (count, history, depth)
        entry = self._graphs.get(key)
        buffers = entry[0] if entry is not None else self.pool.make_buffers(count, 1, history)
        started = time.perf_counter()
        batch = self.pool.prepare_into(
            buffers, ids, [request.pages for request, _ in rows], starts, [1] * count, list(range(count)),
        )
        self._timing["prepare"] += time.perf_counter() - started
        if entry is None:
            before = torch.cuda.memory_allocated(self.loaded.device)
            graph, logits = self._capture(buffers, batch, depth)
            size = sum(t.numel() * t.element_size() for t in vars(buffers).values() if isinstance(t, torch.Tensor))
            size += max(logits.numel() * logits.element_size(),
                        torch.cuda.memory_allocated(self.loaded.device) - before)
            entry = (buffers, graph, logits, size)
            if self._graph_limit is None or size <= self._graph_limit:
                self._graphs[key] = entry
                self._graph_bytes += size
            self._graph_captures += 1
            self._trim_graphs()
        started = time.perf_counter()
        with self._measure_gpu("decode"):
            entry[1].replay()
        self._timing["forward"] += time.perf_counter() - started
        self._graph_replays += 1
        picks = self._timed_pick(entry[2], [request for request, _ in rows])
        return [(request, 1, pick) for (request, _), pick in zip(rows, picks)]

    def _timed_pick(self, logits, requests):
        # The first device-to-host read of the logits happens here, so this also
        # absorbs the GPU time the launch-only measurement above misses.
        started = time.perf_counter()
        picks = select_tokens(logits, requests, self.loaded.device)
        self._timing["sample"] += time.perf_counter() - started
        return picks

    def _prefill(self, rows):
        starts = [request.cursor for request, _ in rows]
        chunks = [(request.replay or request.prompt)[start:start + advance]
                  for (request, advance), start in zip(rows, starts)]
        lengths = [len(chunk) for chunk in chunks]
        emit = [row for row, ((request, advance), start) in enumerate(zip(rows, starts))
                if not request.output and start + advance >= len(request.prompt)]
        tables = [request.pages for request, _ in rows]
        width = max(lengths)
        started = time.perf_counter()
        if self.pool.flash is None:
            history = max(a + b for a, b in zip(starts, lengths))
            buffers = self._prefill_buffers
            if buffers is None or (buffers.batch, buffers.width, buffers.history) != (len(rows), width, history):
                buffers = self.pool.make_buffers(len(rows), width, history)
                self._prefill_buffers = buffers
            batch = self.pool.prepare_into(buffers, chunks, tables, starts, lengths, emit)
            ids = buffers.ids
        else:
            batch = self.pool.prepare(tables, starts, lengths, emit)
            ids = torch.tensor([list(chunk) + [0] * (width - len(chunk)) for chunk in chunks],
                               device=self.loaded.device)
        self._timing["prepare"] += time.perf_counter() - started
        started = time.perf_counter()
        with self._measure_gpu("prefill"):
            logits = self._forward_depth(ids, batch, rows[0][0].depth, rows[0][0].sampling.exit_threshold)
        self._timing["forward"] += time.perf_counter() - started
        picks = self._timed_pick(logits, [rows[row][0] for row in emit]) if emit else []
        results = [(request, advance, None) for request, advance in rows]
        for row, pick in zip(emit, picks):
            results[row] = (rows[row][0], rows[row][1], pick)
        return results

    def _decode(self, rows):
        starts = [request.cursor for request, _ in rows]
        ids = [[request.output[-1]] for request, _ in rows]
        threshold = rows[0][0].sampling.exit_threshold
        if self.use_graphs and threshold is None:
            try:
                return self._graph_decode(rows, ids, starts)
            except Exception:
                LOGGER.warning("CUDA graph path disabled after a capture failure", exc_info=True)
                self.use_graphs = False
                self._graphs.clear()
                self._graph_bytes = 0
        started = time.perf_counter()
        tables = [request.pages for request, _ in rows]
        batch = self.pool.prepare(tables, starts, [1] * len(rows), list(range(len(rows))))
        self._timing["prepare"] += time.perf_counter() - started
        started = time.perf_counter()
        with self._measure_gpu("decode"):
            logits = self._forward_depth(torch.tensor(ids, device=self.loaded.device), batch, rows[0][0].depth, threshold)
        self._timing["forward"] += time.perf_counter() - started
        picks = self._timed_pick(logits, [request for request, _ in rows])
        return [(request, 1, pick) for (request, _), pick in zip(rows, picks)]

    def _decode_round(self, rows):
        by_round = defaultdict(list)
        for request, _ in rows:
            by_round[request.round_index].append(request)
        groups, ordered, hidden = [], [], []
        started = time.perf_counter()
        for depth, requests in sorted(by_round.items()):
            batch = self.pool.prepare([r.pages for r in requests], [r.cursor for r in requests],
                                      [1] * len(requests), list(range(len(requests))))
            groups.append((depth, batch, requests))
            ordered.extend(requests)
            if depth == 0:
                ids = torch.tensor([[r.output[-1]] for r in requests], device=self.loaded.device)
                hidden.append(self.model.model.embed_tokens(ids))
            else:
                hidden.append(torch.stack([r.latent for r in requests]))
        self._timing["prepare"] += time.perf_counter() - started
        self._mixed_round_batches += int(len(groups) > 1)
        self._round_rows += len(ordered)
        started = time.perf_counter()
        with self._measure_gpu("round"):
            latents = self.model.decode_round(torch.cat(hidden), groups, self.pool)
        self._timing["forward"] += time.perf_counter() - started
        finished, indices = [], []
        for index, request in enumerate(ordered):
            request.round_index += 1
            if request.round_index == request.depth:
                request.round_index = 0
                request.latent = None
                indices.append(index)
                finished.append(request)
            else:
                request.latent = latents[index]
        if not finished:
            return []
        logits = self.model.lm_head(latents[indices, 0]).float()
        picks = self._timed_pick(logits, finished)
        return [(request, 1, pick) for request, pick in zip(finished, picks)]

    def _speculate(self, request):
        """Verify a short depth-one greedy draft with the request's full depth."""
        sample = request.sampling
        if (sample.temperature != 0 or sample.logprobs is not None or sample.exit_threshold is not None
                or request.depth < 2 or request.round_index):
            return []
        pool, start = self.pool, request.cursor
        headroom = (len(request.pages) + pool.free_pages) * pool.page_size - start
        count = min(self.speculate_tokens, sample.max_tokens - len(request.output), headroom)
        if count < 2:
            return []
        pool.extend(request.handle.id, request.pages, start + count)
        inputs = [request.output[-1]]
        with self._measure_gpu("speculate"):
            for offset in range(count - 1):
                batch = pool.prepare([request.pages], [start + offset], [1], [0])
                ids = torch.tensor([[inputs[-1]]], device=self.loaded.device)
                inputs.append(int(self._forward_depth(ids, batch, 1).argmax(dim=-1)[0]))
            batch = pool.prepare([request.pages], [start], [count], [0])
            # Verify every proposed position with full-depth causal teacher forcing.
            batch.output_rows = torch.zeros(count, device=self.loaded.device, dtype=torch.long)
            batch.output_cols = torch.arange(count, device=self.loaded.device)
            ids = torch.tensor([inputs], device=self.loaded.device)
            verified = self._forward_depth(ids, batch, request.depth).argmax(dim=-1).tolist()
        results = []
        for index, token in enumerate(verified):
            results.append((request, 1, Pick(token)))
            if index < count - 1 and token != inputs[index + 1]:
                break
        self._speculative_steps += 1
        self._draft_tokens += count - 1
        self._verified_tokens += len(results)
        # The causal mask limits reads to the committed cursor, so positions
        # after a rejected draft can stay in the page until overwritten.
        return results

    def _by_depth(self, rows, runner):
        buckets = defaultdict(list)
        for row in rows:
            request = row[0]
            buckets[request.depth, request.sampling.exit_threshold].append(row)
        return [item for bucket in buckets.values() for item in runner(bucket)]

    def _step(self, active):
        """Run one step: prefill rows and decode rows go through separate sub-batches.

        Mixing the two used to force the whole batch onto the eager path, which cost
        every decode row a graph replay whenever a new request started prefill.
        """
        while True:
            plan = self._plan(active)
            with self._condition:
                try:
                    self._reserve(plan)
                    break
                except _PoolFull as failure:
                    if self._preempt(active, failure.request):
                        continue
                    raise
        results = []
        prefill = [(request, advance) for request, advance, kind in plan if kind == "prefill"]
        decode = [(request, advance) for request, advance, kind in plan if kind == "decode"]
        if self.speculate_tokens and len(decode) == 1 and not prefill:
            results = self._speculate(decode[0][0])
            if results:
                return results
        gated = [row for row in decode if row[0].sampling.exit_threshold is not None]
        decode = [row for row in decode if row[0].sampling.exit_threshold is None]
        if gated:
            results.extend(self._by_depth(gated, self._decode))
        if decode:
            mixed = len({request.depth for request, _ in decode}) > 1
            pending = any(request.round_index for request, _ in decode)
            if self.ouro and (pending or self.scheduler == "round" or (self.scheduler == "auto" and mixed)):
                results.extend(self._decode_round(decode))
            else:
                results.extend(self._by_depth(decode, self._decode))
        if prefill:
            results.extend(self._by_depth(prefill, self._prefill))
        return results

    def _commit(self, results):
        for request, advance, pick in results:
            if request.handle.done.is_set() or request.handle.cancelled.is_set():
                continue
            previous = request.cursor
            request.cursor += advance
            if previous < len(request.prompt) and not request.handle.cancelled.is_set():
                self.prefix.publish(request, previous, request.cursor)
            if request.replay:
                self._replayed_tokens += advance
                if request.cursor >= len(request.replay):
                    request.replay = ()
            if pick is None or request.handle.cancelled.is_set():
                continue
            handle = request.handle
            if not request.output:
                request.first = time.perf_counter()
            request.output.append(pick.token)
            if request.sampling.logprobs is not None:
                request.logprobs.append((pick.logprob, pick.top))
            self._generated += 1
            if not request.sampling.ignore_eos and pick.token in self.loaded.eos:
                reason = "stop"
            elif len(request.output) == request.sampling.max_tokens:
                reason = "length"
            else:
                reason = self._stop_hit(request)
            if handle.events is not None:
                try:
                    handle.events.put_nowait(TokenEvent(pick.token, reason, pick.logprob, pick.top))
                except queue.Full:
                    self._finish(request, error=Overloaded("Streaming consumer fell behind"))
                    continue
            if reason is not None:
                self._finish(request, finish_reason=reason)

    def _stop_hit(self, request):
        text = self.tokenizer.decode(request.output)
        cut = min((text.find(stop) for stop in request.sampling.stop if stop in text), default=-1)
        if cut < 0:
            return None
        request.stopped_text = text[:cut]
        return "stop"

    @torch.inference_mode()
    def _run(self):
        if self.loaded.device.type == "cuda":
            with torch.cuda.device(self.loaded.device):
                self._run_worker()
        else:
            self._run_worker()

    def _run_worker(self):
        try:
            while True:
                with self._condition:
                    if self._closing:
                        return
                    active = self._take_batch()
                    if not active:
                        self._collect_gpu(drain=True)
                        self._condition.wait()
                        continue
                started = time.perf_counter()
                results = self._step(active)
                with self._condition:
                    committed = time.perf_counter()
                    self._commit(results)
                    self._timing["commit"] += time.perf_counter() - committed
                    self._timing["step"] += time.perf_counter() - started
                    self._steps += 1
                    self._collect_gpu()
        except Exception as exc:
            self._failure = exc
        finally:
            with self._condition:
                self._closing = True
                error = self._failure if self._failure is not None else Overloaded("Inference worker closed")
                for request in self._active[:] + list(self._waiting):
                    self._finish(request, error=error)
                self._waiting.clear()

    def close(self, *, drain=False, timeout=None):
        with self._condition:
            if drain:
                self._draining = True
                self._condition.wait_for(
                    lambda: self._closing or not (self._active or self._waiting), timeout=timeout,
                )
            self._closing = True
            self._condition.notify_all()
        self._worker.join()
        self._graphs.clear()
        self._graph_bytes = 0
        with self._condition:
            while self.prefix.evict():
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
