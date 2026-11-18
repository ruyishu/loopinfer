"""Single-owner GPU worker: admission, continuous batching, split prefill/decode steps."""

import logging
import math
import queue
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from .cache import PagePool, PoolExhausted
from .loading import LoadedModel, load_model

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

    def __post_init__(self):
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


def _extras(row, token, count):
    scores = torch.log_softmax(row.float(), -1)
    chosen = float(scores[token])
    if count < 1:
        return chosen, ()
    values, ids = torch.topk(scores, min(count, scores.numel()))
    return chosen, tuple(zip(ids.tolist(), values.tolist()))


def select_tokens(logits, requests, device) -> list[Pick]:
    greedy = logits.argmax(dim=-1).tolist()
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
    ):
        if min(context, page_size, max_batch, prefill_chunk, queue_size, stream_buffer, graph_budget) < 1:
            raise ValueError("Engine limits must be positive")
        if attention not in {"auto", "sdpa", "triton"}:
            raise ValueError("attention must be one of auto, sdpa, triton")
        if kv_mib is not None and (not math.isfinite(kv_mib) or kv_mib <= 0):
            raise ValueError("kv_mib must be finite and positive")
        self.loaded = model if isinstance(model, LoadedModel) else load_model(
            model, revision=revision, loops=loops, device=device, dtype=dtype,
        )
        self.model, self.tokenizer = self.loaded.model, self.loaded.tokenizer
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

        self.attention = attention
        # The Triton kernel is experimental: it matches SDPA numerically but is
        # slower than gather + SDPA at the context lengths we serve, so "auto"
        # keeps the reference path.
        self.pool.use_kernel = attention == "triton" and kernel_available(self.loaded.device)
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._graph_replays = self._graph_captures = self._graph_evictions = 0
        self._condition = threading.Condition()
        self._waiting = deque()
        self._active: list[Request] = []
        self._next = self._preemptions = self._completed = self._failed = self._generated = 0
        self._memory_blocked = False
        self._steps = 0
        self._timing = dict.fromkeys(("prepare", "forward", "sample", "commit", "step"), 0.0)
        self._closing = False
        self._failure = None
        self._worker = threading.Thread(target=self._run, name="loopinfer-worker", daemon=True)
        self._worker.start()

    @property
    def ready(self):
        return self._worker.is_alive() and not self._closing

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
            }

    def submit(self, prompt: str | list[int], sampling=None, *, stream=False) -> Handle:
        sampling = Sampling() if sampling is None else sampling
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
            if self._closing:
                raise Overloaded("Inference worker is unavailable")
            if len(self._waiting) >= self.queue_size:
                raise Overloaded("Inference queue is full")
            handle = Handle(self._next, stream, self.stream_buffer)
            self._next += 1
            self._waiting.append(Request(handle, ids, sampling, time.perf_counter()))
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
            if self._memory_blocked:
                break
            if self.pool.free_pages < self._first_step_pages(request) and self._active:
                break
            self._waiting.popleft()
            self._active.append(request)
        return self._active

    def _first_step_pages(self, request):
        return -(-min(len(request.prompt), self.prefill_chunk) // self.pool.page_size)

    def _plan(self, active):
        plan = []
        for request in active:
            if request.cursor < len(request.prompt):
                advance = min(self.prefill_chunk, len(request.prompt) - request.cursor)
                plan.append((request, advance, "prefill"))
            else:
                plan.append((request, 1, "decode"))
        return plan

    def _reserve(self, plan):
        for request, advance, _ in plan:
            try:
                self.pool.extend(request.handle.id, request.pages, request.cursor + advance)
            except PoolExhausted:
                self._memory_blocked = True
                raise _PoolFull(request) from None

    def _preempt(self, active, protected=None):
        """Free KV pages by dropping a request; a restarted request re-runs its prompt."""
        fallback = None
        for request in reversed(active):
            if not request.pages or request is protected:
                continue
            if request.handle.events is not None:
                fallback = fallback or request
                continue
            self.pool.release(request.handle.id, request.pages)
            request.cursor = 0
            request.output.clear()
            request.logprobs.clear()
            request.first = 0.0
            request.stopped_text = None
            active.remove(request)
            self._waiting.appendleft(request)
            self._preemptions += 1
            LOGGER.info("Preempted request %s to free KV pages", request.handle.id)
            return True
        if fallback is not None:
            # Streamed text cannot be retracted, so a streaming request is failed
            # instead of being restarted.
            self._preemptions += 1
            self._finish(fallback, error=Overloaded("Request preempted to free KV pages"))
            return True
        return False

    def _history_bucket(self, length):
        return min(self.max_context, -(-length // HISTORY_GRAIN) * HISTORY_GRAIN)

    def _capture(self, buffers, batch):
        device = self.loaded.device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.model(buffers.ids, batch, self.pool)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = self.model(buffers.ids, batch, self.pool)
        return graph, logits

    def _trim_graphs(self):
        while len(self._graphs) > self.graph_budget:
            key = next(iter(self._graphs))
            entry = self._graphs.pop(key)
            torch.cuda.synchronize(self.loaded.device)
            del entry
            self._graph_evictions += 1

    def _graph_decode(self, rows, ids, starts):
        count = len(rows)
        history = self._history_bucket(max(starts) + 1)
        key = (count, history)
        entry = self._graphs.get(key)
        buffers = entry[0] if entry is not None else self.pool.make_buffers(count, 1, history)
        started = time.perf_counter()
        batch = self.pool.prepare_into(
            buffers, ids, [request.pages for request, _ in rows], starts, [1] * count, list(range(count)),
        )
        self._timing["prepare"] += time.perf_counter() - started
        if entry is None:
            graph, logits = self._capture(buffers, batch)
            entry = (buffers, graph, logits)
            self._graphs[key] = entry
            self._graph_captures += 1
            self._trim_graphs()
        started = time.perf_counter()
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
        chunks = [request.prompt[start:start + advance] for (request, advance), start in zip(rows, starts)]
        lengths = [len(chunk) for chunk in chunks]
        emit = [row for row, ((request, advance), start) in enumerate(zip(rows, starts))
                if start + advance >= len(request.prompt)]
        width = max(lengths)
        started = time.perf_counter()
        buffers = self.pool.make_buffers(len(rows), width, max(a + b for a, b in zip(starts, lengths)))
        batch = self.pool.prepare_into(buffers, chunks, [request.pages for request, _ in rows], starts, lengths, emit)
        self._timing["prepare"] += time.perf_counter() - started
        ids = torch.tensor([chunk + [0] * (width - len(chunk)) for chunk in chunks], device=self.loaded.device)
        started = time.perf_counter()
        logits = self.model(ids, batch, self.pool)
        self._timing["forward"] += time.perf_counter() - started
        picks = self._timed_pick(logits, [rows[row][0] for row in emit]) if emit else []
        results = [(request, advance, None) for request, advance in rows]
        for row, pick in zip(emit, picks):
            results[row] = (rows[row][0], rows[row][1], pick)
        return results

    def _decode(self, rows):
        starts = [request.cursor for request, _ in rows]
        ids = [[request.output[-1]] for request, _ in rows]
        if self.use_graphs:
            try:
                return self._graph_decode(rows, ids, starts)
            except Exception:
                LOGGER.warning("CUDA graph path disabled after a capture failure", exc_info=True)
                self.use_graphs = False
                self._graphs.clear()
        started = time.perf_counter()
        buffers = self.pool.make_buffers(len(rows), 1, max(starts) + 1)
        batch = self.pool.prepare_into(
            buffers, ids, [request.pages for request, _ in rows], starts, [1] * len(rows),
            list(range(len(rows))),
        )
        self._timing["prepare"] += time.perf_counter() - started
        started = time.perf_counter()
        logits = self.model(torch.tensor(ids, device=self.loaded.device), batch, self.pool)
        self._timing["forward"] += time.perf_counter() - started
        picks = self._timed_pick(logits, [request for request, _ in rows])
        return [(request, 1, pick) for (request, _), pick in zip(rows, picks)]

    @torch.inference_mode()
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
        if prefill:
            results.extend(self._prefill(prefill))
        if decode:
            results.extend(self._decode(decode))
        return results

    def _commit(self, results):
        for request, advance, pick in results:
            request.cursor += advance
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

    def _run(self):
        try:
            while True:
                with self._condition:
                    if self._closing:
                        return
                    active = self._take_batch()
                    if not active:
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
        except Exception as exc:
            self._failure = exc
        finally:
            with self._condition:
                self._closing = True
                error = self._failure if self._failure is not None else Overloaded("Inference worker closed")
                for request in self._active[:] + list(self._waiting):
                    self._finish(request, error=error)
                self._waiting.clear()

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._worker.join()
        self._graphs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
