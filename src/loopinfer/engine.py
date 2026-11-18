"""Single-owner GPU worker with request admission and continuous batching."""

import math
import queue
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch

from .cache import PagePool
from .loading import LoadedModel, load_model


class Overloaded(RuntimeError):
    pass


@dataclass(frozen=True)
class Sampling:
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    ignore_eos: bool = False

    def __post_init__(self):
        if not isinstance(self.max_tokens, int) or not 1 <= self.max_tokens:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < self.top_p <= 1 or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_p must be in (0, 1] and top_k must be a nonnegative integer")
        if self.seed is not None and (not isinstance(self.seed, int) or not 0 <= self.seed < 2**64):
            raise ValueError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    finish_reason: str
    first_token_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    finish_reason: str | None


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
    reserved_pages: int
    created: float
    first: float = 0
    cursor: int = 0
    pages: list[int] = field(default_factory=list)
    output: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None


def select_tokens(logits, requests, device):
    greedy = logits.argmax(dim=-1)
    sampled = []
    for row, request in enumerate(requests):
        sampling = request.sampling
        if sampling.temperature == 0:
            sampled.append(greedy[row])
            continue
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
        token = torch.multinomial(scaled.softmax(0), 1, generator=request.generator).squeeze(0)
        sampled.append(order.gather(0, token[None])[0] if order is not None else token)
    return torch.stack(sampled).tolist()


class Engine:
    """Use as a context manager. Concurrent submissions share one KV pool."""

    def __init__(
        self, model: str | LoadedModel, *, revision=None, loops=None, device="auto", dtype="auto",
        context=2048, page_size=16, kv_mib=None, max_batch=8, prefill_chunk=128,
        queue_size=128, stream_buffer=64,
    ):
        if min(context, page_size, max_batch, prefill_chunk, queue_size, stream_buffer) < 1:
            raise ValueError("Engine limits must be positive")
        if kv_mib is not None and (not math.isfinite(kv_mib) or kv_mib <= 0):
            raise ValueError("kv_mib must be finite and positive")
        self.loaded = model if isinstance(model, LoadedModel) else load_model(
            model, revision=revision, loops=loops, device=device, dtype=dtype,
        )
        self.model, self.tokenizer = self.loaded.model, self.loaded.tokenizer
        self.max_context = min(context, self.model.max_positions)
        self.max_batch, self.prefill_chunk, self.queue_size = max_batch, prefill_chunk, queue_size
        self.stream_buffer = stream_buffer
        page_bytes = page_size * self.model.cache_spec.token_bytes(self.loaded.dtype)
        if kv_mib is None:
            memory = int(torch.cuda.mem_get_info(self.loaded.device)[0] * 0.65) if (
                self.loaded.device.type == "cuda"
            ) else 256 * 1024**2
        else:
            memory = int(kv_mib * 1024**2)
        needed = max_batch * math.ceil(self.max_context / page_size)
        pages = min(needed, memory // page_bytes)
        if pages < 1:
            raise ValueError(f"KV budget does not fit one page ({page_bytes / 1024**2:.2f} MiB)")
        self.pool = PagePool(self.model.cache_spec, pages, page_size, self.loaded.device, self.loaded.dtype)
        self._condition = threading.Condition()
        self._waiting = deque()
        self._active: list[Request] = []
        self._next = self._reserved = self._completed = self._failed = self._generated = 0
        self._closing = False
        self._failure = None
        self._worker = threading.Thread(target=self._run, name="loopinfer-worker", daemon=True)
        self._worker.start()

    @property
    def ready(self):
        return self._worker.is_alive() and not self._closing

    def stats(self):
        with self._condition:
            return {
                "active": len(self._active), "waiting": len(self._waiting), "kv_pages_total": self.pool.pages,
                "kv_pages_reserved": self._reserved, "kv_pages_used": len(self.pool.owners),
                "kv_bytes_per_token": self.pool.spec.token_bytes(self.loaded.dtype),
                "completed": self._completed, "failed": self._failed, "generated_tokens": self._generated,
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
        pages = math.ceil(capacity / self.pool.page_size)
        if pages > self.pool.pages:
            limit = self.pool.pages * self.pool.page_size
            raise ValueError(f"Request exceeds KV pool capacity of {limit} cached positions")
        with self._condition:
            if self._closing:
                raise Overloaded("Inference worker is unavailable")
            if len(self._waiting) >= self.queue_size:
                raise Overloaded("Inference queue is full")
            handle = Handle(self._next, stream, self.stream_buffer)
            self._next += 1
            self._waiting.append(Request(handle, ids, sampling, pages, time.perf_counter()))
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
            self._reserved -= request.reserved_pages
        if error is None:
            ended = time.perf_counter()
            result = Generation(
                self.tokenizer.decode(request.output), tuple(request.output), len(request.prompt),
                finish_reason, request.first - request.created, ended - request.created,
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
            if self._reserved + request.reserved_pages > self.pool.pages:
                break
            self._waiting.popleft()
            self._reserved += request.reserved_pages
            self._active.append(request)
        return self._active[:]

    @torch.inference_mode()
    def _step(self, active):
        chunks, starts, emit = [], [], []
        for row, request in enumerate(active):
            start = request.cursor
            if start < len(request.prompt):
                tokens = request.prompt[start:start + self.prefill_chunk]
            else:
                tokens = request.output[-1:]
            chunks.append(tokens)
            starts.append(start)
            if start + len(tokens) >= len(request.prompt):
                emit.append(row)
        lengths = [len(chunk) for chunk in chunks]
        with self._condition:
            for request, start, length in zip(active, starts, lengths):
                self.pool.extend(request.handle.id, request.pages, start + length)
        batch = self.pool.prepare([r.pages for r in active], starts, lengths, emit)
        width = max(lengths)
        ids = torch.tensor([chunk + [0] * (width - len(chunk)) for chunk in chunks], device=self.loaded.device)
        logits = self.model(ids, batch, self.pool)
        picked = select_tokens(logits, [active[row] for row in emit], self.loaded.device) if emit else []
        return lengths, list(zip(emit, picked))

    def _commit(self, active, lengths, tokens):
        for request, length in zip(active, lengths):
            request.cursor += length
        for row, token in tokens:
            request = active[row]
            handle = request.handle
            if handle.cancelled.is_set():
                continue
            if not request.output:
                request.first = time.perf_counter()
            request.output.append(token)
            self._generated += 1
            if not request.sampling.ignore_eos and token in self.loaded.eos:
                reason = "stop"
            elif len(request.output) == request.sampling.max_tokens:
                reason = "length"
            else:
                reason = None
            if handle.events is not None:
                try:
                    handle.events.put_nowait(TokenEvent(token, reason))
                except queue.Full:
                    self._finish(request, error=Overloaded("Streaming consumer fell behind"))
                    continue
            if reason is not None:
                self._finish(request, finish_reason=reason)

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
                lengths, picked = self._step(active)
                with self._condition:
                    self._commit(active, lengths, picked)
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
