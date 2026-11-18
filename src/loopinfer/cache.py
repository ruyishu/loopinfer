from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


class PoolExhausted(RuntimeError):
    """The KV pool cannot grow a page table any further."""


@dataclass(frozen=True)
class CacheSpec:
    domains: int
    layers: int
    heads: int
    head_dim: int

    def token_bytes(self, dtype: torch.dtype) -> int:
        return 2 * self.domains * self.layers * self.heads * self.head_dim * dtype.itemsize


@dataclass
class CacheBatch:
    positions: torch.Tensor
    valid: torch.Tensor
    write_slots: torch.Tensor
    write_rows: torch.Tensor
    write_cols: torch.Tensor
    history_slots: torch.Tensor
    causal: torch.Tensor
    output_rows: torch.Tensor
    output_cols: torch.Tensor
    tables: torch.Tensor | None = None
    lengths: torch.Tensor | None = None


@dataclass(frozen=True)
class RoundBatch:
    groups: list

    @property
    def sizes(self):
        return [len(rows) for _, _, rows in self.groups]


@dataclass
class StepBuffers:
    """Reusable device tensors for one forward pass.

    Fixed addresses let a CUDA Graph capture the whole decode step once and replay it.
    """

    ids: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    write_slots: torch.Tensor
    write_rows: torch.Tensor
    write_cols: torch.Tensor
    history_slots: torch.Tensor
    causal: torch.Tensor
    output_rows: torch.Tensor
    output_cols: torch.Tensor
    batch: int
    width: int
    history: int


def pages_to_slots(table, page_size: int) -> np.ndarray:
    """Flatten a page table into slot indices, ordered by paged position."""
    if not table:
        return np.empty(0, dtype=np.int64)
    base = np.asarray(table, dtype=np.int64) * page_size
    return (base[:, None] + np.arange(page_size, dtype=np.int64)).reshape(-1)


class PagePool:
    """Physical pages are shared across requests, never across live owners.

    Pages are handed out on demand as a request's cached length grows; callers that
    run out of pages see :class:`PoolExhausted` and can preempt a request.
    """

    def __init__(self, spec: CacheSpec, pages: int, page_size: int, device, dtype):
        if min(pages, page_size) < 1:
            raise ValueError("KV pool requires at least one page")
        self.spec, self.pages, self.page_size = spec, pages, page_size
        self.device = torch.device(device)
        self.tensor = torch.empty(
            spec.domains, spec.layers, 2, pages * page_size, spec.heads, spec.head_dim,
            device=device, dtype=dtype,
        )
        self.free = deque(range(pages))
        self.owners: dict[int, set[int]] = {}
        self.flash = None
        self.peak = 0
        self.use_kernel = False

    @property
    def used(self) -> int:
        return self.pages - len(self.free)

    @property
    def free_pages(self) -> int:
        return len(self.free)

    def extend(self, request: int, table: list[int], length: int) -> int:
        """Grow ``table`` until it covers ``length`` cached positions.

        Pages added before the pool runs dry stay with ``request``, so a retry after
        preemption resumes where this call stopped.
        """
        needed = -(-length // self.page_size)
        added = 0
        while len(table) < needed:
            if not self.free:
                raise PoolExhausted(f"KV pool holds {self.pages} pages, all {self.used} in use")
            page = self.free.popleft()
            self.owners[page] = {request}
            table.append(page)
            added += 1
        if added:
            self.peak = max(self.peak, self.used)
        return added

    def release(self, request: int, table: list[int]) -> None:
        for page in table:
            self.drop(request, page)
        table.clear()

    def retain(self, request, page):
        self.owners[page].add(request)

    def drop(self, request, page):
        owners = self.owners[page]
        owners.remove(request)
        if not owners:
            del self.owners[page]
            self.free.append(page)

    def make_buffers(self, batch: int, width: int, history: int) -> StepBuffers:
        device = self.device

        def long(*shape):
            return torch.empty(*shape, dtype=torch.long, device=device)

        return StepBuffers(
            ids=long(batch, width), positions=long(batch, width),
            valid=torch.empty(batch, width, dtype=torch.bool, device=device),
            write_slots=long(batch * width), write_rows=long(batch * width), write_cols=long(batch * width),
            history_slots=long(batch, history),
            causal=torch.empty(batch, 1, width, history, dtype=torch.bool, device=device),
            output_rows=long(batch), output_cols=long(batch),
            batch=batch, width=width, history=history,
        )

    def prepare(self, tables, starts, lengths, emit_rows) -> CacheBatch:
        """Allocate exact-size buffers for a one-off pass (used by tests and eager prefill)."""
        batch, width = len(tables), max(lengths)
        if self.flash is not None:
            return self.flash.prepare(tables, starts, lengths, emit_rows)
        buffers = self.make_buffers(batch, width, max(a + b for a, b in zip(starts, lengths)))
        return self.prepare_into(buffers, [[0] * width for _ in range(batch)], tables, starts, lengths, emit_rows)

    def prepare_into(self, buffers: StepBuffers, ids, tables, starts, lengths, emit_rows) -> CacheBatch:
        """Fill preallocated buffers in place; no device allocation, safe to replay under a graph.

        Only per-row scalars travel over PCIe: page tables are padded to the longest
        table in the batch, and the per-token slot list and causal mask are built on
        the device from them. Building either on the host costs O(batch * history)
        Python work per step, which dominates at long contexts.
        """
        batch, width, history = len(tables), buffers.width, buffers.history
        if batch != buffers.batch or max(lengths) > width:
            raise ValueError("Step buffers do not match the batch shape")
        limit = max(start + length for start, length in zip(starts, lengths))
        if history < limit:
            raise ValueError("Step buffers do not cover the history length")
        device, page = self.device, self.page_size

        write_slots, write_rows, write_cols = [], [], []
        for row, (table, start, length) in enumerate(zip(tables, starts, lengths)):
            positions = np.arange(start, start + length, dtype=np.int64)
            if length and len(table) * page < start + length:
                raise RuntimeError("Page table does not cover the written positions")
            write_slots.append(np.asarray(table, dtype=np.int64)[positions // page] * page + positions % page)
            write_rows.append(np.full(length, row, dtype=np.int64))
            write_cols.append(np.arange(length, dtype=np.int64))
        written = int(sum(lengths))

        def upload(target, values):
            target.copy_(torch.from_numpy(np.ascontiguousarray(values)), non_blocking=True)

        buffered = np.zeros((batch, width), dtype=np.int64)
        for row, chunk in enumerate(ids):
            buffered[row, :len(chunk)] = chunk
        upload(buffers.ids, buffered)
        upload(buffers.write_slots[:written], np.concatenate(write_slots))
        upload(buffers.write_rows[:written], np.concatenate(write_rows))
        upload(buffers.write_cols[:written], np.concatenate(write_cols))
        upload(buffers.output_rows[:len(emit_rows)], np.asarray(emit_rows, dtype=np.int64))
        upload(buffers.output_cols[:len(emit_rows)], np.asarray([lengths[row] - 1 for row in emit_rows],
                                                               dtype=np.int64))

        tables_max = max(len(table) for table in tables)
        padded = torch.zeros(batch, tables_max, dtype=torch.int32)
        for row, table in enumerate(tables):
            if table:
                padded[row, :len(table)] = torch.tensor(table, dtype=torch.int32)
        padded = padded.to(device, non_blocking=True)

        start_t = torch.tensor(starts, dtype=torch.long, device=device)
        count_t = torch.tensor(lengths, dtype=torch.long, device=device)
        buffers.positions.copy_(start_t[:, None] + torch.arange(width, device=device))
        buffers.valid.copy_(torch.arange(width, device=device)[None] < count_t[:, None])

        offsets = torch.arange(history, device=device)
        live = offsets < (start_t + count_t)[:, None]
        page_of_slot = (offsets // page).clamp(max=tables_max - 1)
        slots = padded.index_select(1, page_of_slot).long() * page + (offsets % page)
        buffers.history_slots.copy_(torch.where(live, slots, torch.zeros_like(slots)))

        logical = offsets[None, None, :]
        buffers.causal.copy_(
            ((logical <= buffers.positions[:, :, None]) & (logical < (start_t + count_t)[:, None, None]))[:, None]
        )
        return CacheBatch(
            buffers.positions, buffers.valid, buffers.write_slots[:written], buffers.write_rows[:written],
            buffers.write_cols[:written], buffers.history_slots, buffers.causal,
            buffers.output_rows[:len(emit_rows)], buffers.output_cols[:len(emit_rows)],
            padded, start_t + count_t,
        )

    def decode_attention(self, domain, layer, query, batch, groups):
        """Paged decode attention; None when this step or backend cannot use the kernel."""
        if self.flash is not None:
            return self.flash.run(domain, layer, query, batch)
        if not self.use_kernel or batch.positions.size(1) != 1 or batch.tables is None:
            return None
        from .kernels import paged_decode

        keys, values = self.tensor[domain, layer]
        return paged_decode(query, keys, values, batch.tables, batch.lengths, self.page_size, groups)

    def write(self, domain, layer, batch: CacheBatch, key, value):
        # Model tensors are [batch, heads, query, dimension]. Pool slots hold [heads, dimension].
        target = self.tensor[domain, layer]
        target[0].index_copy_(0, batch.write_slots, key.transpose(1, 2)[batch.write_rows, batch.write_cols])
        target[1].index_copy_(0, batch.write_slots, value.transpose(1, 2)[batch.write_rows, batch.write_cols])

    def read(self, domain, layer, batch: CacheBatch):
        keys, values = self.tensor[domain, layer]
        keys = keys[batch.history_slots]
        values = values[batch.history_slots]
        # A padding index can refer to stale/uninitialized memory. Clear it before SDPA.
        live = batch.causal[:, 0, -1, :, None, None]
        return keys.masked_fill(~live, 0).transpose(1, 2), values.masked_fill(~live, 0).transpose(1, 2)
