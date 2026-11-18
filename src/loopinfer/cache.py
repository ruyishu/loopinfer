from collections import deque
from dataclasses import dataclass

import torch


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


class PagePool:
    """Physical pages are shared across requests, never across live owners."""

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
        self.owners: dict[int, int] = {}

    def extend(self, request: int, table: list[int], length: int) -> None:
        count = max(0, (length + self.page_size - 1) // self.page_size - len(table))
        if count > len(self.free):
            raise RuntimeError("KV page reservation invariant violated")
        for _ in range(count):
            page = self.free.popleft()
            self.owners[page] = request
            table.append(page)

    def release(self, request: int, table: list[int]) -> None:
        for page in table:
            if self.owners.pop(page) != request:
                raise RuntimeError("KV page owner mismatch")
            self.free.append(page)
        table.clear()

    def prepare(self, tables, starts, lengths, emit_rows) -> CacheBatch:
        batch, width, history = len(tables), max(lengths), max(a + b for a, b in zip(starts, lengths))
        row_map, col_map, writes, row_history = [], [], [], []
        for row, (table, start, count) in enumerate(zip(tables, starts, lengths)):
            slots = [page * self.page_size + offset for page in table for offset in range(self.page_size)]
            writes.extend(slots[start:start + count])
            row_map.extend([row] * count)
            col_map.extend(range(count))
            row_history.append(slots[:start + count] + [0] * (history - start - count))

        def long(values):
            return torch.tensor(values, device=self.device, dtype=torch.long)

        positions = long(starts)[:, None] + torch.arange(width, device=self.device)
        counts = long(lengths)
        valid = torch.arange(width, device=self.device)[None] < counts[:, None]
        logical = torch.arange(history, device=self.device)[None, None]
        causal = (logical <= positions[:, :, None]) & (logical < (long(starts) + counts)[:, None, None])
        return CacheBatch(
            positions, valid, long(writes), long(row_map), long(col_map), long(row_history),
            causal[:, None], long(emit_rows), long([lengths[row] - 1 for row in emit_rows]),
        )

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
