"""Optional direct, packed paged attention backed by independent FlashInfer."""

import importlib.util
from itertools import accumulate

import torch

from .cache import CacheBatch


def unavailable(device, dtype=None):
    if device.type != "cuda":
        return "requires CUDA"
    if torch.cuda.get_device_capability(device)[0] < 8:
        return "requires NVIDIA SM80 or newer"
    if dtype not in (None, torch.float16, torch.bfloat16):
        return "requires float16 or bfloat16"
    if importlib.util.find_spec("flashinfer") is None:
        return "install loopinfer[flashinfer]"
    return None


class PagedAttention:
    def __init__(self, pool, heads):
        import flashinfer

        self.pool, self.heads = pool, heads
        self.workspace = torch.empty(128 * 1024**2, device=pool.device, dtype=torch.uint8)
        self.prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD", backend="fa2")
        self.decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace, "NHD", use_tensor_cores=True)

    def prepare(self, tables, starts, lengths, emit):
        page = self.pool.page_size
        width = max(lengths)
        device = self.pool.device

        def tensor(items, dtype=torch.int64):
            return torch.tensor(items, device=device, dtype=dtype)

        slots, rows, cols, pages, page_counts, tail = [], [], [], [], [], []
        for row, (table, start, count) in enumerate(zip(tables, starts, lengths)):
            end = start + count
            needed = (end + page - 1) // page
            pages.extend(table[:needed])
            page_counts.append(needed)
            tail.append((end - 1) % page + 1)
            for position in range(start, end):
                slots.append(table[position // page] * page + position % page)
                rows.append(row)
                cols.append(position - start)
        indptr = tensor([0, *accumulate(page_counts)], torch.int32)
        indices = tensor(pages, torch.int32)
        last = tensor(tail, torch.int32)
        spec = self.pool.spec
        dtype = self.pool.tensor.dtype
        dimensions = (self.heads, spec.heads, spec.head_dim, page)
        # Mixed-round groups need independent plans while sharing a single allocator.
        # Their plans are made on demand when switching groups inside a layer.
        positions = tensor(starts)[:, None] + torch.arange(width, device=device)
        batch = CacheBatch(
            positions, torch.arange(width, device=device)[None] < tensor(lengths)[:, None],
            tensor(slots), tensor(rows), tensor(cols), None, None,
            tensor(emit), tensor([lengths[row] - 1 for row in emit]),
        )
        batch.paged_plan = (indptr, indices, last, dimensions, dtype,
                            tensor([0, *accumulate(lengths)], torch.int32))
        return batch

    def run(self, domain, layer, query, batch):
        indptr, indices, last, dimensions, dtype, qo = batch.paged_plan
        decode = query.size(2) == 1
        wrapper = self.decode if decode else self.prefill
        if getattr(wrapper, "_loopinfer_batch", None) is not batch:
            options = {"q_data_type": dtype, "kv_data_type": dtype}
            if decode:
                wrapper.plan(indptr, indices, last, *dimensions, **options)
            else:
                wrapper.plan(qo, indptr, indices, last, *dimensions, causal=True, **options)
            wrapper._loopinfer_batch = batch
        keys, values = self.pool.tensor[domain, layer]
        page = self.pool.page_size
        keys = keys.view(self.pool.pages, page, *keys.shape[-2:])
        values = values.view(self.pool.pages, page, *values.shape[-2:])
        packed = query.transpose(1, 2)[batch.write_rows, batch.write_cols]
        output = wrapper.run(packed, (keys, values))
        result = torch.zeros_like(query.transpose(1, 2))
        result[batch.write_rows, batch.write_cols] = output
        return result.transpose(1, 2)
