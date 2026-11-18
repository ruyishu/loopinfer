"""Paged decode attention: read KV straight from the page pool with Triton.

The gather-then-SDPA path materialises a dense ``[batch, heads, history, dim]``
tensor per layer per round; at long contexts that is several times the traffic of
the attention itself. This kernel indexes the page table in place, so each KV byte
is read once and no mask is built.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton ships with CUDA PyTorch builds
    triton = tl = None


def available(device) -> bool:
    return triton is not None and torch.device(device).type == "cuda"


if triton is not None:
    @triton.jit
    def _paged_decode(
        q_ptr, k_ptr, v_ptr, table_ptr, len_ptr, out_ptr,
        stride_qt, stride_qh, stride_kt, stride_kh, stride_ot, stride_oh, stride_tb,
        GROUPS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
        PAGE: tl.constexpr, BLOCK_N: tl.constexpr, SCALE: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        kv_head = head // GROUPS
        length = tl.load(len_ptr + row)
        offs_d = tl.arange(0, BLOCK_D)
        dim_mask = offs_d < HEAD_DIM
        q = tl.load(q_ptr + row * stride_qt + head * stride_qh + offs_d, mask=dim_mask, other=0.0)
        q = q.to(tl.float32) * SCALE
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        running_max = float("-inf")
        running_sum = 0.0
        for start in range(0, length, BLOCK_N):
            offs_n = start + tl.arange(0, BLOCK_N)
            live = offs_n < length
            pages = tl.load(table_ptr + row * stride_tb + offs_n // PAGE, mask=live, other=0)
            slots = pages * PAGE + offs_n % PAGE
            gather = slots[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :]
            keys = tl.load(k_ptr + gather, mask=live[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.where(live, tl.sum(keys * q[None, :], axis=1), float("-inf"))
            block_max = tl.max(scores, 0)
            new_max = tl.maximum(running_max, block_max)
            rescale = tl.exp2((running_max - new_max) * 1.4426950408889634)
            weights = tl.exp2((scores - new_max) * 1.4426950408889634)
            values = tl.load(v_ptr + gather, mask=live[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)
            running_sum = running_sum * rescale + tl.sum(weights, 0)
            acc = acc * rescale + tl.sum(weights[:, None] * values, axis=0)
            running_max = new_max
        tl.store(out_ptr + row * stride_ot + head * stride_oh + offs_d, (acc / running_sum).to(out_ptr.dtype.element_ty),
                 mask=dim_mask)


def paged_decode(query, keys, values, tables, lengths, page_size, groups):
    """query: [batch, heads, 1, dim]; keys/values: [tokens, kv_heads, dim]."""
    batch, heads, width, dim = query.shape
    if width != 1:
        raise ValueError("Paged decode attention expects one query row per request")
    out = torch.empty_like(query)
    block_d = 1 << (dim - 1).bit_length()
    _paged_decode[(batch, heads)](
        query, keys, values, tables, lengths, out,
        query.stride(0), query.stride(1), keys.stride(0), keys.stride(1), out.stride(0), out.stride(1),
        tables.stride(0),
        groups, dim, block_d, page_size, 64, 1.0 / math.sqrt(dim),
        num_warps=4, num_stages=2,
    )
    return out
