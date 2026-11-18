import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from loopinfer.cache import CacheSpec, PagePool
from loopinfer.paged import PagedAttention, unavailable


@pytest.mark.parametrize("lengths", [[1, 1], [3, 1]])
def test_flashinfer_paged_causal_alignment(lengths):
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dtype = torch.float16
    if reason := unavailable(device, dtype):
        pytest.skip(reason)
    torch.manual_seed(37)
    pool = PagePool(CacheSpec(1, 1, 2, 64), 8, 16, device, dtype)
    pool.tensor.normal_()
    tables, starts = [[], []], [15, 20]
    for row, (table, start, count) in enumerate(zip(tables, starts, lengths)):
        pool.extend(row, table, start + count)
    dense = pool.prepare(tables, starts, lengths, [0, 1])
    keys, values = pool.read(0, 0, dense)
    query = torch.randn(2, 4, max(lengths), 64, device=device, dtype=dtype)
    expected = scaled_dot_product_attention(
        query, keys.repeat_interleave(2, dim=1), values.repeat_interleave(2, dim=1), attn_mask=dense.causal,
    )
    pool.flash = PagedAttention(pool, 4)
    paged = pool.prepare(tables, starts, lengths, [0, 1])
    result = pool.decode_attention(0, 0, query, paged, 2)
    for index, count in enumerate(lengths):
        torch.testing.assert_close(result[index, :, :count], expected[index, :, :count], rtol=0.02, atol=0.003)
