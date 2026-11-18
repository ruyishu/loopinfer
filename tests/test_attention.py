import pytest
import torch
from torch.nn import functional as F

from loopinfer.kernels import available, paged_decode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not available("cuda"), reason="needs CUDA and Triton"
)


def test_paged_decode_matches_sdpa():
    torch.manual_seed(0)
    batch, heads, kv_heads, dim, page, pages, width = 2, 4, 2, 64, 16, 6, 33
    groups = heads // kv_heads
    device = torch.device("cuda")
    tokens = pages * page
    keys = torch.randn(tokens, kv_heads, dim, device=device)
    values = torch.randn(tokens, kv_heads, dim, device=device)
    query = torch.randn(batch, heads, 1, dim, device=device)
    fits = -(-width // page)
    tables = torch.stack([torch.randperm(pages, device=device)[:fits].to(torch.int32) for _ in range(batch)])
    lengths = torch.tensor([width, 17], device=device)

    got = paged_decode(query, keys, values, tables, lengths, page, groups)

    expected = []
    for row, length in enumerate(lengths.tolist()):
        slots = torch.cat([
            tables[row, index].long() * page + torch.arange(page, device=device)
            for index in range(-(-length // page))
        ])[:length]
        dense_k = keys[slots].transpose(0, 1).repeat_interleave(groups, dim=0)[None]
        dense_v = values[slots].transpose(0, 1).repeat_interleave(groups, dim=0)[None]
        expected.append(F.scaled_dot_product_attention(query[row: row + 1], dense_k, dense_v))
    expected = torch.cat(expected)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_paged_decode_rejects_multi_query():
    device = torch.device("cuda")
    with pytest.raises(ValueError):
        paged_decode(
            torch.zeros(1, 2, 2, 8, device=device), torch.zeros(4, 2, 8, device=device),
            torch.zeros(4, 2, 8, device=device), torch.zeros(1, 1, dtype=torch.int32, device=device),
            torch.tensor([4], device=device), 16, 1,
        )
