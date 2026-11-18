import torch

from loopinfer.cache import CacheSpec, PagePool


def test_prepare_into_matches_prepare():
    spec = CacheSpec(domains=2, layers=3, heads=2, head_dim=4)
    pool = PagePool(spec, pages=8, page_size=2, device="cpu", dtype=torch.float32)
    tables = [[], []]
    pool.extend(0, tables[0], 3)
    pool.extend(1, tables[1], 3)
    starts, lengths, emit = [0, 1], [3, 3], [0, 1]

    eager = pool.prepare(tables, starts, lengths, emit)
    buffers = pool.make_buffers(2, 3, 8)
    lazy = pool.prepare_into(buffers, [[5, 6, 7], [8, 9, 0]], tables, starts, lengths, emit)

    assert torch.equal(eager.positions, lazy.positions)
    assert torch.equal(eager.valid, lazy.valid)
    assert torch.equal(eager.write_slots, lazy.write_slots)
    assert torch.equal(eager.write_rows, lazy.write_rows)
    assert torch.equal(eager.write_cols, lazy.write_cols)
    assert torch.equal(eager.output_rows, lazy.output_rows)
    assert torch.equal(eager.output_cols, lazy.output_cols)
    assert torch.equal(eager.history_slots, lazy.history_slots[:, :eager.history_slots.size(1)])
    assert torch.equal(eager.causal, lazy.causal[..., :eager.causal.size(-1)])
    assert buffers.ids.tolist() == [[5, 6, 7], [8, 9, 0]]


def test_prepare_into_rejects_shape_mismatch():
    spec = CacheSpec(domains=1, layers=1, heads=1, head_dim=2)
    pool = PagePool(spec, pages=4, page_size=2, device="cpu", dtype=torch.float32)
    table = []
    pool.extend(0, table, 2)
    buffers = pool.make_buffers(1, 1, 2)
    try:
        pool.prepare_into(buffers, [[1, 2]], [table], [0], [2], [0])
    except ValueError:
        return
    raise AssertionError("expected a shape mismatch")
