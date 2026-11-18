from concurrent.futures import ThreadPoolExecutor

import pytest

from loopinfer import Engine, Sampling, load_model
from loopinfer.bench import make_trace
from loopinfer.models import Ouro


@pytest.mark.parametrize("scheduler", ["bucket", "auto", "round"])
def test_depth_scheduler_matches_individual_depths(checkpoint, scheduler):
    if checkpoint.name != "ouro":
        pytest.skip("Ouro depth contract")
    path = str(checkpoint)
    samples = [Sampling(max_tokens=6, ignore_eos=True, loop_depth=depth) for depth in (2, 4)]
    with Engine(path, device="cpu", scheduler="bucket") as engine:
        expected = [engine.submit("a b c", sample).wait(10).token_ids for sample in samples]
    with Engine(path, device="cpu", scheduler=scheduler) as engine:
        with engine._condition:
            pending = [engine.submit("a b c", sample) for sample in samples]
        assert [row.wait(10).token_ids for row in pending] == expected
        assert engine.stats()["kv_pages_used"] == 0
        if scheduler != "bucket":
            assert engine.stats()["mixed_round_batches"] > 0


def test_prefix_reuse_and_isolation(checkpoint):
    if checkpoint.name != "ouro":
        pytest.skip("Ouro prefix semantics")
    with Engine(str(checkpoint), device="cpu", page_size=2, prefill_chunk=3, prefix_cache_pages=8) as engine:
        prompt = [3, 4, 5, 6, 7, 8, 9]
        sample = Sampling(max_tokens=3, ignore_eos=True, loop_depth=2)
        first = engine.submit(prompt, sample, cache_namespace="a").wait(10)
        second = engine.submit(prompt, sample, cache_namespace="a").wait(10)
        assert first.token_ids == second.token_ids
        assert engine.stats()["prefix_hit_tokens"] == 6
        engine.submit(prompt, sample, cache_namespace="b").wait(10)
        engine.submit(prompt, Sampling(max_tokens=3, loop_depth=4), cache_namespace="a").wait(10)
        assert engine.stats()["prefix_hit_tokens"] == 6


def test_streaming_preemption_replays_committed_tokens(checkpoint):
    loaded = load_model(str(checkpoint), device="cpu")
    page_bytes = 2 * loaded.model.cache_spec.token_bytes(loaded.dtype)
    sample = Sampling(max_tokens=8, temperature=0.8, seed=42, ignore_eos=True)
    prompt = [3, 4, 5]
    with Engine(loaded, context=12, page_size=2, max_batch=2) as engine:
        expected = engine.submit(prompt, sample).wait(10).token_ids
    with Engine(loaded, context=12, page_size=2, max_batch=2, kv_mib=5 * page_bytes / 1024**2) as engine:
        with engine._condition:
            first = engine.submit(prompt, sample, stream=True)
            second = engine.submit(prompt, sample, stream=True)
        with ThreadPoolExecutor(2) as pool:
            handles = list(pool.map(lambda h: tuple(event.token_id for event in h.consume(10)), (first, second)))
        assert handles == [expected, expected]
        assert engine.stats()["preemptions"] > 0
        assert engine.stats()["replayed_tokens"] > 0
        assert engine.stats()["kv_pages_used"] == 0


def test_ouro_greedy_variants(checkpoint):
    if checkpoint.name != "ouro":
        pytest.skip("Ouro depth contract")
    path = str(checkpoint)
    sample = Sampling(max_tokens=12, ignore_eos=True)
    with Engine(path, device="cpu") as engine:
        expected = engine.submit("a b c", sample).wait(10).token_ids
        shallow = engine.submit("a b c", Sampling(max_tokens=1, loop_depth=1)).wait(10)
        gated = engine.submit("a b c", Sampling(max_tokens=1, exit_threshold=0.0001)).wait(10)
        assert shallow.token_ids == gated.token_ids
    with Engine(path, device="cpu", fuse_projections=True, speculate_tokens=4) as engine:
        assert isinstance(engine.model, Ouro)
        assert engine.submit("a b c", sample).wait(10).token_ids == expected
        assert engine.stats()["speculative_steps"] > 0


def test_benchmark_workload_reproducibility():
    options = dict(input_tokens=8, vocab_size=32, shared_prefix=4, request_rate=2)
    trace = make_trace("mixed-depth", 20, **options)
    assert trace == make_trace("mixed-depth", 20, **options)
    assert set(item["loop_depth"] for item in trace) == {2, 4}
    assert len({tuple(item["prompt"][:4]) for item in trace}) == 1
    assert len({tuple(item["prompt"][4:]) for item in trace}) == len(trace)
    assert sorted(item["arrival_s"] for item in trace) == [item["arrival_s"] for item in trace]
