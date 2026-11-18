import math

import pytest
import torch
from torch.nn import functional as F

from loopinfer import Engine, Sampling, load_model
from loopinfer.cache import PagePool
from loopinfer.models import DenseToLoop


@torch.inference_mode()
def uncached_reference(model, ids):
    """Full-sequence reference using explicit scores. It performs no pool operations."""
    dense = isinstance(model, DenseToLoop)
    core = model.backbone.model if dense else model.model
    head = model.backbone.lm_head if dense else model.lm_head
    embeddings = core.embed_tokens(ids)
    x = embeddings
    causal = torch.ones(ids.size(1), ids.size(1), dtype=torch.bool).tril()
    phase = torch.outer(torch.arange(ids.size(1)).float(), model.theta ** (
        -torch.arange(0, model.head_dim, 2).float() / model.head_dim
    ))
    phase = torch.cat([phase, phase], dim=-1)

    def proj(vector, linear, norm, rotary):
        y = norm(linear(vector).reshape(1, len(phase), -1, model.head_dim)).transpose(1, 2)
        if rotary:
            a, b = y.chunk(2, dim=-1)
            y = y * phase.cos() + torch.cat([-b, a], -1) * phase.sin()
        return y

    history, rounds = [], []
    for round_index in range(model.loops):
        if dense and round_index:
            x = model.updater(embeddings, x)
        for index, layer in enumerate(core.layers):
            a = layer.self_attn
            z = layer.input_layernorm(x)
            q = proj(z, a.q_proj, a.q_norm, True)
            k = proj(z, a.k_proj, a.k_norm, True)
            v = proj(z, a.v_proj, torch.nn.Identity(), False)
            mask = causal
            if dense and round_index == 0:
                history.append((k, v))
            elif dense:
                old_k, old_v = history[index]
                k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
                mask = torch.cat((causal, torch.eye(ids.size(1), dtype=torch.bool)), dim=1)
            k = k.repeat_interleave(a.groups, dim=1)
            v = v.repeat_interleave(a.groups, dim=1)
            score = (q @ k.transpose(-2, -1)) / math.sqrt(model.head_dim)
            attended = (score.masked_fill(~mask, -torch.inf).softmax(-1) @ v).transpose(1, 2).flatten(2)
            x = x + layer.input_layernorm_2(a.o_proj(attended))
            x = x + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(x)))
        x = core.norm(x)
        rounds.append(x)
    if not dense:
        return head(x)[0, -1].float(), rounds
    readouts = [F.log_softmax(head(h).masked_fill(~model.valid_vocab, -torch.inf), -1) for h in rounds]
    mixed = torch.stack(readouts).logsumexp(0) - math.log(len(rounds))
    return mixed[0, -1].float(), rounds


def test_cached_forward_and_reference(checkpoint):
    loaded = load_model(str(checkpoint), device="cpu")
    model = loaded.model
    pool = PagePool(model.cache_spec, 8, 2, loaded.device, loaded.dtype)
    ids = torch.tensor([[3, 4, 3, 5, 6]])
    table = []
    with torch.inference_mode():
        for start, stop in [(0, 3), (3, 4), (4, 5)]:
            pool.extend(0, table, stop)
            inputs = pool.prepare([table], [start], [stop - start], [0])
            rounds = []
            actual = model(ids[:, start:stop], inputs, pool, trace=rounds)[0]
            expected, states = uncached_reference(model, ids[:, :stop])
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            for a, b in zip(rounds, states, strict=True):
                torch.testing.assert_close(a, b[:, -1], atol=1e-5, rtol=1e-4)
    pool.release(0, table)
    assert not pool.owners


def test_mixed_batch_matches_individual(checkpoint):
    texts = ["a b c d e", "b", "a a g f c d", "a b"]
    opts = {"context": 12, "page_size": 2, "prefill_chunk": 2, "device": "cpu"}
    sample = Sampling(max_tokens=4, ignore_eos=True, temperature=.9, top_p=.95, top_k=5, seed=3)
    with Engine(str(checkpoint), max_batch=1, **opts) as engine:
        expected = engine.generate(texts, sample)
    with Engine(str(checkpoint), max_batch=4, **opts) as engine:
        actual = engine.generate(list(reversed(texts)), sample)
        assert [r.token_ids for r in expected] == [r.token_ids for r in reversed(actual)]
        assert engine.stats()["kv_pages_used"] == engine.stats()["kv_pages_reserved"] == 0
        with pytest.raises(ValueError, match="context limit"):
            engine.submit([3] * 13, sample)


def test_cancel_and_pool_reuse(checkpoint):
    with Engine(str(checkpoint), context=16, page_size=2, device="cpu", max_batch=1) as engine:
        handle = engine.submit("a b", Sampling(max_tokens=14, ignore_eos=True), stream=True)
        for _ in handle.consume(timeout=5):
            handle.cancel()
            break
        assert handle.done.wait(5)
        later = engine.submit("b a", Sampling(max_tokens=4, ignore_eos=True))
        assert len(later.wait(5).token_ids) == 4
        assert engine.stats()["kv_pages_used"] == 0


def test_budget_and_slow_stream_do_not_strand_worker(checkpoint):
    with Engine(str(checkpoint), context=12, page_size=2, stream_buffer=1, device="cpu") as engine:
        handle = engine.submit("a b", Sampling(max_tokens=5, ignore_eos=True), stream=True)
        with pytest.raises(RuntimeError, match="fell behind"):
            handle.wait(5)
        assert engine.stats()["kv_pages_reserved"] == 0
        assert engine.generate("a", Sampling(max_tokens=2, ignore_eos=True)).finish_reason == "length"
