"""Independent implementations of the published Ouro and dense-to-loop forward semantics."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..cache import CacheSpec


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden):
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(hidden.dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        self.head_dim, self.theta = head_dim, theta

    def forward(self, positions, dtype):
        exponents = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=positions.device)
        inverse = self.theta ** (-exponents / self.head_dim)
        phases = torch.cat((positions.float()[..., None] * inverse,) * 2, dim=-1)
        return phases.cos().to(dtype)[:, None], phases.sin().to(dtype)[:, None]


def rotate(hidden, rope):
    left, right = hidden.chunk(2, dim=-1)
    return hidden * rope[0] + torch.cat((-right, left), dim=-1) * rope[1]


class MLP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gate_proj = nn.Linear(model.width, model.intermediate_size, bias=False)
        self.up_proj = nn.Linear(model.width, model.intermediate_size, bias=False)
        self.down_proj = nn.Linear(model.intermediate_size, model.width, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Attention(nn.Module):
    def __init__(self, model, ouro):
        super().__init__()
        self.dim = model.head_dim
        self.q_proj = nn.Linear(model.width, model.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(model.width, model.kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(model.width, model.kv_heads * self.dim, bias=False)
        self.o_proj = nn.Linear(model.heads * self.dim, model.width, bias=False)
        self.q_norm = nn.Identity() if ouro else RMSNorm(self.dim, model.eps)
        self.k_norm = nn.Identity() if ouro else RMSNorm(self.dim, model.eps)
        self.groups = model.heads // model.kv_heads

    def forward(self, hidden, rope, pool, batch, domain, layer, write):
        shape = (*hidden.shape[:2], -1, self.dim)
        query = rotate(self.q_norm(self.q_proj(hidden).reshape(shape)).transpose(1, 2), rope)
        key = rotate(self.k_norm(self.k_proj(hidden).reshape(shape)).transpose(1, 2), rope)
        value = self.v_proj(hidden).reshape(shape).transpose(1, 2)
        if write:
            pool.write(domain, layer, batch, key, value)
            attended = pool.decode_attention(domain, layer, query, batch, self.groups)
            if attended is not None:
                return self.o_proj(attended.transpose(1, 2).reshape(*hidden.shape[:2], -1))
        keys, values = pool.read(domain, layer, batch)
        mask = batch.causal
        if not write:
            # Each row sees only its own transient KV from this round.
            keys = torch.cat((keys, key), dim=2)
            values = torch.cat((values, value), dim=2)
            self_mask = torch.eye(hidden.size(1), dtype=torch.bool, device=hidden.device)
            mask = torch.cat((mask, self_mask[None, None] & batch.valid[:, None, None]), dim=-1)
        if self.groups > 1:
            keys = keys.repeat_interleave(self.groups, dim=1)
            values = values.repeat_interleave(self.groups, dim=1)
        attended = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        return self.o_proj(attended.transpose(1, 2).reshape(*hidden.shape[:2], -1))


class DecoderLayer(nn.Module):
    def __init__(self, model, ouro):
        super().__init__()
        self.self_attn = Attention(model, ouro)
        self.mlp = MLP(model)
        self.input_layernorm = RMSNorm(model.width, model.eps)
        self.post_attention_layernorm = RMSNorm(model.width, model.eps)
        self.input_layernorm_2 = RMSNorm(model.width, model.eps) if ouro else nn.Identity()
        self.post_attention_layernorm_2 = RMSNorm(model.width, model.eps) if ouro else nn.Identity()

    def forward(self, hidden, rope, pool, batch, domain, layer, write):
        update = self.self_attn(self.input_layernorm(hidden), rope, pool, batch, domain, layer, write)
        hidden = hidden + self.input_layernorm_2(update)
        return hidden + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(hidden)))


class Core(nn.Module):
    def __init__(self, model, ouro):
        super().__init__()
        self.embed_tokens = nn.Embedding(model.vocab_size, model.width)
        self.layers = nn.ModuleList(DecoderLayer(model, ouro) for _ in range(model.layer_count))
        self.norm = RMSNorm(model.width, model.eps)
        self.rotary = Rotary(model.head_dim, model.theta)
        if ouro:
            self.early_exit_gate = nn.Linear(model.width, 1)

    def round(self, hidden, rope, pool, batch, domain, write=True):
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, rope, pool, batch, domain, index, write)
        return self.norm(hidden)


class NativeModel(nn.Module):
    supported_types = ()

    def __init__(self, config, loops, default_loops):
        super().__init__()
        if (
            config.get("hidden_act", "silu") != "silu"
            or config.get("rope_scaling")
            or (config.get("rope_parameters") or {}).get("rope_type", "default") != "default"
            or config.get("use_sliding_window")
            or config.get("attention_bias")
            or any(item != "full_attention" for item in (config.get("layer_types") or ()))
        ):
            raise ValueError("Only full attention, default RoPE, bias-free projections and SiLU are supported")
        self.width = int(config["hidden_size"])
        self.intermediate_size = int(config["intermediate_size"])
        self.vocab_size = int(config["vocab_size"])
        self.layer_count = int(config["num_hidden_layers"])
        self.heads = int(config["num_attention_heads"])
        self.kv_heads = int(config.get("num_key_value_heads", self.heads))
        self.head_dim = int(config.get("head_dim", self.width // self.heads))
        self.max_positions = int(config["max_position_embeddings"])
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.theta = float(config.get("rope_theta", (config.get("rope_parameters") or {}).get("rope_theta", 10000)))
        self.loops = default_loops if loops is None else loops
        if min(self.layer_count, self.width, self.vocab_size, self.heads, self.kv_heads, self.head_dim,
               self.max_positions, self.intermediate_size) < 1 or self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("Invalid model dimensions")
        if not 1 <= self.loops <= default_loops:
            raise ValueError(f"loops must be between 1 and {default_loops}")

    def selected(self, hidden, batch):
        return hidden[batch.output_rows, batch.output_cols]


class Ouro(NativeModel):
    supported_types = ("ouro",)

    def __init__(self, config, loops=None):
        super().__init__(config, loops, int(config.get("total_ut_steps", 4)))
        self.model = Core(self, ouro=True)
        self.lm_head = nn.Linear(self.width, self.vocab_size, bias=False)
        if config.get("tie_word_embeddings"):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.cache_spec = CacheSpec(self.loops, self.layer_count, self.kv_heads, self.head_dim)

    def forward(self, ids, batch, pool, *, trace=None):
        hidden = self.model.embed_tokens(ids)
        rope = self.model.rotary(batch.positions, hidden.dtype)
        for depth in range(self.loops):
            hidden = self.model.round(hidden, rope, pool, batch, depth)
            if trace is not None:
                trace.append(self.selected(hidden, batch))
        return self.lm_head(self.selected(hidden, batch)).float()


class InputUpdater(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.embedding_norm = RMSNorm(width, eps)
        self.hidden_norm = RMSNorm(width, eps)
        self.input_norm = RMSNorm(width, eps)
        self.output_norm = RMSNorm(width, eps)
        self.fuse = nn.Linear(width * 2, width, bias=False)
        self.gate = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, width, bias=False)
        self.down = nn.Linear(width, width, bias=False)

    def forward(self, embedding, hidden):
        value = self.fuse(torch.cat((self.embedding_norm(embedding), self.hidden_norm(hidden)), dim=-1))
        value = self.input_norm(value)
        return self.output_norm(self.down(F.silu(self.gate(value)) * self.up(value)))


class DenseBackbone(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = Core(model, ouro=False)
        self.lm_head = nn.Linear(model.width, model.vocab_size, bias=False)
        if config.get("tie_word_embeddings"):
            self.lm_head.weight = self.model.embed_tokens.weight


class DenseToLoop(NativeModel):
    supported_types = ("dense-to-loop",)

    def __init__(self, config, loops=None):
        super().__init__(config, loops, 2)
        self.backbone = DenseBackbone(self, config)
        self.updater = InputUpdater(self.width, self.eps)
        self.cache_spec = CacheSpec(1, self.layer_count, self.kv_heads, self.head_dim)
        self.register_buffer("valid_vocab", torch.ones(self.vocab_size, dtype=torch.bool), persistent=False)

    def set_tokenizer(self, tokenizer):
        ids = list(tokenizer.get_vocab().values())
        if not ids or min(ids) < 0 or max(ids) >= self.vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds checkpoint vocabulary")
        device = self.backbone.lm_head.weight.device
        self.valid_vocab = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        self.valid_vocab[torch.tensor(ids, device=device)] = True

    def forward(self, ids, batch, pool, *, trace=None):
        core = self.backbone.model
        embedding = core.embed_tokens(ids)
        rope = core.rotary(batch.positions, embedding.dtype)
        first = core.round(embedding, rope, pool, batch, 0)
        selected = self.selected(first, batch)
        first_logp = self.backbone.lm_head(selected).float().masked_fill(~self.valid_vocab, -torch.inf).log_softmax(-1)
        if trace is not None:
            trace.append(selected)
        if self.loops == 1:
            return first_logp
        second = core.round(self.updater(embedding, first), rope, pool, batch, 0, write=False)
        selected = self.selected(second, batch)
        if trace is not None:
            trace.append(selected)
        second_logp = self.backbone.lm_head(selected).float().masked_fill(~self.valid_vocab, -torch.inf).log_softmax(-1)
        return torch.logaddexp(first_logp, second_logp) - math.log(2)
