"""RoPE Transformer for the Sorted Translation task, RYS-compatible.

The model is an encoder-only stack that reads an unsorted sequence and predicts
all sorted tokens in parallel (one ``vocab``-way classification per position).
It is deliberately built so that:

- the input embedding ``W_E`` consumes a one-hot vector and the output head
  ``W_U`` is a separate ``d_model -> vocab`` matrix (no weight tying), isolating
  a large distinct decoding head;
- position is encoded only with Rotary Positional Embeddings (RoPE) applied to
  the per-head query/key vectors, so the model extrapolates to longer
  out-of-distribution sequence lengths with no learned positional table;
- every layer is one entry of ``model.model.layers`` and each round runs
  ``forward(hidden, *, attention_mask=None) -> (hidden,)``, so the existing
  :func:`rys.surgery.apply_rys` hook replays windows unchanged.

With ``pre_norm=True`` each round is purely additive (``h <- h + F(LN(h))`` for
both sublayers), so the telescoping identity ``x_j = x_i + S_ij`` holds and the
rho/phi CKA decomposition applies exactly. ``weight_tied`` shares one round
across depth, turning the stack into an iterated map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SortedTranslationConfig:
    """Configuration for :class:`SortedTranslationTransformer`."""

    vocab_in: int = 128
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_mlp: int = 512
    dropout: float = 0.0
    pre_norm: bool = True
    weight_tied: bool = False
    rope_base: float = 10000.0


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` tables of shape ``(seq_len, head_dim)``."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim.")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)  # (seq_len, half)
    emb = torch.cat([angles, angles], dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding. ``x`` is ``(B, H, S, head_dim)``; tables ``(S, head_dim)``."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class RoPESelfAttentionRound(nn.Module):
    """One pre-norm self-attention + MLP round with rotary positions."""

    def __init__(self, config: SortedTranslationConfig) -> None:
        super().__init__()
        d = config.d_model
        self.pre_norm = config.pre_norm
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        if self.head_dim * config.n_heads != d:
            raise ValueError("d_model must be divisible by n_heads.")
        # NoPE (no positional encoding) and causal masking are opt-in via config;
        # defaults keep the original bidirectional-RoPE behaviour for callers
        # whose config predates these fields.
        self.use_rope = getattr(config, "use_rope", True)
        self.causal = getattr(config, "causal", False)
        if self.use_rope and self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.rope_base = config.rope_base
        self.attn_norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, config.d_mlp),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_mlp, d),
        )

    def _attention(self, h: torch.Tensor) -> torch.Tensor:
        b, s, d = h.shape
        x = self.attn_norm(h) if self.pre_norm else h
        q = self.q(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            cos, sin = _build_rope_cache(s, self.head_dim, base=self.rope_base, device=h.device, dtype=h.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, H, S, S)
        if self.causal:
            causal_mask = torch.ones(s, s, device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = self.attn_dropout(scores.softmax(dim=-1))
        ctx = attn @ v  # (B, H, S, head_dim)
        ctx = ctx.transpose(1, 2).reshape(b, s, d)
        return self.out(ctx)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor]:
        if self.pre_norm:
            hidden = hidden + self._attention(hidden)
            hidden = hidden + self.mlp(self.mlp_norm(hidden))
        else:
            hidden = self.attn_norm(hidden + self._attention(hidden))
            hidden = self.mlp_norm(hidden + self.mlp(hidden))
        return (hidden,)


class SortedTranslationBackbone(nn.Module):
    """Backbone exposing ``layers`` for RYS replay over sorting rounds."""

    def __init__(self, config: SortedTranslationConfig) -> None:
        super().__init__()
        self.config = config
        # W_E consumes a one-hot input vector; no bias and no positional table.
        self.embed = nn.Linear(config.vocab_in, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        if config.weight_tied:
            shared = RoPESelfAttentionRound(config)
            self.layers = nn.ModuleList(shared for _ in range(config.n_layers))
        else:
            self.layers = nn.ModuleList(RoPESelfAttentionRound(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        one_hot = F.one_hot(input_ids, num_classes=self.config.vocab_in).to(self.embed.weight.dtype)
        hidden = self.dropout(self.embed(one_hot))
        trace: list[torch.Tensor] = []
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=None)[0]
            if return_hidden_states:
                trace.append(hidden.clone())  # raw pre-final-norm states preserve telescoping
        return self.norm(hidden), (trace if return_hidden_states else None)


class SortedTranslationTransformer(nn.Module):
    """RYS-compatible encoder-only parallel sorter.

    ``self.model.layers`` mirrors the Llama-style layout so
    :func:`rys.surgery.apply_rys` can replay windows. The classification head
    ``unembed`` (``W_U``) is a distinct ``d_model -> vocab`` matrix and is never
    tied to the input embedding ``W_E``.
    """

    def __init__(self, config: SortedTranslationConfig) -> None:
        super().__init__()
        self.config = config
        self.model = SortedTranslationBackbone(config)
        self.unembed = nn.Linear(config.d_model, config.vocab_in)

    def _readout(self, states: torch.Tensor) -> torch.Tensor:
        return self.unembed(states)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        target_ids: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_round_logits: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        states, trace = self.model(
            input_ids,
            return_hidden_states=return_hidden_states or return_round_logits,
        )
        logits = self._readout(states)
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_in),
                target_ids.reshape(-1),
            )
        round_logits = None
        if return_round_logits and trace is not None:
            round_logits = [self._readout(self.model.norm(s)) for s in trace]
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": trace,
            "round_logits": round_logits,
        }
