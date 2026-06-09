"""Constraint-propagation solver for graph k-colouring, RYS-compatible.

Parallel to :mod:`rys.sat_message_passing` and
:mod:`rys.nqueens_message_passing`.  The colouring constraint graph is the input
graph itself, so each round is one **adjacency-masked** multi-head attention +
MLP update over the vertex states: a vertex aggregates information from its
neighbours and refines its own colour belief.

Every round is one entry of ``model.model.layers`` so
:func:`rys.surgery.apply_rys` and the CKA capture utilities work unchanged.
With ``pre_norm=True`` the vertex residual stream is purely additive
(``h <- h + F(LN(h))`` for both sublayers), so the telescoping identity
``x_j = x_i + S_ij`` holds and the rho/phi CKA decomposition applies exactly.
``weight_tied`` shares one round across depth, turning the stack into an
iterated map for the deep-equilibrium reading.

The solver carries **no absolute vertex identity**: a vertex is described only
by its given colour, its is-given flag, and its degree (a permutation-invariant
symmetry-breaking feature).  This keeps the model permutation-equivariant and,
with a fixed colour vocabulary, lets it generalise to larger graphs out of
distribution without the untrained-vocabulary collapse seen on N-Queens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ColoringMPConfig:
    """Configuration for :class:`ColoringMessagePassingModel`."""

    max_v: int
    n_colors: int = 3
    max_degree: int = 16
    d_model: int = 64
    n_rounds: int = 8
    n_heads: int = 4
    d_mlp: int = 128
    dropout: float = 0.0
    pre_norm: bool = True
    weight_tied: bool = False


@dataclass
class _GraphContext:
    """Per-batch attention mask shared by every message-passing round."""

    attn_mask: torch.Tensor  # (B, 1, V, V) bool: query may attend to key


class ColoringRound(nn.Module):
    """One adjacency-masked multi-head attention + MLP update over vertices."""

    def __init__(self, config: ColoringMPConfig) -> None:
        super().__init__()
        d = config.d_model
        self.pre_norm = config.pre_norm
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        if self.head_dim * config.n_heads != d:
            raise ValueError("d_model must be divisible by n_heads.")
        self.attn_norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, config.d_mlp),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_mlp, d),
        )
        self.ctx: _GraphContext | None = None

    def _attention(self, h: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("Graph context was not set on the round.")
        b, n, d = h.shape
        x = self.attn_norm(h) if self.pre_norm else h
        q = self.q(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, H, V, V)
        scores = scores.masked_fill(~ctx.attn_mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        ctx_heads = attn @ v  # (B, H, V, head_dim)
        ctx_flat = ctx_heads.transpose(1, 2).reshape(b, n, d)
        return self.out(ctx_flat)

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


class ColoringBackbone(nn.Module):
    """Backbone exposing ``layers`` for RYS replay over colouring rounds."""

    def __init__(self, config: ColoringMPConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.given_embedding = nn.Embedding(config.n_colors + 1, d)  # last index = unknown
        self.is_given_embedding = nn.Embedding(2, d)
        self.degree_embedding = nn.Embedding(config.max_degree + 1, d)
        self.embed_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        if config.weight_tied:
            shared = ColoringRound(config)
            self.layers = nn.ModuleList(shared for _ in range(config.n_rounds))
        else:
            self.layers = nn.ModuleList(ColoringRound(config) for _ in range(config.n_rounds))
        self.norm = nn.LayerNorm(d)

    def embed(self, given_color: torch.Tensor, is_given: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        deg = degree.clamp(max=self.config.max_degree)
        h = (
            self.given_embedding(given_color)
            + self.is_given_embedding(is_given.long())
            + self.degree_embedding(deg)
        )
        return self.dropout(self.embed_norm(h))

    def build_context(self, adjacency: torch.Tensor, vertex_mask: torch.Tensor) -> _GraphContext:
        # A vertex may attend to its real neighbours; the always-on self-loop
        # keeps the softmax well-defined for isolated and padded vertices alike.
        eye = torch.eye(adjacency.shape[-1], dtype=torch.bool, device=adjacency.device)
        allowed = adjacency & vertex_mask.unsqueeze(1) & vertex_mask.unsqueeze(2)
        allowed = allowed | eye
        return _GraphContext(attn_mask=allowed.unsqueeze(1))  # (B, 1, V, V)

    def forward(
        self,
        *,
        given_color: torch.Tensor,
        is_given: torch.Tensor,
        degree: torch.Tensor,
        adjacency: torch.Tensor,
        vertex_mask: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        hidden = self.embed(given_color, is_given, degree)
        context = self.build_context(adjacency, vertex_mask)
        trace: list[torch.Tensor] = []
        for layer in self.layers:
            layer.ctx = context
            hidden = layer(hidden, attention_mask=None)[0]
            if return_hidden_states:
                trace.append(hidden.clone())  # raw pre-final-norm states preserve telescoping
        return self.norm(hidden), (trace if return_hidden_states else None)


class ColoringMessagePassingModel(nn.Module):
    """RYS-compatible graph k-colouring completion solver."""

    def __init__(self, config: ColoringMPConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ColoringBackbone(config)
        self.color_head = nn.Linear(config.d_model, config.n_colors)

    def forward(
        self,
        *,
        given_color: torch.Tensor,
        is_given: torch.Tensor,
        degree: torch.Tensor,
        adjacency: torch.Tensor,
        vertex_mask: torch.Tensor,
        return_hidden_states: bool = False,
        return_round_logits: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        states, trace = self.model(
            given_color=given_color,
            is_given=is_given,
            degree=degree,
            adjacency=adjacency,
            vertex_mask=vertex_mask,
            return_hidden_states=return_hidden_states or return_round_logits,
        )
        logits = self.color_head(states)
        round_logits = None
        if return_round_logits and trace is not None:
            round_logits = [self.color_head(self.model.norm(s)) for s in trace]
        return {"logits": logits, "hidden_states": trace, "round_logits": round_logits, "loss": None}
