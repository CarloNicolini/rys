"""Constraint-propagation solver for N-Queens, RYS-compatible.

Parallel to :mod:`rys.sat_message_passing`.  The N-Queens constraint graph is a
*complete* graph over row nodes whose edges carry a relative row offset (the
offset is what makes diagonal constraints meaningful), so each round is one
relative-position attention + MLP update over the ``N`` row states.

Every round is one entry of ``model.model.layers`` so :func:`rys.surgery.apply_rys`
and the CKA capture utilities work unchanged.  With ``pre_norm=True`` the row
residual stream is purely additive (``h <- h + F(LN(h))`` for both sublayers),
so the telescoping identity ``x_j = x_i + S_ij`` holds and the rho/phi CKA
decomposition applies exactly.  ``weight_tied`` shares one round across depth,
turning the stack into an iterated map for the deep-equilibrium reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class QueensMPConfig:
    """Configuration for :class:`QueensMessagePassingModel`."""

    max_n: int
    d_model: int = 64
    n_rounds: int = 8
    n_heads: int = 4
    d_mlp: int = 128
    dropout: float = 0.0
    pre_norm: bool = True
    weight_tied: bool = False


@dataclass
class _RowContext:
    row_mask: torch.Tensor  # (B, R) bool


class QueensRound(nn.Module):
    """One relative-position attention + MLP update over row states."""

    def __init__(self, config: QueensMPConfig) -> None:
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
        # Relative row-offset tables: index = (r - s) + (max_n - 1) in [0, 2 max_n - 2].
        self.rel_bias = nn.Embedding(2 * config.max_n - 1, config.n_heads)
        self.rel_val = nn.Embedding(2 * config.max_n - 1, d)
        offsets = torch.arange(config.max_n)
        idx = (offsets[:, None] - offsets[None, :]) + (config.max_n - 1)
        self.register_buffer("offset_idx", idx, persistent=False)
        self.mlp_norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, config.d_mlp),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_mlp, d),
        )
        self.ctx: _RowContext | None = None

    def _attention(self, h: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("Row context was not set on the round.")
        b, r, d = h.shape
        x = self.attn_norm(h) if self.pre_norm else h
        q = self.q(x).view(b, r, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, r, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, r, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, H, R, R)
        rel_b = self.rel_bias(self.offset_idx).permute(2, 0, 1).unsqueeze(0)  # (1, H, R, R)
        scores = scores + rel_b
        key_mask = ctx.row_mask[:, None, None, :]  # (B,1,1,R)
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        ctx_heads = attn @ v  # (B, H, R, head_dim)
        ctx_flat = ctx_heads.transpose(1, 2).reshape(b, r, d)
        # Relative value injection: sum_s attn[r,s] * rel_val(r-s), head-averaged.
        rel_v = self.rel_val(self.offset_idx)  # (R, R, d)
        attn_mean = attn.mean(dim=1)  # (B, R, R)
        rel_term = torch.einsum("brs,rsd->brd", attn_mean, rel_v)
        return self.out(ctx_flat + rel_term)

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


class QueensBackbone(nn.Module):
    """Backbone exposing ``layers`` for RYS replay over constraint rounds."""

    def __init__(self, config: QueensMPConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.given_embedding = nn.Embedding(config.max_n + 1, d)  # last index = unknown
        self.is_given_embedding = nn.Embedding(2, d)
        self.row_embedding = nn.Embedding(config.max_n, d)
        self.embed_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        if config.weight_tied:
            shared = QueensRound(config)
            self.layers = nn.ModuleList(shared for _ in range(config.n_rounds))
        else:
            self.layers = nn.ModuleList(QueensRound(config) for _ in range(config.n_rounds))
        self.norm = nn.LayerNorm(d)

    def embed(self, given_col: torch.Tensor, is_given: torch.Tensor) -> torch.Tensor:
        b, r = given_col.shape
        rows = torch.arange(r, device=given_col.device).unsqueeze(0).expand(b, r)
        h = (
            self.given_embedding(given_col)
            + self.is_given_embedding(is_given.long())
            + self.row_embedding(rows)
        )
        return self.dropout(self.embed_norm(h))

    def forward(
        self,
        *,
        given_col: torch.Tensor,
        is_given: torch.Tensor,
        row_mask: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        hidden = self.embed(given_col, is_given)
        context = _RowContext(row_mask=row_mask)
        trace: list[torch.Tensor] = []
        for layer in self.layers:
            layer.ctx = context
            hidden = layer(hidden, attention_mask=None)[0]
            if return_hidden_states:
                trace.append(hidden.clone())  # raw pre-final-norm states preserve telescoping
        return self.norm(hidden), (trace if return_hidden_states else None)


class QueensMessagePassingModel(nn.Module):
    """RYS-compatible N-Queens completion solver."""

    def __init__(self, config: QueensMPConfig) -> None:
        super().__init__()
        self.config = config
        self.model = QueensBackbone(config)
        self.column_head = nn.Linear(config.d_model, config.max_n)

    def _readout(self, states: torch.Tensor, col_mask: torch.Tensor | None) -> torch.Tensor:
        logits = self.column_head(states)
        if col_mask is not None:
            logits = logits.masked_fill(~col_mask[:, None, :], -1e9)
        return logits

    def forward(
        self,
        *,
        given_col: torch.Tensor,
        is_given: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_round_logits: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        states, trace = self.model(
            given_col=given_col,
            is_given=is_given,
            row_mask=row_mask,
            return_hidden_states=return_hidden_states or return_round_logits,
        )
        logits = self._readout(states, col_mask)
        round_logits = None
        if return_round_logits and trace is not None:
            round_logits = [self._readout(self.model.norm(s), col_mask) for s in trace]
        return {"logits": logits, "hidden_states": trace, "round_logits": round_logits, "loss": None}
