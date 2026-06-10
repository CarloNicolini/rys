"""Clause-variable message-passing SAT solver, RYS-compatible.

Each message-passing round is one entry of ``model.model.layers`` so the
existing :func:`rys.surgery.apply_rys` hook and the CKA capture utilities work
unchanged.  A round implements one NeuroSAT-style update::

    variables -> clauses : each clause aggregates its three signed literals
    clauses -> variables : each variable aggregates the clauses it appears in

The state is a single packed tensor ``(batch, max_vars + max_clauses, d)`` so it
matches the ``layer(hidden, attention_mask=...) -> (hidden,)`` calling
convention.  The bipartite adjacency is supplied per batch through a lightweight
context object set on every round before the forward loop.

The model is deliberately permutation-equivariant: variables and clauses start
from a single shared learned vector and are distinguished only by the formula
structure.  This is the property that lets the solver generalise to larger
formulas (more variables and clauses) out of distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MessagePassingConfig:
    """Configuration for :class:`MessagePassingSatModel`.

    ``pre_norm`` keeps the variable residual stream purely additive
    (``h <- h + F(LN(h))``) so the telescoping identity ``x_j = x_i + S_ij``
    holds and the rho/phi CKA decomposition applies exactly.      ``weight_tied``
    shares one round across depth, turning the stack into an iterated map
    ``h <- h + F(h)`` whose repeated application is the deep-equilibrium
    reading of RYS.  ``var_id_embeddings`` gives each variable a distinct
    learned initial state, deliberately breaking the permutation-equivariance
    that otherwise caps expressivity at the Weisfeiler-Leman limit.
    """

    max_vars: int
    max_clauses: int
    d_model: int = 64
    n_rounds: int = 8
    d_mlp: int = 128
    dropout: float = 0.0
    clause_arity: int = 3
    pre_norm: bool = True
    weight_tied: bool = False
    var_id_embeddings: bool = False
    stochastic: bool = False
    log_sigma_init: float = -2.0
    log_sigma_min: float = -6.0
    log_sigma_max: float = 2.0


@dataclass
class _BipartiteContext:
    """Per-batch adjacency shared by every message-passing round."""

    var_idx: torch.Tensor  # (B, C, arity) zero-based variable indices
    sign_ids: torch.Tensor  # (B, C, arity) 1=positive, 2=negative, 0=pad
    slot_valid: torch.Tensor  # (B, C, arity) float mask of real literals
    clause_mask: torch.Tensor  # (B, C) float mask of real clauses
    max_vars: int


class MessagePassingRound(nn.Module):
    """One variable<->clause message-passing round with residual updates."""

    def __init__(self, config: MessagePassingConfig) -> None:
        super().__init__()
        d = config.d_model
        self.max_vars = config.max_vars
        self.pre_norm = config.pre_norm
        self.literal_in = nn.Linear(d, d)
        self.literal_sign = nn.Embedding(3, d, padding_idx=0)
        self.clause_update = nn.Sequential(
            nn.Linear(2 * d, config.d_mlp),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_mlp, d),
        )
        self.clause_norm = nn.LayerNorm(d)
        self.clause_out = nn.Linear(d, d)
        self.message_sign = nn.Embedding(3, d, padding_idx=0)
        self.var_update = nn.Sequential(
            nn.Linear(2 * d, config.d_mlp),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_mlp, d),
        )
        self.var_norm = nn.LayerNorm(d)
        self.stochastic = config.stochastic
        if config.stochastic:
            # Learned Gaussian guidance on the deterministic variable update.
            self.mu_head = nn.Linear(d, d)
            self.log_sigma_head = nn.Linear(d, d)
            nn.init.zeros_(self.mu_head.weight)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.zeros_(self.log_sigma_head.weight)
            nn.init.constant_(self.log_sigma_head.bias, config.log_sigma_init)
            self._log_sigma_min = config.log_sigma_min
            self._log_sigma_max = config.log_sigma_max
        self.ctx: _BipartiteContext | None = None
        # Set per forward by the backbone: whether to draw noise, and an
        # optional list collecting mean log-sigma for the variance regulariser.
        self.sample: bool = False
        self.noise_sink: list[torch.Tensor] | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor]:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("Message-passing context was not set on the round.")
        n_vars = self.max_vars
        var_states = hidden[:, :n_vars]
        clause_states = hidden[:, n_vars:]
        batch, n_clauses, arity = ctx.var_idx.shape
        d = var_states.shape[-1]

        # Variables -> clauses: gather the literals of each clause and pool them.
        # In pre-norm mode the message reads a normalised copy of the variable
        # stream, leaving the stream itself untouched until the residual add.
        var_for_msg = self.var_norm(var_states) if self.pre_norm else var_states
        flat_idx = ctx.var_idx.reshape(batch, n_clauses * arity, 1).expand(batch, n_clauses * arity, d)
        gathered = torch.gather(var_for_msg, 1, flat_idx).reshape(batch, n_clauses, arity, d)
        literals = self.literal_in(gathered) + self.literal_sign(ctx.sign_ids)
        literals = literals * ctx.slot_valid.unsqueeze(-1)
        clause_message = literals.sum(dim=2)

        # Clause stream stays bounded (post-norm); it is an auxiliary buffer and
        # the readout never reads it directly.
        clause_in = self.clause_norm(clause_states) if self.pre_norm else clause_states
        clause_states = clause_states + self.clause_update(torch.cat([clause_in, clause_message], dim=-1))
        if not self.pre_norm:
            clause_states = self.clause_norm(clause_states)
        clause_states = clause_states * ctx.clause_mask.unsqueeze(-1)

        # Clauses -> variables: scatter each clause back to its literals.
        clause_signal = self.clause_out(self.clause_norm(clause_states) if self.pre_norm else clause_states)
        per_slot = clause_signal.unsqueeze(2) + self.message_sign(ctx.sign_ids)
        per_slot = per_slot * ctx.slot_valid.unsqueeze(-1)
        var_message = torch.zeros(batch, n_vars, d, device=hidden.device, dtype=hidden.dtype)
        scatter_idx = ctx.var_idx.reshape(batch, n_clauses * arity, 1).expand(batch, n_clauses * arity, d)
        var_message = var_message.scatter_add(1, scatter_idx, per_slot.reshape(batch, n_clauses * arity, d))

        delta = self.var_update(torch.cat([var_for_msg, var_message], dim=-1))
        if self.stochastic:
            # GRAM-style learned guidance: deterministic drift mu plus optional
            # reparameterised exploration.  The deterministic operator used by
            # the rho/phi analysis is F + mu (noise off); sampling adds width.
            mu = self.mu_head(delta)
            log_sigma = self.log_sigma_head(delta).clamp(self._log_sigma_min, self._log_sigma_max)
            if self.noise_sink is not None:
                self.noise_sink.append(log_sigma.mean())
            delta = delta + mu
            if self.sample:
                delta = delta + torch.exp(log_sigma) * torch.randn_like(delta)
        if self.pre_norm:
            # Purely additive residual stream: h <- h + F(LN(h)).  Telescoping
            # x_j = x_i + S_ij holds, so the rho/phi decomposition is exact.
            var_states = var_states + delta
        else:
            var_states = self.var_norm(var_states + delta)

        return (torch.cat([var_states, clause_states], dim=1),)


class MessagePassingBackbone(nn.Module):
    """Backbone exposing ``layers`` for RYS replay over message-passing rounds."""

    def __init__(self, config: MessagePassingConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.var_init = nn.Parameter(torch.randn(d) * 0.02)
        self.clause_init = nn.Parameter(torch.randn(d) * 0.02)
        if config.var_id_embeddings:
            self.var_id_embed = nn.Embedding(config.max_vars, d)
            nn.init.normal_(self.var_id_embed.weight, std=0.02)
        if config.weight_tied:
            shared = MessagePassingRound(config)
            self.layers = nn.ModuleList(shared for _ in range(config.n_rounds))
        else:
            self.layers = nn.ModuleList(MessagePassingRound(config) for _ in range(config.n_rounds))
        self.norm = nn.LayerNorm(d)

    def build_context(
        self,
        clause_variable_ids: torch.Tensor,
        clause_sign_ids: torch.Tensor,
        clause_mask: torch.Tensor,
    ) -> _BipartiteContext:
        var_idx = (clause_variable_ids - 1).clamp_min(0)
        slot_valid = (clause_variable_ids > 0).to(clause_variable_ids.dtype)
        slot_valid = slot_valid * clause_mask.unsqueeze(-1)
        return _BipartiteContext(
            var_idx=var_idx,
            sign_ids=clause_sign_ids * (clause_variable_ids > 0),
            slot_valid=slot_valid.float(),
            clause_mask=clause_mask.float(),
            max_vars=self.config.max_vars,
        )

    def forward(
        self,
        *,
        clause_variable_ids: torch.Tensor,
        clause_sign_ids: torch.Tensor,
        clause_mask: torch.Tensor,
        return_hidden_states: bool = False,
        sample: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
        batch = clause_variable_ids.shape[0]
        d = self.config.d_model
        if self.config.var_id_embeddings:
            ids = torch.arange(self.config.max_vars, device=clause_variable_ids.device)
            var_states = self.var_id_embed(ids).unsqueeze(0).expand(batch, self.config.max_vars, d)
        else:
            var_states = self.var_init.view(1, 1, d).expand(batch, self.config.max_vars, d)
        # The clause stream is sized from the input, not from ``config.max_clauses``.
        # Every clause-side weight (literal/clause/message modules) is shared across
        # clauses, so the only thing that ever needed the configured maximum was this
        # initial buffer.  Reading the count from the batch lets a loader pad clauses
        # to the per-batch maximum instead of the global one with identical math
        # (padded clauses are fully masked), which removes the dominant wasted compute.
        n_clauses = clause_variable_ids.shape[1]
        clause_states = self.clause_init.view(1, 1, d).expand(batch, n_clauses, d)
        hidden = torch.cat([var_states, clause_states], dim=1).contiguous()

        context = self.build_context(clause_variable_ids, clause_sign_ids, clause_mask)
        sink: list[torch.Tensor] | None = [] if self.config.stochastic else None
        trace: list[torch.Tensor] = []
        for layer in self.layers:
            layer.ctx = context
            layer.sample = sample
            layer.noise_sink = sink
            hidden = layer(hidden, attention_mask=None)[0]
            if return_hidden_states:
                # Raw (pre-final-norm) variable residual stream, so the
                # telescoping identity needed by the rho/phi test is preserved.
                trace.append(hidden[:, : self.config.max_vars].clone())

        mean_log_sigma = torch.stack(sink).mean() if sink else None
        var_states = self.norm(hidden[:, : self.config.max_vars])
        return var_states, (trace if return_hidden_states else None), mean_log_sigma


class MessagePassingSatModel(nn.Module):
    """RYS-compatible clause-variable SAT assignment solver."""

    def __init__(self, config: MessagePassingConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MessagePassingBackbone(config)
        self.assignment_head = nn.Linear(config.d_model, 2)

    def forward(
        self,
        *,
        clause_variable_ids: torch.Tensor,
        clause_sign_ids: torch.Tensor,
        clause_mask: torch.Tensor,
        return_hidden_states: bool = False,
        return_round_logits: bool = False,
        sample: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        var_states, trace, mean_log_sigma = self.model(
            clause_variable_ids=clause_variable_ids,
            clause_sign_ids=clause_sign_ids,
            clause_mask=clause_mask,
            return_hidden_states=return_hidden_states or return_round_logits,
            sample=sample,
        )
        logits = self.assignment_head(var_states)
        round_logits = None
        if return_round_logits and trace is not None:
            # ``trace`` holds raw pre-norm states; apply the final norm before
            # the head so each round read-out matches the terminal convention.
            round_logits = [self.assignment_head(self.model.norm(states)) for states in trace]
        return {
            "logits": logits,
            "hidden_states": trace,
            "round_logits": round_logits,
            "mean_log_sigma": mean_log_sigma,
            "loss": None,
        }
