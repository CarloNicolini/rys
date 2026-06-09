"""Tiny residual Transformer for controlled synthetic SAT experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TinyTransformerConfig:
    """Configuration for :class:`TinySatTransformer`."""

    vocab_size: int
    max_seq_len: int
    d_model: int = 64
    n_layers: int = 6
    n_heads: int = 4
    d_mlp: int = 128
    dropout: float = 0.0
    n_classes: int = 2


@dataclass(frozen=True)
class FactorizedCNFTransformerConfig:
    """Configuration for :class:`FactorizedCNFSatTransformer`."""

    max_vars: int
    max_clauses: int
    d_model: int = 64
    n_layers: int = 6
    n_heads: int = 4
    d_mlp: int = 128
    dropout: float = 0.0
    n_classes: int = 2
    use_position_embedding: bool = True

    @property
    def max_seq_len(self) -> int:
        return 1 + 3 * self.max_clauses


@dataclass(frozen=True)
class FactorizedAssignmentTransformerConfig:
    """Configuration for :class:`FactorizedCNFAssignmentTransformer`."""

    max_vars: int
    max_clauses: int
    d_model: int = 64
    n_layers: int = 6
    n_heads: int = 4
    d_mlp: int = 128
    dropout: float = 0.0
    n_classes: int = 2
    use_position_embedding: bool = True

    @property
    def formula_seq_len(self) -> int:
        return 1 + 3 * self.max_clauses

    @property
    def max_seq_len(self) -> int:
        return self.formula_seq_len + self.max_vars


class TinyTransformerBlock(nn.Module):
    """Compatibility wrapper around PyTorch's pre-norm encoder layer."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_mlp,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor]:
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        hidden_states = self.layer(
            hidden_states,
            src_key_padding_mask=key_padding_mask,
        )
        return (hidden_states,)


class TinyTransformerBackbone(nn.Module):
    """Backbone exposing ``layers`` for compatibility with ``apply_rys``."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(TinyTransformerBlock(config) for _ in range(config.n_layers))
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {input_ids.shape[1]} exceeds max_seq_len={self.config.max_seq_len}."
            )

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_trace: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            if return_hidden_states:
                hidden_trace.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states, hidden_trace if return_hidden_states else None


class TinySatTransformer(nn.Module):
    """Classifier over encoded 3-SAT formulas.

    The nested ``self.model.layers`` layout mirrors Llama-style decoders so the
    existing :func:`rys.surgery.apply_rys` context manager can replay windows.
    """

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TinyTransformerBackbone(config)
        self.classifier = nn.Linear(config.d_model, config.n_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        hidden_states, hidden_trace = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
        )
        cls_state = hidden_states[:, 0, :]
        logits = self.classifier(cls_state)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss, "hidden_states": hidden_trace}


class FactorizedCNFBackbone(nn.Module):
    """Backbone over factorized literal coordinates.

    This is still a plain residual Transformer stack, but the input embedding
    exposes SAT structure directly: variable id, sign, clause id, literal slot,
    and token type are separate coordinates instead of being serialized as text.
    """

    def __init__(self, config: FactorizedCNFTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.variable_embedding = nn.Embedding(config.max_vars + 1, config.d_model, padding_idx=0)
        self.sign_embedding = nn.Embedding(3, config.d_model, padding_idx=0)
        self.clause_embedding = nn.Embedding(config.max_clauses + 1, config.d_model, padding_idx=0)
        self.slot_embedding = nn.Embedding(4, config.d_model, padding_idx=0)
        self.token_type_embedding = nn.Embedding(4, config.d_model, padding_idx=0)
        self.position_embedding = (
            nn.Embedding(config.max_seq_len, config.d_model)
            if config.use_position_embedding
            else None
        )
        self.embedding_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TinyTransformerBlock(
                TinyTransformerConfig(
                    vocab_size=1,
                    max_seq_len=config.max_seq_len,
                    d_model=config.d_model,
                    n_layers=config.n_layers,
                    n_heads=config.n_heads,
                    d_mlp=config.d_mlp,
                    dropout=config.dropout,
                    n_classes=config.n_classes,
                )
            )
            for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        *,
        variable_ids: torch.Tensor,
        sign_ids: torch.Tensor,
        clause_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if variable_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {variable_ids.shape[1]} exceeds max_seq_len={self.config.max_seq_len}."
            )

        hidden_states = (
            self.variable_embedding(variable_ids)
            + self.sign_embedding(sign_ids)
            + self.clause_embedding(clause_ids)
            + self.slot_embedding(slot_ids)
            + self.token_type_embedding(token_type_ids)
        )
        if self.position_embedding is not None:
            positions = torch.arange(variable_ids.shape[1], device=variable_ids.device).unsqueeze(0)
            hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.dropout(self.embedding_norm(hidden_states))
        hidden_trace: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            if return_hidden_states:
                hidden_trace.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states, hidden_trace if return_hidden_states else None


class FactorizedCNFSatTransformer(nn.Module):
    """RYS-compatible classifier over factorized CNF formulas."""

    def __init__(self, config: FactorizedCNFTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = FactorizedCNFBackbone(config)
        self.classifier = nn.Linear(config.d_model, config.n_classes)

    def forward(
        self,
        *,
        variable_ids: torch.Tensor,
        sign_ids: torch.Tensor,
        clause_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        hidden_states, hidden_trace = self.model(
            variable_ids=variable_ids,
            sign_ids=sign_ids,
            clause_ids=clause_ids,
            slot_ids=slot_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
        )
        cls_state = hidden_states[:, 0, :]
        logits = self.classifier(cls_state)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss, "hidden_states": hidden_trace}


class FactorizedCNFAssignmentTransformer(nn.Module):
    """RYS-compatible generator of canonical satisfying assignments.

    The input is a factorized CNF followed by one query token per variable.  The
    model emits a binary logit pair for each fixed query slot.
    """

    def __init__(self, config: FactorizedAssignmentTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = FactorizedCNFBackbone(config)  # type: ignore[arg-type]
        self.assignment_head = nn.Linear(config.d_model, 2)

    def forward(
        self,
        *,
        variable_ids: torch.Tensor,
        sign_ids: torch.Tensor,
        clause_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        assignment_labels: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        hidden_states, hidden_trace = self.model(
            variable_ids=variable_ids,
            sign_ids=sign_ids,
            clause_ids=clause_ids,
            slot_ids=slot_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
        )
        query_states = hidden_states[
            :,
            self.config.formula_seq_len : self.config.formula_seq_len + self.config.max_vars,
            :,
        ]
        logits = self.assignment_head(query_states)
        target = assignment_labels if assignment_labels is not None else labels
        loss = None
        if target is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 2),
                target.reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss, "hidden_states": hidden_trace}
