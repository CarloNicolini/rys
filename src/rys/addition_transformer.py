"""RoPE Transformer for the N-ary addition task, RYS-compatible.

The model is an encoder-only stack that reads a serialised addition problem
(``315 + 120 + ... =`` followed by ``answer_width`` answer slots) and predicts
the sum's digit classes at the trailing answer positions in parallel (one
10-way classification per slot). It is deliberately built so that:

- the input embedding ``W_E`` consumes a one-hot token vector (digits plus the
  structural ``+``/``=``/``ANS`` tokens) and the output head ``W_U`` is a
  separate ``d_model -> 10`` digit-class matrix (no weight tying), isolating a
  distinct decoding head;
- position is encoded only with Rotary Positional Embeddings (RoPE), so the
  model extrapolates to more operands (longer inputs) out of distribution with
  no learned positional table;
- every layer is one entry of ``model.model.layers`` and each round runs
  ``forward(hidden, *, attention_mask=None) -> (hidden,)``, so the existing
  :func:`rys.surgery.apply_rys` hook replays windows unchanged.

The attention/MLP round is reused verbatim from the sorted-translation probe
(:class:`rys.sorted_translation_transformer.RoPESelfAttentionRound`), so the
``rho/phi`` CKA decomposition applies exactly. ``weight_tied`` shares one round
across depth, turning the stack into an iterated map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rys.addition_data import N_DIGIT_CLASSES, VOCAB_IN
from rys.sorted_translation_transformer import RoPESelfAttentionRound


@dataclass(frozen=True)
class AdditionConfig:
    """Configuration for :class:`AdditionTransformer`.

    Field names match the attributes :class:`RoPESelfAttentionRound` reads, so
    the round is duck-compatible with this config.
    """

    vocab_in: int = VOCAB_IN
    n_digit_classes: int = N_DIGIT_CLASSES
    d_model: int = 128
    n_layers: int = 8
    n_heads: int = 4
    d_mlp: int = 512
    dropout: float = 0.0
    pre_norm: bool = True
    weight_tied: bool = False
    rope_base: float = 10000.0
    use_rope: bool = True
    causal: bool = False


class AdditionBackbone(nn.Module):
    """Backbone exposing ``layers`` for RYS replay over addition rounds."""

    def __init__(self, config: AdditionConfig) -> None:
        super().__init__()
        self.config = config
        # W_E consumes a one-hot input token; no bias and no positional table.
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


class AdditionTransformer(nn.Module):
    """RYS-compatible encoder-only n-ary adder.

    ``self.model.layers`` mirrors the Llama-style layout so
    :func:`rys.surgery.apply_rys` can replay windows. The classification head
    ``unembed`` (``W_U``) is a distinct ``d_model -> 10`` matrix and is never
    tied to the input embedding ``W_E``. Logits are emitted at every position;
    the loss and metrics read only the trailing ``K = target_ids.shape[1]``
    answer slots.
    """

    def __init__(self, config: AdditionConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AdditionBackbone(config)
        self.unembed = nn.Linear(config.d_model, config.n_digit_classes)

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
            answer_width = target_ids.shape[1]
            answer_logits = logits[:, -answer_width:, :]
            loss = F.cross_entropy(
                answer_logits.reshape(-1, self.config.n_digit_classes),
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
