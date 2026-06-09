"""Tests for the N-Queens constraint-propagation solver."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rys.nqueens_data import (
    QueensDataset,
    make_queens_examples,
    soft_nqueens_loss,
    verify_boards_tensor,
)
from rys.nqueens_message_passing import QueensMessagePassingModel, QueensMPConfig
from rys.surgery import apply_rys
from rys.theory_validation import junction_mismatch, rho_phi_table, theory_fit


def _loader() -> DataLoader:
    examples = make_queens_examples(8, n=8, n_givens=2, seed=0)
    dataset = QueensDataset(examples, max_n=10)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def _model() -> QueensMessagePassingModel:
    torch.manual_seed(0)
    config = QueensMPConfig(max_n=10, d_model=16, n_rounds=4, n_heads=2, d_mlp=32)
    return QueensMessagePassingModel(config).eval()


def _kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "given_col": batch["given_col"],
        "is_given": batch["is_given"],
        "row_mask": batch["row_mask"],
        "col_mask": batch["col_mask"],
    }


def test_forward_shapes_and_round_logits() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        out = model(**_kwargs(batch), return_round_logits=True)
    assert out["logits"].shape == (4, 10, 10)
    assert len(out["round_logits"]) == 4
    assert len(out["hidden_states"]) == 4
    # Masked columns are suppressed.
    assert torch.isneginf(out["logits"][0, 0, 8]) or out["logits"][0, 0, 8] < -1e8


def test_apply_rys_reversible_and_active() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        base = model(**_kwargs(batch))["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=1):
            noop = model(**_kwargs(batch))["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=2):
            repeated = model(**_kwargs(batch))["logits"]
        after = model(**_kwargs(batch))["logits"]
    torch.testing.assert_close(base, noop)
    torch.testing.assert_close(base, after)
    assert not torch.allclose(base, repeated)


def test_prenorm_variable_stream_is_additive() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        trace = model(**_kwargs(batch), return_hidden_states=True)["hidden_states"]
    increments = sum(trace[t + 1] - trace[t] for t in range(len(trace) - 1))
    telescoped = trace[-1] - trace[0]
    torch.testing.assert_close(increments, telescoped, rtol=1e-4, atol=1e-4)


def test_soft_loss_and_verifier_agree_on_confident_board() -> None:
    examples = make_queens_examples(4, n=8, n_givens=2, seed=3)
    dataset = QueensDataset(examples, max_n=8)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    labels = batch["labels"].clamp_min(0)
    logits = torch.full((4, 8, 8), -6.0)
    for b in range(4):
        for r in range(8):
            logits[b, r, labels[b, r]] = 6.0
    loss = soft_nqueens_loss(logits, batch["row_mask"], batch["col_mask"], given_ce_weight=0.0)
    valid = verify_boards_tensor(logits.argmax(-1), batch["row_mask"], batch["given_col"], batch["is_given"])
    assert bool(valid.all())
    assert float(loss) < 0.05


def test_rho_phi_pipeline_runs() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        trace = model(**_kwargs(batch), return_hidden_states=True)["hidden_states"]
    mask = batch["row_mask"].bool()
    rows = []
    for layer_idx, hidden in enumerate(trace):
        for row_idx in range(hidden.shape[0]):
            rows.append({
                "prompt_id": f"p{row_idx}",
                "layer": layer_idx,
                "activation": hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                "strategy": "row",
            })
    activations = pd.DataFrame(rows)
    table = rho_phi_table(activations)
    assert {"R", "cos_phi", "one_minus_cka_full"} <= set(table.columns)
    fit = theory_fit(table)
    assert fit["n_pairs_total"] == len(table)
    assert junction_mismatch(activations, window=(0, 2)) >= 0.0
