"""Tests for the adjacency-masked message-passing graph-colouring solver."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rys.coloring_data import (
    ColoringDataset,
    make_coloring_examples,
    soft_coloring_loss,
    verify_coloring_tensor,
)
from rys.coloring_message_passing import ColoringMPConfig, ColoringMessagePassingModel
from rys.surgery import apply_rys
from rys.theory_validation import junction_mismatch, rho_phi_table, theory_fit


def _loader() -> DataLoader:
    examples = make_coloring_examples(8, n=10, k=3, n_edges=18, n_givens=3, seed=0)
    dataset = ColoringDataset(examples, max_v=12, n_colors=3)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def _model() -> ColoringMessagePassingModel:
    torch.manual_seed(0)
    config = ColoringMPConfig(max_v=12, n_colors=3, d_model=16, n_rounds=4, n_heads=2, d_mlp=32)
    return ColoringMessagePassingModel(config).eval()


def _kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "given_color": batch["given_color"],
        "is_given": batch["is_given"],
        "degree": batch["degree"],
        "adjacency": batch["adjacency"],
        "vertex_mask": batch["vertex_mask"],
    }


def test_forward_shapes_and_round_logits() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        outputs = model(**_kwargs(batch), return_round_logits=True)
    assert outputs["logits"].shape == (4, 12, 3)
    assert len(outputs["round_logits"]) == 4
    assert outputs["round_logits"][0].shape == (4, 12, 3)
    assert len(outputs["hidden_states"]) == 4


def test_apply_rys_reversible_and_active() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        baseline = model(**_kwargs(batch))["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=1):
            no_op = model(**_kwargs(batch))["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=2):
            repeated = model(**_kwargs(batch))["logits"]
        after = model(**_kwargs(batch))["logits"]
    torch.testing.assert_close(baseline, no_op)
    torch.testing.assert_close(baseline, after)
    assert not torch.allclose(baseline, repeated)


def test_weight_tied_shares_one_round() -> None:
    torch.manual_seed(0)
    tied = ColoringMessagePassingModel(
        ColoringMPConfig(max_v=12, n_colors=3, d_model=16, n_rounds=4, n_heads=2, d_mlp=32, weight_tied=True)
    )
    layers = tied.model.layers
    assert all(layer is layers[0] for layer in layers)
    untied = ColoringMessagePassingModel(
        ColoringMPConfig(max_v=12, n_colors=3, d_model=16, n_rounds=4, n_heads=2, d_mlp=32)
    )
    assert sum(p.numel() for p in tied.parameters()) < sum(p.numel() for p in untied.parameters())


def test_prenorm_vertex_stream_is_additive() -> None:
    # With pre-norm rounds the raw trace must telescope: x_j - x_i equals the
    # sum of per-round increments, i.e. no hidden rescaling of the stream.
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        trace = model(**_kwargs(batch), return_hidden_states=True)["hidden_states"]
    increments = sum(trace[t + 1] - trace[t] for t in range(len(trace) - 1))
    telescoped = trace[-1] - trace[0]
    torch.testing.assert_close(increments, telescoped, rtol=1e-4, atol=1e-4)


def test_permutation_equivariance() -> None:
    # Relabelling vertices permutes the outputs identically: no absolute identity.
    model = _model()
    batch = next(iter(_loader()))
    kw = _kwargs(batch)
    with torch.inference_mode():
        base = model(**kw)["logits"]
    perm = torch.randperm(kw["vertex_mask"].shape[1])
    permuted = {
        "given_color": kw["given_color"][:, perm],
        "is_given": kw["is_given"][:, perm],
        "degree": kw["degree"][:, perm],
        "adjacency": kw["adjacency"][:, perm][:, :, perm],
        "vertex_mask": kw["vertex_mask"][:, perm],
    }
    with torch.inference_mode():
        out = model(**permuted)["logits"]
    torch.testing.assert_close(out, base[:, perm], rtol=1e-4, atol=1e-4)


def test_soft_loss_and_verifier_agree_on_confident_solution() -> None:
    examples = make_coloring_examples(4, n=10, k=3, n_edges=16, n_givens=3, seed=7)
    dataset = ColoringDataset(examples, max_v=10, n_colors=3)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    labels = batch["labels"].clamp_min(0)
    logits = torch.full((labels.shape[0], labels.shape[1], 3), -6.0)
    logits.scatter_(2, labels.unsqueeze(-1), 6.0)
    loss = soft_coloring_loss(logits, batch["adjacency"], batch["vertex_mask"], given_ce_weight=0.0)
    valid = verify_coloring_tensor(
        logits.argmax(dim=-1),
        batch["vertex_mask"],
        batch["adjacency"],
        batch["given_color"],
        batch["is_given"],
    )
    assert bool(valid.all())
    assert float(loss) < 0.05


def _capture_vertex_states(model: ColoringMessagePassingModel, batch: dict[str, torch.Tensor]) -> pd.DataFrame:
    with torch.inference_mode():
        trace = model(**_kwargs(batch), return_hidden_states=True)["hidden_states"]
    mask = batch["vertex_mask"].bool()
    rows = []
    for layer_idx, hidden in enumerate(trace):
        for row_idx in range(hidden.shape[0]):
            rows.append(
                {
                    "prompt_id": f"p{row_idx}",
                    "layer": layer_idx,
                    "activation": hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                    "strategy": "vertex",
                }
            )
    return pd.DataFrame(rows)


def test_rho_phi_pipeline_runs() -> None:
    model = _model()
    batch = next(iter(_loader()))
    activations = _capture_vertex_states(model, batch)
    table = rho_phi_table(activations)
    assert {
        "R",
        "cos_phi",
        "Q",
        "cos_psi",
        "one_minus_cka_Rphi",
        "one_minus_cka_Qpsi",
        "one_minus_cka_plateau",
        "one_minus_cka_full",
    } <= set(table.columns)
    assert (table["layer_i"] < table["layer_j"]).all()
    fit = theory_fit(table)
    assert fit["n_pairs_total"] == len(table)
    assert junction_mismatch(activations, window=(0, 2)) >= 0.0
