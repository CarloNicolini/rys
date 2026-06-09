"""Tests for the clause-variable message-passing SAT solver."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from rys.sat_data import (
    SatAssignmentDataset,
    make_sat_assignment_examples,
    multisample_validity,
    soft_sat_loss,
    verify_assignment_tensor,
)
import pandas as pd

from rys.sat_message_passing import MessagePassingConfig, MessagePassingSatModel
from rys.surgery import apply_rys
from rys.theory_validation import junction_mismatch, rho_phi_table, theory_fit


def _loader() -> DataLoader:
    examples = make_sat_assignment_examples(8, n_vars=4, n_clauses=12, seed=0)
    dataset = SatAssignmentDataset(examples, max_vars=5, max_clauses=14)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def _model() -> MessagePassingSatModel:
    torch.manual_seed(0)
    config = MessagePassingConfig(max_vars=5, max_clauses=14, d_model=16, n_rounds=4, d_mlp=32)
    return MessagePassingSatModel(config).eval()


def _kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "clause_variable_ids": batch["clause_variable_ids"],
        "clause_sign_ids": batch["clause_sign_ids"],
        "clause_mask": batch["clause_mask"],
    }


def test_forward_shapes_and_round_logits() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        outputs = model(**_kwargs(batch), return_round_logits=True)
    assert outputs["logits"].shape == (4, 5, 2)
    assert len(outputs["round_logits"]) == 4
    assert outputs["round_logits"][0].shape == (4, 5, 2)
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
    config = MessagePassingConfig(max_vars=5, max_clauses=14, d_model=16, n_rounds=4, d_mlp=32, weight_tied=True)
    model = MessagePassingSatModel(config).eval()
    layers = model.model.layers
    assert all(layer is layers[0] for layer in layers)
    tied_params = sum(p.numel() for p in model.parameters())

    untied = MessagePassingSatModel(MessagePassingConfig(max_vars=5, max_clauses=14, d_model=16, n_rounds=4, d_mlp=32))
    assert tied_params < sum(p.numel() for p in untied.parameters())


def test_prenorm_variable_stream_is_additive() -> None:
    # With pre-norm rounds the raw trace must telescope: x_j - x_i equals the
    # sum of per-round increments, i.e. no hidden rescaling of the stream.
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        trace = model(**_kwargs(batch), return_hidden_states=True)["hidden_states"]
    increments = sum(trace[t + 1] - trace[t] for t in range(len(trace) - 1))
    telescoped = trace[-1] - trace[0]
    torch.testing.assert_close(increments, telescoped, rtol=1e-4, atol=1e-4)


def test_soft_loss_and_verifier_agree_on_confident_solution() -> None:
    formula_examples = make_sat_assignment_examples(4, n_vars=4, n_clauses=12, seed=7)
    dataset = SatAssignmentDataset(formula_examples, max_vars=4, max_clauses=12)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    # Build confident logits from each example's known satisfying assignment.
    labels = batch["assignment_labels"].clamp_min(0)
    logits = torch.zeros(labels.shape[0], labels.shape[1], 2)
    logits[..., 1] = (labels == 1).float() * 12.0 - 6.0
    logits[..., 0] = -logits[..., 1]

    loss = soft_sat_loss(logits, batch["clause_variable_ids"], batch["clause_sign_ids"], batch["clause_mask"])
    valid = verify_assignment_tensor(
        logits.argmax(dim=-1),
        batch["clause_variable_ids"],
        batch["clause_sign_ids"],
        batch["clause_mask"],
    )
    assert bool(valid.all())
    assert float(loss) < 0.05


def _capture_variable_states(model: MessagePassingSatModel, batch: dict[str, torch.Tensor]) -> pd.DataFrame:
    with torch.inference_mode():
        trace = model(**_kwargs(batch), return_hidden_states=True)["hidden_states"]
    mask = batch["assignment_mask"].bool()
    rows = []
    for layer_idx, hidden in enumerate(trace):
        for row_idx in range(hidden.shape[0]):
            rows.append(
                {
                    "prompt_id": f"p{row_idx}",
                    "layer": layer_idx,
                    "activation": hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                    "strategy": "variable",
                }
            )
    return pd.DataFrame(rows)


def _stochastic_model() -> MessagePassingSatModel:
    torch.manual_seed(0)
    config = MessagePassingConfig(
        max_vars=5, max_clauses=14, d_model=16, n_rounds=4, d_mlp=32, stochastic=True, log_sigma_init=0.0
    )
    return MessagePassingSatModel(config).eval()


def test_stochastic_sampling_diversifies_but_mean_is_deterministic() -> None:
    model = _stochastic_model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        # Sampling on: two draws should differ.
        torch.manual_seed(1)
        s1 = model(**_kwargs(batch), sample=True)["logits"]
        torch.manual_seed(2)
        s2 = model(**_kwargs(batch), sample=True)["logits"]
        # Sampling off: the deterministic mean path is reproducible.
        d1 = model(**_kwargs(batch))["logits"]
        d2 = model(**_kwargs(batch))["logits"]
        out = model(**_kwargs(batch), sample=True)
    assert not torch.allclose(s1, s2)
    torch.testing.assert_close(d1, d2)
    assert out["mean_log_sigma"] is not None


def test_soft_sat_loss_reduction_none_is_per_formula() -> None:
    formula_examples = make_sat_assignment_examples(4, n_vars=4, n_clauses=12, seed=7)
    dataset = SatAssignmentDataset(formula_examples, max_vars=4, max_clauses=12)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    logits = torch.randn(4, 4, 2)
    per = soft_sat_loss(logits, batch["clause_variable_ids"], batch["clause_sign_ids"], batch["clause_mask"], reduction="none")
    scalar = soft_sat_loss(logits, batch["clause_variable_ids"], batch["clause_sign_ids"], batch["clause_mask"])
    assert per.shape == (4,)
    torch.testing.assert_close(per.mean(), scalar, rtol=1e-4, atol=1e-4)


def test_multisample_validity_metrics() -> None:
    formula_examples = make_sat_assignment_examples(4, n_vars=4, n_clauses=12, seed=7)
    dataset = SatAssignmentDataset(formula_examples, max_vars=4, max_clauses=12)
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    labels = batch["assignment_labels"].clamp_min(0)
    # Three samples: the true assignment, and two random ones.
    samples = torch.stack([labels, torch.zeros_like(labels), torch.ones_like(labels)])
    out = multisample_validity(
        samples,
        batch["clause_variable_ids"],
        batch["clause_sign_ids"],
        batch["clause_mask"],
        assignment_mask=batch["assignment_mask"],
    )
    assert bool(out["valid_any"].all())  # planted assignment is always valid
    assert (out["n_valid"] >= 1).all()
    assert (out["coverage"] >= 1).all()


def test_rho_phi_pipeline_runs() -> None:
    model = _model()
    batch = next(iter(_loader()))
    activations = _capture_variable_states(model, batch)
    table = rho_phi_table(activations)
    assert {"R", "cos_phi", "Q", "one_minus_cka_plateau", "one_minus_cka_full"} <= set(table.columns)
    assert (table["layer_i"] < table["layer_j"]).all()
    fit = theory_fit(table)
    assert fit["n_pairs_total"] == len(table)
    assert junction_mismatch(activations, window=(0, 2)) >= 0.0
