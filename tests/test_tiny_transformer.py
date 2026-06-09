"""Tests for the tiny SAT Transformer and tensor activation capture."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from rys.cka import cka_matrix
from rys.sat_data import (
    SatAssignmentDataset,
    SatDataset,
    factorized_assignment_sequence_length,
    factorized_sequence_length,
    make_sat_assignment_examples,
    make_sat_examples,
    sequence_length,
    vocab_size,
)
from rys.surgery import apply_rys
from rys.tensor_activations import capture_tensor_residual_stream
from rys.tiny_transformer import (
    FactorizedAssignmentTransformerConfig,
    FactorizedCNFAssignmentTransformer,
    FactorizedCNFSatTransformer,
    FactorizedCNFTransformerConfig,
    TinySatTransformer,
    TinyTransformerConfig,
)


def _loader() -> DataLoader:
    examples = make_sat_examples(
        8,
        n_vars=4,
        n_clauses=8,
        seed=0,
        balanced=False,
    )
    dataset = SatDataset(examples, max_vars=4, max_clauses=8)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def _model() -> TinySatTransformer:
    torch.manual_seed(0)
    config = TinyTransformerConfig(
        vocab_size=vocab_size(4),
        max_seq_len=sequence_length(8),
        d_model=16,
        n_layers=3,
        n_heads=4,
        d_mlp=32,
    )
    return TinySatTransformer(config).eval()


def _factorized_model() -> FactorizedCNFSatTransformer:
    torch.manual_seed(0)
    config = FactorizedCNFTransformerConfig(
        max_vars=4,
        max_clauses=8,
        d_model=16,
        n_layers=3,
        n_heads=4,
        d_mlp=32,
    )
    return FactorizedCNFSatTransformer(config).eval()


def _assignment_loader() -> DataLoader:
    examples = make_sat_assignment_examples(
        8,
        n_vars=4,
        n_clauses=12,
        seed=0,
    )
    dataset = SatAssignmentDataset(examples, max_vars=5, max_clauses=12)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def _assignment_model() -> FactorizedCNFAssignmentTransformer:
    torch.manual_seed(0)
    config = FactorizedAssignmentTransformerConfig(
        max_vars=5,
        max_clauses=12,
        d_model=16,
        n_layers=3,
        n_heads=4,
        d_mlp=32,
    )
    return FactorizedCNFAssignmentTransformer(config).eval()


def test_tiny_sat_transformer_forward_shape() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            return_hidden_states=True,
        )
    assert outputs["logits"].shape == (4, 2)
    assert outputs["loss"] is not None
    assert len(outputs["hidden_states"]) == 3
    assert outputs["hidden_states"][0].shape == (4, sequence_length(8), 16)


def test_tiny_sat_transformer_apply_rys_reversible() -> None:
    model = _model()
    batch = next(iter(_loader()))
    with torch.inference_mode():
        baseline = model(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=1):
            no_op = model(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=2):
            modified = model(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
        after = model(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]

    torch.testing.assert_close(baseline, no_op)
    torch.testing.assert_close(baseline, after)
    assert not torch.allclose(baseline, modified)


def test_tensor_activation_capture_feeds_cka() -> None:
    model = _model()
    activations = capture_tensor_residual_stream(
        model,
        _loader(),
        token_strategy="cls",
        max_batches=2,
    )
    assert set(activations.columns) == {"prompt_id", "layer", "activation", "strategy"}
    assert sorted(activations["layer"].unique().tolist()) == [0, 1, 2]

    M = cka_matrix(activations, unbiased=False)
    assert M.shape == (3, 3)
    np.testing.assert_allclose(M.to_numpy(), M.to_numpy().T, atol=1e-5)
    assert np.isfinite(M.to_numpy()).all()


def test_factorized_cnf_transformer_forward_and_rys() -> None:
    model = _factorized_model()
    batch = next(iter(_loader()))
    kwargs = {
        "variable_ids": batch["variable_ids"],
        "sign_ids": batch["sign_ids"],
        "clause_ids": batch["clause_ids"],
        "slot_ids": batch["slot_ids"],
        "token_type_ids": batch["token_type_ids"],
        "attention_mask": batch["factor_attention_mask"],
    }
    with torch.inference_mode():
        outputs = model(**kwargs, labels=batch["labels"], return_hidden_states=True)
        baseline = outputs["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=2):
            modified = model(**kwargs)["logits"]
        after = model(**kwargs)["logits"]

    assert baseline.shape == (4, 2)
    assert len(outputs["hidden_states"]) == 3
    assert outputs["hidden_states"][0].shape == (4, factorized_sequence_length(8), 16)
    torch.testing.assert_close(baseline, after)
    assert not torch.allclose(baseline, modified)


def test_factorized_activation_capture_feeds_cka() -> None:
    model = _factorized_model()
    activations = capture_tensor_residual_stream(
        model,
        _loader(),
        architecture="factorized",
        token_strategy="cls",
        max_batches=2,
    )
    assert sorted(activations["layer"].unique().tolist()) == [0, 1, 2]
    M = cka_matrix(activations, unbiased=False)
    assert M.shape == (3, 3)
    assert np.isfinite(M.to_numpy()).all()


def test_factorized_assignment_transformer_forward_and_rys() -> None:
    model = _assignment_model()
    batch = next(iter(_assignment_loader()))
    kwargs = {
        "variable_ids": batch["variable_ids"],
        "sign_ids": batch["sign_ids"],
        "clause_ids": batch["clause_ids"],
        "slot_ids": batch["slot_ids"],
        "token_type_ids": batch["token_type_ids"],
        "attention_mask": batch["factor_attention_mask"],
    }
    with torch.inference_mode():
        outputs = model(**kwargs, assignment_labels=batch["assignment_labels"], return_hidden_states=True)
        baseline = outputs["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=2):
            modified = model(**kwargs)["logits"]
        after = model(**kwargs)["logits"]

    assert baseline.shape == (4, 5, 2)
    assert outputs["loss"] is not None
    assert len(outputs["hidden_states"]) == 3
    assert outputs["hidden_states"][0].shape == (4, factorized_assignment_sequence_length(5, 12), 16)
    torch.testing.assert_close(baseline, after)
    assert not torch.allclose(baseline, modified)
