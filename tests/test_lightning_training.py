"""Smoke tests for the shared Lightning training wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rys.coloring_data import ColoringDataset, make_coloring_examples
from rys.coloring_message_passing import ColoringMessagePassingModel, ColoringMPConfig
from rys.nqueens_data import QueensDataset, make_queens_examples
from rys.nqueens_message_passing import QueensMessagePassingModel, QueensMPConfig
from rys.sat_data import (
    SatAssignmentDataset,
    SatDataset,
    make_sat_assignment_examples,
    make_sat_examples,
    sequence_length,
    vocab_size,
)
from rys.sat_message_passing import MessagePassingConfig, MessagePassingSatModel
from rys.sorted_translation_data import SortedTranslationDataset, make_sorted_examples
from rys.sorted_translation_transformer import SortedTranslationConfig, SortedTranslationTransformer
from rys.surgery import apply_rys
from rys.tiny_transformer import (
    FactorizedCNFSatTransformer,
    FactorizedCNFTransformerConfig,
    TinySatTransformer,
    TinyTransformerConfig,
)
from rys.training.modules import (
    ClassifierLitModule,
    ColoringMPLitModule,
    NqueensMPLitModule,
    SatMPLitModule,
    SortedTranslationLitModule,
)
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model, resolve_training_checkpoint


def test_classifier_lightning_checkpoint_roundtrip(tmp_path: Path) -> None:
    examples = make_sat_examples(8, n_vars=4, n_clauses=8, seed=0, balanced=False)
    loader = DataLoader(SatDataset(examples, max_vars=4, max_clauses=8), batch_size=4)
    config = TinyTransformerConfig(
        vocab_size=vocab_size(4),
        max_seq_len=sequence_length(8),
        d_model=16,
        n_layers=2,
        n_heads=4,
        d_mlp=32,
    )
    lit = ClassifierLitModule(TinySatTransformer(config), lr=1e-3, weight_decay=0.0, model_config=config)
    ckpt = _fit_and_checkpoint(tmp_path, lit, loader, loader)
    model, _ = load_rys_model(ckpt, lambda _config: TinySatTransformer(config))
    batch = next(iter(loader))
    with torch.inference_mode(), apply_rys(model, window=(0, 1), n_repeats=1):
        assert model(batch["input_ids"], attention_mask=batch["attention_mask"])["logits"].shape == (4, 2)


def test_factorized_classifier_lightning_checkpoint_roundtrip(tmp_path: Path) -> None:
    # SatDataset carries both flat and factorized fields; the factorized model is
    # keyword-only, so the classifier module must route by model type, not batch.
    examples = make_sat_examples(8, n_vars=4, n_clauses=8, seed=0, balanced=False)
    loader = DataLoader(SatDataset(examples, max_vars=4, max_clauses=8), batch_size=4)
    config = FactorizedCNFTransformerConfig(max_vars=4, max_clauses=8, d_model=16, n_layers=2, n_heads=4, d_mlp=32)
    lit = ClassifierLitModule(FactorizedCNFSatTransformer(config), lr=1e-3, weight_decay=0.0, model_config=config)
    ckpt = _fit_and_checkpoint(tmp_path, lit, loader, loader)
    model, _ = load_rys_model(ckpt, lambda _config: FactorizedCNFSatTransformer(config))
    batch = next(iter(loader))
    with torch.inference_mode(), apply_rys(model, window=(0, 1), n_repeats=1):
        out = model(
            variable_ids=batch["variable_ids"],
            sign_ids=batch["sign_ids"],
            clause_ids=batch["clause_ids"],
            slot_ids=batch["slot_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["factor_attention_mask"],
        )
    assert out["logits"].shape == (4, 2)


def test_sat_mp_lightning_checkpoint_roundtrip(tmp_path: Path) -> None:
    loader = _sat_assignment_loader()
    config = MessagePassingConfig(max_vars=4, max_clauses=10, d_model=16, n_rounds=2, d_mlp=32)
    lit = SatMPLitModule(MessagePassingSatModel(config), lr=1e-3, weight_decay=0.0, model_config=config)
    ckpt = _fit_and_checkpoint(tmp_path, lit, loader, loader)
    model, _ = load_rys_model(ckpt, lambda _config: MessagePassingSatModel(config))
    batch = next(iter(loader))
    with torch.inference_mode(), apply_rys(model, window=(0, 1), n_repeats=1):
        out = model(
            clause_variable_ids=batch["clause_variable_ids"],
            clause_sign_ids=batch["clause_sign_ids"],
            clause_mask=batch["clause_mask"],
        )
    assert out["logits"].shape == (4, 4, 2)


def test_coloring_and_nqueens_lightning_checkpoint_roundtrip(tmp_path: Path) -> None:
    coloring_examples = make_coloring_examples(8, n=5, k=3, n_edges=6, n_givens=1, seed=0)
    coloring_loader = DataLoader(ColoringDataset(coloring_examples, max_v=5, n_colors=3), batch_size=4)
    coloring_config = ColoringMPConfig(max_v=5, n_colors=3, d_model=12, n_rounds=2, n_heads=3, d_mlp=24)
    coloring_lit = ColoringMPLitModule(
        ColoringMessagePassingModel(coloring_config),
        lr=1e-3,
        weight_decay=0.0,
        given_ce_weight=0.25,
        model_config=coloring_config,
    )
    coloring_ckpt = _fit_and_checkpoint(tmp_path / "coloring", coloring_lit, coloring_loader, coloring_loader)
    coloring_model, _ = load_rys_model(
        coloring_ckpt,
        lambda _config: ColoringMessagePassingModel(coloring_config),
    )
    assert _runs_coloring_with_rys(coloring_model, next(iter(coloring_loader)))

    queens_examples = make_queens_examples(8, n=5, n_givens=1, seed=0)
    queens_loader = DataLoader(QueensDataset(queens_examples, max_n=5), batch_size=4)
    queens_config = QueensMPConfig(max_n=5, d_model=12, n_rounds=2, n_heads=3, d_mlp=24)
    queens_lit = NqueensMPLitModule(
        QueensMessagePassingModel(queens_config),
        lr=1e-3,
        weight_decay=0.0,
        given_ce_weight=0.25,
        model_config=queens_config,
    )
    queens_ckpt = _fit_and_checkpoint(tmp_path / "queens", queens_lit, queens_loader, queens_loader)
    queens_model, _ = load_rys_model(queens_ckpt, lambda _config: QueensMessagePassingModel(queens_config))
    assert _runs_queens_with_rys(queens_model, next(iter(queens_loader)))


def test_sorted_translation_lightning_checkpoint_roundtrip(tmp_path: Path) -> None:
    examples = make_sorted_examples(8, 6, seed=0, vocab=16)
    loader = DataLoader(SortedTranslationDataset(examples), batch_size=4)
    config = SortedTranslationConfig(vocab_in=16, d_model=16, n_layers=2, n_heads=4, d_mlp=32)
    lit = SortedTranslationLitModule(
        SortedTranslationTransformer(config),
        lr=1e-3,
        weight_decay=0.0,
        model_config=config,
    )
    ckpt = _fit_and_checkpoint(tmp_path, lit, loader, loader)
    model, _ = load_rys_model(ckpt, lambda _config: SortedTranslationTransformer(config))
    batch = next(iter(loader))
    with torch.inference_mode(), apply_rys(model, window=(0, 1), n_repeats=1):
        assert model(batch["input_ids"])["logits"].shape == (4, 6, 16)


def test_sorted_translation_lightning_multi_loader_fit(tmp_path: Path) -> None:
    # The script feeds a *list* of per-length train loaders; Lightning then hands
    # ``training_step`` a list batch, which must not break logging/metrics.
    train_loaders = [
        DataLoader(SortedTranslationDataset(make_sorted_examples(8, length, seed=0, vocab=16)), batch_size=4)
        for length in (6, 8)
    ]
    val_loader = DataLoader(SortedTranslationDataset(make_sorted_examples(8, 6, seed=1, vocab=16)), batch_size=4)
    config = SortedTranslationConfig(vocab_in=16, d_model=16, n_layers=2, n_heads=4, d_mlp=32)
    lit = SortedTranslationLitModule(
        SortedTranslationTransformer(config),
        lr=1e-3,
        weight_decay=0.0,
        model_config=config,
    )
    trainer = build_trainer(
        tmp_path,
        max_epochs=1,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
        enable_progress_bar=False,
    )
    trainer.fit(lit, train_dataloaders=train_loaders, val_dataloaders=val_loader)
    assert best_checkpoint_path(trainer).exists()


def test_cache_batches_on_device_preserves_values() -> None:
    from rys.training.device_cache import cache_batches_on_device

    device = torch.device("cpu")
    loader = DataLoader(
        SatAssignmentDataset(
            make_sat_assignment_examples(8, n_vars=4, n_clauses=10, seed=0), max_vars=4, max_clauses=10
        ),
        batch_size=4,
    )
    cached = cache_batches_on_device(loader, device)
    originals = list(loader)
    assert len(cached) == len(originals)
    for cached_batch, original in zip(cached, originals):
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(cached_batch[key], value.to(device))
                # The cached tensor is already resident, so .to(device) is a no-op.
                assert cached_batch[key].to(device) is cached_batch[key]
            else:
                assert cached_batch[key] == value


def test_log_rys_progress_yields_all_windows(capsys) -> None:
    from rys.training.rys_logging import log_rys_progress

    windows = [(i, i + 1) for i in range(5)]
    seen = list(log_rys_progress(windows, device="cpu", depth=5))
    assert seen == windows
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    events = [line["event"] for line in lines]
    assert events[0] == "start"
    assert events[-1] == "done"
    assert lines[0]["device"] == "cpu"
    assert lines[0]["n_windows"] == 5
    assert all(line["depth"] == 5 for line in lines)


def test_resolve_training_checkpoint_falls_back_to_disk(tmp_path: Path) -> None:
    ckpt = tmp_path / "best-03.ckpt"
    ckpt.write_bytes(b"stub")
    trainer = build_trainer(
        tmp_path,
        max_epochs=1,
        monitor="valid_assignment_rate",
        enable_progress_bar=False,
    )
    assert resolve_training_checkpoint(trainer, tmp_path) == ckpt


def _sat_assignment_loader() -> DataLoader:
    examples = make_sat_assignment_examples(8, n_vars=4, n_clauses=10, seed=0)
    return DataLoader(SatAssignmentDataset(examples, max_vars=4, max_clauses=10), batch_size=4)


def _fit_and_checkpoint(tmp_path: Path, lit, train_loader: DataLoader, val_loader: DataLoader) -> Path:
    trainer = build_trainer(
        tmp_path,
        max_epochs=1,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
        enable_progress_bar=False,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    ckpt = best_checkpoint_path(trainer)
    assert ckpt.exists()
    return ckpt


def _runs_coloring_with_rys(model: ColoringMessagePassingModel, batch: dict[str, torch.Tensor]) -> bool:
    with torch.inference_mode(), apply_rys(model, window=(0, 1), n_repeats=1):
        out = model(
            given_color=batch["given_color"],
            is_given=batch["is_given"],
            degree=batch["degree"],
            adjacency=batch["adjacency"],
            vertex_mask=batch["vertex_mask"],
        )
    return out["logits"].shape == (4, 5, 3)


def _runs_queens_with_rys(model: QueensMessagePassingModel, batch: dict[str, torch.Tensor]) -> bool:
    with torch.inference_mode(), apply_rys(model, window=(0, 1), n_repeats=1):
        out = model(
            given_col=batch["given_col"],
            is_given=batch["is_given"],
            row_mask=batch["row_mask"],
            col_mask=batch["col_mask"],
        )
    return out["logits"].shape == (4, 5, 5)
