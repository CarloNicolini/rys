"""LightningDataModule wrappers around existing RYS datasets."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
from torch.utils.data import DataLoader, Dataset

from rys.coloring_data import ColoringDataset, ColoringExample, make_coloring_splits
from rys.nqueens_data import QueensDataset, QueensExample, make_queens_splits
from rys.sat_data import (
    SatAssignmentDataset,
    SatAssignmentExample,
    SatDataset,
    make_sat_assignment_splits,
    make_sat_splits,
)
from rys.sorted_translation_data import (
    SortedExample,
    SortedTranslationDataset,
    make_sorted_splits,
)


class SplitDataModule(L.LightningDataModule):
    """A minimal DataModule for already-constructed split datasets."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        *,
        batch_size: int,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.datasets = dict(datasets)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def predict_dataloader(self) -> list[DataLoader]:
        return [
            self._loader(name, shuffle=False)
            for name in self.datasets
            if name not in {"train", "val"}
        ]

    def dataloader(self, split: str) -> DataLoader:
        return self._loader(split, shuffle=False)

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )


class SatClassificationDataModule(SplitDataModule):
    def __init__(
        self,
        *,
        n_train: int,
        n_val: int,
        n_test: int,
        n_vars: int,
        n_clauses: int,
        seed: int,
        batch_size: int,
        max_vars: int | None = None,
        max_clauses: int | None = None,
    ) -> None:
        splits = make_sat_splits(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed,
        )
        max_vars = max_vars or n_vars
        max_clauses = max_clauses or n_clauses
        datasets = {
            name: SatDataset(examples, max_vars=max_vars, max_clauses=max_clauses)
            for name, examples in splits.items()
        }
        super().__init__(datasets, batch_size=batch_size)


class SatAssignmentDataModule(SplitDataModule):
    def __init__(
        self,
        *,
        n_train: int,
        n_val: int,
        n_test: int,
        n_vars: int,
        n_clauses: int,
        seed: int,
        batch_size: int,
        max_vars: int | None = None,
        max_clauses: int | None = None,
        extra_splits: Mapping[str, list[SatAssignmentExample]] | None = None,
    ) -> None:
        splits = make_sat_assignment_splits(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed,
        )
        if extra_splits:
            splits.update(extra_splits)
        max_vars = max_vars or max(example.n_vars for examples in splits.values() for example in examples)
        max_clauses = max_clauses or max(example.n_clauses for examples in splits.values() for example in examples)
        datasets = {
            name: SatAssignmentDataset(examples, max_vars=max_vars, max_clauses=max_clauses)
            for name, examples in splits.items()
        }
        super().__init__(datasets, batch_size=batch_size)


class ColoringDataModule(SplitDataModule):
    def __init__(
        self,
        *,
        n_train: int,
        n_val: int,
        n_test: int,
        n: int,
        k: int,
        n_edges: int,
        n_givens: int,
        seed: int,
        batch_size: int,
        max_v: int | None = None,
        extra_splits: Mapping[str, list[ColoringExample]] | None = None,
    ) -> None:
        splits = make_coloring_splits(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n=n,
            k=k,
            n_edges=n_edges,
            n_givens=n_givens,
            seed=seed,
        )
        if extra_splits:
            splits.update(extra_splits)
        max_v = max_v or max(example.n for examples in splits.values() for example in examples)
        datasets = {
            name: ColoringDataset(examples, max_v=max_v, n_colors=k)
            for name, examples in splits.items()
        }
        super().__init__(datasets, batch_size=batch_size)


class QueensDataModule(SplitDataModule):
    def __init__(
        self,
        *,
        n_train: int,
        n_val: int,
        n_test: int,
        n: int,
        n_givens: int,
        seed: int,
        batch_size: int,
        max_n: int | None = None,
        extra_splits: Mapping[str, list[QueensExample]] | None = None,
    ) -> None:
        splits = make_queens_splits(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            n=n,
            n_givens=n_givens,
            seed=seed,
        )
        if extra_splits:
            splits.update(extra_splits)
        max_n = max_n or max(example.n for examples in splits.values() for example in examples)
        datasets = {name: QueensDataset(examples, max_n=max_n) for name, examples in splits.items()}
        super().__init__(datasets, batch_size=batch_size)


class SortedTranslationDataModule(SplitDataModule):
    def __init__(
        self,
        *,
        train_len: int,
        ood_lens: tuple[int, ...],
        n_train: int,
        n_val: int,
        n_test: int,
        vocab: int,
        seed: int,
        batch_size: int,
        extra_splits: Mapping[str, list[SortedExample]] | None = None,
    ) -> None:
        splits = make_sorted_splits(
            train_len=train_len,
            ood_lens=ood_lens,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            vocab=vocab,
            seed=seed,
        )
        if extra_splits:
            splits.update(extra_splits)
        datasets = {name: SortedTranslationDataset(examples) for name, examples in splits.items()}
        super().__init__(datasets, batch_size=batch_size)
