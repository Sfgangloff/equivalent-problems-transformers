"""PyTorch Dataset over the base Sudoku puzzle set, with the alphabet
relabeling applied lazily per __getitem__.

Only one base dataset ever exists on disk (digits 1-9, produced by
scripts/generate_sudoku.py). Alphabet A, alphabet B, and the union of both
are all views over that same file/indices, so they can never drift out of
sync with each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .alphabets import relabel

Split = Literal["train", "val", "test"]
_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


class SudokuDataset(Dataset):
    """One (puzzle, solution, blank_mask) view of the base dataset, in the
    given alphabet ("A" or "B")."""

    def __init__(self, base_path: str | Path, split: Split, alphabet: str):
        data = np.load(base_path)
        mask = data["split"] == _SPLIT_CODE[split]
        self.puzzles = data["puzzles"][mask]
        self.solutions = data["solutions"][mask]
        self.alphabet = alphabet

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        puzzle = relabel(self.puzzles[idx].tolist(), self.alphabet)
        solution = relabel(self.solutions[idx].tolist(), self.alphabet)
        puzzle_t = torch.tensor(puzzle, dtype=torch.long)
        solution_t = torch.tensor(solution, dtype=torch.long)
        blank_mask = puzzle_t == 0
        return puzzle_t, solution_t, blank_mask


def union_dataset(base_path: str | Path, split: Split) -> ConcatDataset:
    """Alphabet-A and alphabet-B views of the SAME underlying puzzles,
    concatenated -- used to train the mixed-alphabet reference model."""
    return ConcatDataset(
        [SudokuDataset(base_path, split, "A"), SudokuDataset(base_path, split, "B")]
    )
