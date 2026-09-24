"""PyTorch Dataset over the base Sudoku puzzle set, with the alphabet
relabeling applied lazily per __getitem__.

Only one base dataset ever exists on disk (digits 1-9, produced by
scripts/generate_sudoku.py). Alphabet A, alphabet B, the union of both, and
any K-way multi-alphabet combination are all views over that same
file/indices, so they can never drift out of sync with each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from .alphabets import ALPHABETS, mixed_relabel, offset_for_letter, relabel

Split = Literal["train", "val", "test"]
_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


class SudokuDataset(Dataset):
    """One (puzzle, solution, blank_mask) view of the base dataset, in the
    given alphabet. `alphabets` maps letter -> offset (defaults to the
    2-alphabet A/B scheme); pass a wider map (see sudoku.alphabets) to use
    an alphabet beyond A/B, e.g. one built via offset_for_letter."""

    def __init__(
        self,
        base_path: str | Path,
        split: Split,
        alphabet: str,
        alphabets: dict[str, int] = ALPHABETS,
    ):
        """Load the base `.npz` puzzle set and keep only the rows for `split`."""
        data = np.load(base_path)
        mask = data["split"] == _SPLIT_CODE[split]
        self.puzzles = data["puzzles"][mask]
        self.solutions = data["solutions"][mask]
        self.alphabet = alphabet
        self.alphabets = alphabets

    def __len__(self) -> int:
        """Number of puzzles in this split."""
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (puzzle, solution, blank_mask) for puzzle `idx`, relabeled to this alphabet."""
        puzzle = relabel(self.puzzles[idx].tolist(), self.alphabet, self.alphabets)
        solution = relabel(self.solutions[idx].tolist(), self.alphabet, self.alphabets)
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


def multi_alphabet_dataset(
    base_path: str | Path,
    split: Split,
    alphabet_letters: list[str],
    n_puzzles: int | None = None,
    seed: int = 0,
) -> ConcatDataset:
    """K-way generalization of union_dataset: the SAME underlying puzzle
    subset (optionally size-limited to n_puzzles, one seeded random sample
    shared across every letter), relabeled once per letter in
    `alphabet_letters` and concatenated. Letters need not be contiguous
    from "A" (e.g. skip "E" to reserve it as a held-out alphabet) --
    offsets come from each letter's fixed position (offset_for_letter),
    not from list order, so a given letter's tokens are stable regardless
    of which other letters are included.
    """
    alphabets = {letter: offset_for_letter(letter) for letter in alphabet_letters}

    base = SudokuDataset(base_path, split, alphabet_letters[0], alphabets)
    if n_puzzles is None:
        indices = list(range(len(base)))
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(base), size=min(n_puzzles, len(base)), replace=False).tolist()

    views = [
        Subset(SudokuDataset(base_path, split, letter, alphabets), indices)
        for letter in alphabet_letters
    ]
    return ConcatDataset(views)


class MixedAlphabetDataset(Dataset):
    """Like SudokuDataset, but each digit's token comes from a possibly
    different alphabet per digit_to_letter (see alphabets.mixed_relabel):
    a combination of already-individually-trained symbols never presented
    together as one coherent alphabet during training."""

    def __init__(self, base_path: str | Path, split: Split, digit_to_letter: dict[int, str]):
        """Load the base `.npz` puzzle set and keep only the rows for `split`."""
        data = np.load(base_path)
        mask = data["split"] == _SPLIT_CODE[split]
        self.puzzles = data["puzzles"][mask]
        self.solutions = data["solutions"][mask]
        self.digit_to_letter = digit_to_letter

    def __len__(self) -> int:
        """Number of puzzles in this split."""
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (puzzle, solution, blank_mask) for puzzle `idx`, under the mixed alphabet."""
        puzzle = mixed_relabel(self.puzzles[idx].tolist(), self.digit_to_letter)
        solution = mixed_relabel(self.solutions[idx].tolist(), self.digit_to_letter)
        puzzle_t = torch.tensor(puzzle, dtype=torch.long)
        solution_t = torch.tensor(solution, dtype=torch.long)
        blank_mask = puzzle_t == 0
        return puzzle_t, solution_t, blank_mask
