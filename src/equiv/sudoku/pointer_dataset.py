"""Dataset for the pointer/copy Sudoku model (model/pointer_transformer.py).

Each item additionally carries, per puzzle: `candidate_tokens` (the up to 9
distinct token ids present as GIVENS in that specific puzzle, padded to a
fixed length of 9), `candidate_mask` (which of those 9 slots are real vs
padding), and `target_slot` (for each blank cell, the index into
candidate_tokens matching the true solution digit -- the pointer model's
training target, replacing the classifier model's absolute class id).

A puzzle's target is only well-defined for a blank cell if the correct
digit for that cell is itself among the puzzle's own givens -- not merely
somewhere in the full solved grid. Measured directly on this repo's own
40-clue generator (see PLAN.md): 4/500 (0.80%) of puzzles are missing one
digit from their givens entirely. `filter_fully_represented=True` (the
default, used for training) eagerly drops such puzzles so every training
target is well-defined; pass False for evaluation, where scoring on the
natural, unfiltered distribution matters more than avoiding the ~0.8% of
puzzles this architecture is mechanically unable to solve by construction
(no candidate exists to point to) -- an honest limitation to measure, not
hide. `fully_represented` is still returned per-item either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from .alphabets import ALPHABETS, offset_for_letter, relabel

Split = Literal["train", "val", "test"]
_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def _fully_represented_mask(puzzles: np.ndarray) -> np.ndarray:
    """Vectorized: True where all 9 digits (1-9) appear at least once among
    a puzzle's non-blank (given) cells."""
    n = puzzles.shape[0]
    present = np.zeros((n, 10), dtype=bool)
    rows = np.repeat(np.arange(n), puzzles.shape[1])
    present[rows, puzzles.reshape(-1)] = True
    return present[:, 1:].sum(axis=1) == 9  # exclude column 0 (blank)


class PointerSudokuDataset(Dataset):
    def __init__(
        self,
        base_path: str | Path,
        split: Split,
        alphabet: str,
        alphabets: dict[str, int] = ALPHABETS,
        filter_fully_represented: bool = True,
    ):
        data = np.load(base_path)
        mask = data["split"] == _SPLIT_CODE[split]
        puzzles = data["puzzles"][mask]
        solutions = data["solutions"][mask]

        if filter_fully_represented:
            keep = _fully_represented_mask(puzzles)
            puzzles = puzzles[keep]
            solutions = solutions[keep]

        self.puzzles = puzzles
        self.solutions = solutions
        self.alphabet = alphabet
        self.alphabets = alphabets

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        puzzle = relabel(self.puzzles[idx].tolist(), self.alphabet, self.alphabets)
        solution = relabel(self.solutions[idx].tolist(), self.alphabet, self.alphabets)

        distinct_tokens = sorted(t for t in set(puzzle) if t != 0)
        fully_represented = len(distinct_tokens) == 9
        pad = 9 - len(distinct_tokens)
        candidate_tokens = distinct_tokens + [0] * pad
        candidate_mask = [True] * len(distinct_tokens) + [False] * pad
        token_to_slot = {t: i for i, t in enumerate(distinct_tokens)}

        puzzle_t = torch.tensor(puzzle, dtype=torch.long)
        solution_t = torch.tensor(solution, dtype=torch.long)
        blank_mask = puzzle_t == 0

        # -1 sentinel: this cell's true digit isn't among the puzzle's own
        # givens, so no candidate slot can point to it (only possible when
        # filter_fully_represented=False -- see module docstring).
        target_slot = [
            token_to_slot.get(solution[i], -1) if puzzle[i] == 0 else 0 for i in range(81)
        ]

        return (
            puzzle_t,
            solution_t,
            blank_mask,
            torch.tensor(candidate_tokens, dtype=torch.long),
            torch.tensor(candidate_mask, dtype=torch.bool),
            torch.tensor(target_slot, dtype=torch.long),
            fully_represented,
        )


def multi_alphabet_pointer_dataset(
    base_path: str | Path,
    split: Split,
    alphabet_letters: list[str],
    n_puzzles: int | None = None,
    seed: int = 0,
    filter_fully_represented: bool = True,
) -> ConcatDataset:
    """K-way generalization of PointerSudokuDataset, mirroring
    dataset.multi_alphabet_dataset: the SAME underlying puzzle subset
    (optionally size-limited, one seeded random sample shared across every
    letter), relabeled once per letter and concatenated. "Fully
    represented" is alphabet-invariant (depends only on which canonical
    digits are present, not their token ids), so filtering keeps the same
    underlying indices across every letter -- the shared-subset invariant
    holds after filtering too."""
    alphabets = {letter: offset_for_letter(letter) for letter in alphabet_letters}

    base = PointerSudokuDataset(base_path, split, alphabet_letters[0], alphabets, filter_fully_represented)
    if n_puzzles is None:
        indices = list(range(len(base)))
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(base), size=min(n_puzzles, len(base)), replace=False).tolist()

    views = [
        Subset(PointerSudokuDataset(base_path, split, letter, alphabets, filter_fully_represented), indices)
        for letter in alphabet_letters
    ]
    return ConcatDataset(views)
