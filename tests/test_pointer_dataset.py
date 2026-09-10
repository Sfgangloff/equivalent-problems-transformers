import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.equiv.sudoku.generate import generate_full_grid
from src.equiv.sudoku.pointer_dataset import (
    PointerSudokuDataset,
    _fully_represented_mask,
    multi_alphabet_pointer_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_DATA = REPO_ROOT / "data" / "sudoku_tiny.npz"


@pytest.fixture(scope="module", autouse=True)
def ensure_tiny_dataset():
    if not TINY_DATA.exists():
        subprocess.run(
            [sys.executable, "scripts/generate_sudoku.py", "--config", "configs/tiny.yaml"],
            cwd=REPO_ROOT,
            check=True,
        )


def _make_npz(tmp_path, puzzles, solutions, splits):
    path = tmp_path / "data.npz"
    np.savez(
        path,
        puzzles=np.array(puzzles, dtype=np.int8),
        solutions=np.array(solutions, dtype=np.int8),
        split=np.array(splits, dtype=np.int8),
    )
    return path


def test_target_slot_matches_known_answer(tmp_path):
    rng = random.Random(0)
    solution = generate_full_grid(rng)
    puzzle = list(solution)
    # blank 2 cells only -- each digit still appears ~9 times elsewhere, so
    # the puzzle stays fully represented (deterministic, unlike relying on
    # the ~0.8% natural incidence rate of an under-represented puzzle).
    puzzle[0] = 0
    puzzle[1] = 0

    path = _make_npz(tmp_path, [puzzle], [solution], [0])  # split=0 -> "train"
    ds = PointerSudokuDataset(path, "train", "A", filter_fully_represented=True)
    assert len(ds) == 1

    puzzle_t, solution_t, blank_mask, candidate_tokens, candidate_mask, target_slot, fully_represented = ds[0]
    assert fully_represented
    assert candidate_mask.all()  # exactly 9 distinct candidates, no padding
    assert set(candidate_tokens.tolist()) == set(range(1, 10))

    for i in (0, 1):
        true_digit = solution[i]
        slot = target_slot[i].item()
        assert candidate_tokens[slot].item() == true_digit


def test_puzzle_missing_a_digit_from_givens_is_filtered_when_requested(tmp_path):
    rng = random.Random(1)
    solution = generate_full_grid(rng)
    puzzle = list(solution)
    # blank every cell holding digit 9 (exactly 9 of them in a valid full
    # grid), so digit 9 is entirely absent from this puzzle's givens.
    for i in range(81):
        if solution[i] == 9:
            puzzle[i] = 0

    path = _make_npz(tmp_path, [puzzle], [solution], [0])

    filtered = PointerSudokuDataset(path, "train", "A", filter_fully_represented=True)
    assert len(filtered) == 0  # dropped: not fully represented

    unfiltered = PointerSudokuDataset(path, "train", "A", filter_fully_represented=False)
    assert len(unfiltered) == 1
    _, _, _, candidate_tokens, candidate_mask, target_slot, fully_represented = unfiltered[0]
    assert not fully_represented
    assert candidate_mask.sum().item() == 8  # only 8 distinct digits present as givens
    assert (candidate_tokens[~candidate_mask] == 0).all()  # padding slot

    # blank cells whose true answer is 9 have no valid candidate to point to
    for i in range(81):
        if solution[i] == 9 and puzzle[i] == 0:
            assert target_slot[i].item() == -1


def test_fully_represented_mask_vectorized_matches_hand_built_case():
    puzzles = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9] + [0] * 72,  # all 9 digits present
            [1, 2, 3, 4, 5, 6, 7, 8, 0] + [0] * 72,  # missing 9
        ],
        dtype=np.int8,
    )
    mask = _fully_represented_mask(puzzles)
    assert mask.tolist() == [True, False]


def test_multi_alphabet_pointer_dataset_is_k_way_concatenation_of_same_subset():
    letters = ["A", "B", "C", "D", "F", "G"]  # non-contiguous (skips E), matches train_pointer.py's real usage
    n = 20
    multi = multi_alphabet_pointer_dataset(TINY_DATA, "train", letters, n_puzzles=n, seed=0)
    assert len(multi) == n * len(letters)

    # same underlying puzzle indices for every letter: blank_mask (index 2
    # in the tuple) is alphabet-independent, compare across the K chunks
    masks = [[multi[i + k * n][2] for i in range(n)] for k in range(len(letters))]
    for other in masks[1:]:
        for a, b in zip(masks[0], other):
            assert (a == b).all()
