"""Tests for the K-way multi_alphabet_dataset (sudoku/dataset.py)."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.equiv.sudoku.dataset import SudokuDataset, multi_alphabet_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_DATA = REPO_ROOT / "data" / "sudoku_tiny.npz"


@pytest.fixture(scope="module", autouse=True)
def ensure_tiny_dataset():
    """Generate the tiny fixture dataset once per test module, if not already present."""
    if not TINY_DATA.exists():
        subprocess.run(
            [sys.executable, "scripts/generate_sudoku.py", "--config", "configs/tiny.yaml"],
            cwd=REPO_ROOT,
            check=True,
        )


def test_full_multi_alphabet_dataset_is_k_way_concatenation():
    """With no n_puzzles cap, the multi-alphabet dataset's length is K times one alphabet's."""
    letters = ["A", "B", "C", "D"]
    multi = multi_alphabet_dataset(TINY_DATA, "train", letters)
    single = SudokuDataset(TINY_DATA, "train", "A")
    assert len(multi) == len(letters) * len(single)


def test_non_contiguous_letters_keep_stable_offsets():
    """Skipping a letter (e.g. "E") doesn't shift the offsets of letters after it."""
    # "F" skips "E" -- its tokens must still be offset*5 (its fixed
    # position), not offset*1 (its position within this particular list).
    a_view = multi_alphabet_dataset(TINY_DATA, "val", ["A", "F"])
    # first len(val) items are alphabet A (identity), next are alphabet F
    n = len(SudokuDataset(TINY_DATA, "val", "A"))
    puzzle_a, solution_a, mask_a = a_view[0]
    puzzle_f, solution_f, mask_f = a_view[n]
    assert (mask_a == mask_f).all()
    given_a = puzzle_a[puzzle_a != 0]
    given_f = puzzle_f[puzzle_f != 0]
    assert (given_f == given_a + 45).all()  # F is the 6th letter -> offset 5*9=45


def test_n_puzzles_subset_shared_identically_across_letters():
    """The n_puzzles subset picks the same underlying puzzle indices for every letter."""
    letters = ["A", "B", "C"]
    multi = multi_alphabet_dataset(TINY_DATA, "train", letters, n_puzzles=20, seed=0)
    assert len(multi) == 20 * len(letters)

    # same underlying puzzle indices for every letter: compare blank masks
    # (alphabet-independent) across the three equal-length chunks
    n = 20
    mask_a = [multi[i][2] for i in range(0, n)]
    mask_b = [multi[i][2] for i in range(n, 2 * n)]
    mask_c = [multi[i][2] for i in range(2 * n, 3 * n)]
    for ma, mb, mc in zip(mask_a, mask_b, mask_c):
        assert (ma == mb).all()
        assert (ma == mc).all()


def test_n_puzzles_subset_is_deterministic_given_seed():
    """Two calls with the same seed draw the identical n_puzzles subset."""
    m1 = multi_alphabet_dataset(TINY_DATA, "train", ["A", "B"], n_puzzles=10, seed=7)
    m2 = multi_alphabet_dataset(TINY_DATA, "train", ["A", "B"], n_puzzles=10, seed=7)
    for i in range(len(m1)):
        p1, s1, _ = m1[i]
        p2, s2, _ = m2[i]
        assert (p1 == p2).all() and (s1 == s2).all()
