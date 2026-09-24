"""Tests for the A/B two-alphabet SudokuDataset and union_dataset (sudoku/dataset.py)."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.equiv.sudoku.dataset import SudokuDataset, union_dataset

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


def test_splits_are_disjoint_and_cover_all_puzzles():
    """train+val+test sizes sum to the full puzzle count, with non-empty val/test."""
    train = SudokuDataset(TINY_DATA, "train", "A")
    val = SudokuDataset(TINY_DATA, "val", "A")
    test = SudokuDataset(TINY_DATA, "test", "A")
    assert len(train) + len(val) + len(test) == 500
    assert len(val) > 0 and len(test) > 0


def test_alphabet_a_and_b_are_the_same_underlying_puzzles_relabeled():
    """A and B datasets share the same blank pattern and B's tokens are A's tokens +9."""
    a = SudokuDataset(TINY_DATA, "train", "A")
    b = SudokuDataset(TINY_DATA, "train", "B")
    assert len(a) == len(b)
    for i in range(len(a)):
        puzzle_a, solution_a, mask_a = a[i]
        puzzle_b, solution_b, mask_b = b[i]
        assert (mask_a == mask_b).all()
        blank_a = puzzle_a == 0
        blank_b = puzzle_b == 0
        assert (blank_a == blank_b).all()
        given_a = puzzle_a[~blank_a]
        given_b = puzzle_b[~blank_b]
        assert (given_b == given_a + 9).all()
        assert (solution_b == solution_a + 9).all()


def test_alphabet_tokens_are_in_the_expected_ranges():
    """A's solution tokens fall in [1,9] and B's fall in [10,18]."""
    a = SudokuDataset(TINY_DATA, "train", "A")
    b = SudokuDataset(TINY_DATA, "train", "B")
    for i in range(5):
        puzzle_a, solution_a, _ = a[i]
        assert solution_a.min() >= 1 and solution_a.max() <= 9
        puzzle_b, solution_b, _ = b[i]
        assert solution_b.min() >= 10 and solution_b.max() <= 18


def test_union_dataset_is_concatenation_of_a_and_b():
    """union_dataset's length equals len(A) + len(B)."""
    a = SudokuDataset(TINY_DATA, "val", "A")
    b = SudokuDataset(TINY_DATA, "val", "B")
    union = union_dataset(TINY_DATA, "val")
    assert len(union) == len(a) + len(b)
