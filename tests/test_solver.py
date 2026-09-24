"""Tests for the Sudoku generator and backtracking solver (sudoku/generate.py, sudoku/solver.py)."""

import random

from src.equiv.sudoku.generate import generate_full_grid, generate_puzzle
from src.equiv.sudoku.solver import (
    count_solutions,
    is_solved,
    is_valid_board,
    solve,
)


def test_full_grid_is_valid_and_complete():
    """generate_full_grid produces an 81-cell, fully-filled, valid Sudoku grid."""
    rng = random.Random(0)
    for seed in range(10):
        rng.seed(seed)
        grid = generate_full_grid(rng)
        assert len(grid) == 81
        assert 0 not in grid
        assert is_valid_board(grid)
        assert is_solved(grid)


def test_generated_puzzles_have_unique_solution_matching_ground_truth():
    """Every generated puzzle has exactly one solution, and it matches the recorded ground truth."""
    rng = random.Random(42)
    for _ in range(20):
        puzzle, solution = generate_puzzle(rng, target_clues=40)
        assert is_solved(solution)
        assert count_solutions(puzzle, limit=2) == 1
        solved = solve(puzzle)
        assert solved == solution
        # every given clue in the puzzle must match the solution
        for p, s in zip(puzzle, solution):
            if p != 0:
                assert p == s


def test_carving_reaches_roughly_target_clue_count():
    """carve_puzzle removes cells down to close to (but possibly slightly above) target_clues."""
    rng = random.Random(7)
    puzzle, _ = generate_puzzle(rng, target_clues=40)
    clues = sum(1 for d in puzzle if d != 0)
    # carving stops early if uniqueness would break, so allow some slack above target
    assert 40 <= clues <= 55


def test_unsolvable_board_returns_none():
    """A board engineered to have zero legal candidates for one cell is correctly unsolvable."""
    # Start from a real solved grid (valid givens, no duplicates) so the
    # solver's masks reflect a consistent partial assignment. Blank exactly
    # one cell X (whose correct digit is d), then overwrite a different cell
    # in X's row with d too. Now d becomes "used" in X's row, and since d was
    # the only digit missing from X's row+col+box in a solved grid, X ends up
    # with zero candidates -- fast, deterministic dead-end (no need for the
    # deep backtracking a near-empty board would require).
    rng = random.Random(1)
    solution = generate_full_grid(rng)
    board = list(solution)
    x = 0
    d = board[x]
    board[x] = 0
    row_start = (x // 9) * 9
    other = next(i for i in range(row_start, row_start + 9) if i != x)
    board[other] = d

    assert solve(board) is None
    assert count_solutions(board, limit=2) == 0


def test_is_valid_board_detects_duplicates():
    """A board with the same digit twice in one column is flagged invalid."""
    board = [0] * 81
    board[0] = 5
    board[9] = 5  # same column
    assert not is_valid_board(board)
