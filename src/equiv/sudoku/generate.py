"""Full-grid generation and puzzle carving.

Generates a random full valid 9x9 grid, then removes cells one at a time
(checking after each removal that the puzzle still has a unique solution)
down to a target clue count. Moderate difficulty (many clues) is the point:
this is a controlled testbed for the alphabet-equivalence study, not a
sudoku-solving benchmark.
"""

from __future__ import annotations

import random

from .solver import _find_mrv_cell, _init_masks, box_index, count_solutions


def generate_full_grid(rng: random.Random) -> list[int]:
    board = [0] * 81
    row_mask, col_mask, box_mask = [0] * 9, [0] * 9, [0] * 9
    ok = _fill_recursive(board, row_mask, col_mask, box_mask, rng)
    if not ok:
        raise RuntimeError("failed to generate a full grid (should not happen)")
    return board


def _fill_recursive(
    board: list[int],
    row_mask: list[int],
    col_mask: list[int],
    box_mask: list[int],
    rng: random.Random,
) -> bool:
    found = _find_mrv_cell(board, row_mask, col_mask, box_mask)
    if found is None:
        return True
    idx, candidates = found
    if candidates == 0:
        return False

    r, c = divmod(idx, 9)
    b = box_index(r, c)
    digits = [d + 1 for d in range(9) if candidates & (1 << d)]
    rng.shuffle(digits)
    for digit in digits:
        bit = 1 << (digit - 1)
        board[idx] = digit
        row_mask[r] |= bit
        col_mask[c] |= bit
        box_mask[b] |= bit

        if _fill_recursive(board, row_mask, col_mask, box_mask, rng):
            return True

        board[idx] = 0
        row_mask[r] &= ~bit
        col_mask[c] &= ~bit
        box_mask[b] &= ~bit
    return False


def carve_puzzle(full_grid: list[int], rng: random.Random, target_clues: int = 40) -> list[int]:
    """Remove cells from a full grid while a uniqueness check passes, until
    `target_clues` remain (or no more cells can safely be removed)."""
    puzzle = list(full_grid)
    cell_order = list(range(81))
    rng.shuffle(cell_order)
    clues = 81
    for idx in cell_order:
        if clues <= target_clues:
            break
        saved = puzzle[idx]
        puzzle[idx] = 0
        if count_solutions(puzzle, limit=2) == 1:
            clues -= 1
        else:
            puzzle[idx] = saved
    return puzzle


def generate_puzzle(rng: random.Random, target_clues: int = 40) -> tuple[list[int], list[int]]:
    """Return (puzzle, solution), both in the canonical 1-9 alphabet."""
    solution = generate_full_grid(rng)
    puzzle = carve_puzzle(solution, rng, target_clues=target_clues)
    return puzzle, solution
