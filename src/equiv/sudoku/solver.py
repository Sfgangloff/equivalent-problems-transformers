"""Bitmask backtracking Sudoku solver.

Boards are flat length-81 lists of ints, row-major, 0 = blank, else 1-9.
Always operates in the canonical (1-9) alphabet; alphabet relabeling is
applied only at the dataset layer, never here.
"""

from __future__ import annotations

FULL_MASK = (1 << 9) - 1  # bit (d-1) set means digit d is used


def box_index(row: int, col: int) -> int:
    """Map a 0-indexed (row, col) cell to its 0-8 3x3-box index, row-major."""
    return (row // 3) * 3 + (col // 3)


def _init_masks(board: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Build the row/column/box used-digit bitmasks implied by a board's clues."""
    row_mask = [0] * 9
    col_mask = [0] * 9
    box_mask = [0] * 9
    for r in range(9):
        for c in range(9):
            d = board[r * 9 + c]
            if d:
                bit = 1 << (d - 1)
                row_mask[r] |= bit
                col_mask[c] |= bit
                box_mask[box_index(r, c)] |= bit
    return row_mask, col_mask, box_mask


def _find_mrv_cell(
    board: list[int], row_mask: list[int], col_mask: list[int], box_mask: list[int]
) -> tuple[int, int] | None:
    """Return (index, candidate_mask) for the blank cell with fewest legal
    candidates, or None if the board has no blanks left. A candidate_mask of
    0 means a dead end (some blank cell has no legal digit)."""
    best = -1
    best_mask = 0
    best_count = 10
    for i in range(81):
        if board[i]:
            continue
        r, c = divmod(i, 9)
        used = row_mask[r] | col_mask[c] | box_mask[box_index(r, c)]
        candidates = FULL_MASK & ~used
        count = bin(candidates).count("1")
        if count == 0:
            return i, 0
        if count < best_count:
            best_count = count
            best = i
            best_mask = candidates
            if count == 1:
                break
    if best == -1:
        return None
    return best, best_mask


def count_solutions(board: list[int], limit: int = 2) -> int:
    """Count solutions up to `limit` (stops early once reached)."""
    b = list(board)
    row_mask, col_mask, box_mask = _init_masks(b)
    return _count_recursive(b, row_mask, col_mask, box_mask, limit)


def _count_recursive(
    board: list[int],
    row_mask: list[int],
    col_mask: list[int],
    box_mask: list[int],
    limit: int,
) -> int:
    """Backtracking search counting solutions in place, stopping once `limit` is reached."""
    found = _find_mrv_cell(board, row_mask, col_mask, box_mask)
    if found is None:
        return 1
    idx, candidates = found
    if candidates == 0:
        return 0

    r, c = divmod(idx, 9)
    b = box_index(r, c)
    count = 0
    mask = candidates
    while mask:
        bit = mask & (-mask)
        mask ^= bit
        digit = bit.bit_length()

        board[idx] = digit
        row_mask[r] |= bit
        col_mask[c] |= bit
        box_mask[b] |= bit

        count += _count_recursive(board, row_mask, col_mask, box_mask, limit - count)

        board[idx] = 0
        row_mask[r] &= ~bit
        col_mask[c] &= ~bit
        box_mask[b] &= ~bit

        if count >= limit:
            return count
    return count


def solve(board: list[int]) -> list[int] | None:
    """Return the first solution found, or None if unsolvable."""
    b = list(board)
    row_mask, col_mask, box_mask = _init_masks(b)
    if _solve_recursive(b, row_mask, col_mask, box_mask):
        return b
    return None


def _solve_recursive(
    board: list[int], row_mask: list[int], col_mask: list[int], box_mask: list[int]
) -> bool:
    """Backtracking search filling `board` in place; returns whether it succeeded."""
    found = _find_mrv_cell(board, row_mask, col_mask, box_mask)
    if found is None:
        return True
    idx, candidates = found
    if candidates == 0:
        return False

    r, c = divmod(idx, 9)
    b = box_index(r, c)
    mask = candidates
    while mask:
        bit = mask & (-mask)
        mask ^= bit
        digit = bit.bit_length()

        board[idx] = digit
        row_mask[r] |= bit
        col_mask[c] |= bit
        box_mask[b] |= bit

        if _solve_recursive(board, row_mask, col_mask, box_mask):
            return True

        board[idx] = 0
        row_mask[r] &= ~bit
        col_mask[c] &= ~bit
        box_mask[b] &= ~bit
    return False


def _groups(board: list[int]) -> list[list[int]]:
    """Return all 27 constraint groups (9 rows, 9 columns, 9 boxes) as digit lists."""
    rows = [board[r * 9 : (r + 1) * 9] for r in range(9)]
    cols = [[board[r * 9 + c] for r in range(9)] for c in range(9)]
    boxes: list[list[int]] = [[] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            boxes[box_index(r, c)].append(board[r * 9 + c])
    return rows + cols + boxes


def is_valid_board(board: list[int]) -> bool:
    """No digit repeated within any row/col/box (blanks ignored)."""
    for group in _groups(board):
        digits = [d for d in group if d != 0]
        if len(digits) != len(set(digits)):
            return False
    return True


def is_complete(board: list[int]) -> bool:
    """Whether every cell is filled (no blanks left), independent of validity."""
    return 0 not in board


def is_solved(board: list[int]) -> bool:
    """Whether the board is both complete and a valid Sudoku solution."""
    return is_complete(board) and is_valid_board(board)
