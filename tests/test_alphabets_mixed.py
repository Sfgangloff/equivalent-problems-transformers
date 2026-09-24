"""Tests for the mixed-alphabet recombination transform (sudoku/alphabets.py)."""

from src.equiv.sudoku.alphabets import mixed_class_indices, mixed_relabel, relabel


def test_mixed_relabel_with_uniform_mapping_equals_normal_relabel():
    """A mixed mapping that sends every digit to the same letter equals plain relabel()."""
    board = [0, 1, 5, 9]
    digit_to_letter = {d: "C" for d in range(1, 10)}
    assert mixed_relabel(board, digit_to_letter) == relabel(board, "C", {"C": 18})


def test_mixed_relabel_picks_per_digit_alphabet():
    """Each digit is relabeled using its own assigned letter's offset."""
    # digit 1 -> A (offset 0), digit 5 -> B (offset 9), digit 9 -> D (offset 27)
    digit_to_letter = {1: "A", 5: "B", 9: "D"}
    board = [0, 1, 5, 9]
    assert mixed_relabel(board, digit_to_letter) == [0, 1, 14, 36]


def test_mixed_relabel_preserves_blanks():
    """Blank cells stay 0 regardless of the digit-to-letter mapping."""
    digit_to_letter = {d: "G" for d in range(1, 10)}
    assert mixed_relabel([0, 0, 3], digit_to_letter)[:2] == [0, 0]


def test_mixed_class_indices_are_9_distinct_indices():
    """The 9 per-digit class indices are always distinct, even when letters repeat."""
    digit_to_letter = {1: "A", 2: "A", 3: "B", 4: "C", 5: "D", 6: "F", 7: "G", 8: "A", 9: "B"}
    indices = mixed_class_indices(digit_to_letter)
    assert len(indices) == 9
    assert len(set(indices)) == 9  # each digit's token is a distinct class, even when letters repeat


def test_mixed_class_indices_uniform_mapping_matches_class_range():
    """A mixed mapping that sends every digit to the same letter equals that letter's class_range()."""
    from src.equiv.sudoku.alphabets import class_range

    digit_to_letter = {d: "D" for d in range(1, 10)}
    indices = mixed_class_indices(digit_to_letter)
    lo, hi = class_range("D", {"D": 27})
    assert indices == list(range(lo, hi))
