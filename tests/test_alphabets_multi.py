"""Tests for the K-way (K>2) multi-alphabet generalization (sudoku/alphabets.py)."""

from src.equiv.sudoku.alphabets import (
    ALPHABETS,
    NUM_CLASSES,
    VOCAB_SIZE,
    class_range,
    make_alphabets,
    num_classes_for,
    relabel,
    to_canonical,
    vocab_size_for,
)


def test_make_alphabets_2_matches_original_constants():
    """make_alphabets(2) reproduces the original fixed A/B constants exactly."""
    assert make_alphabets(2) == ALPHABETS
    assert vocab_size_for(2) == VOCAB_SIZE
    assert num_classes_for(2) == NUM_CLASSES


def test_make_alphabets_k():
    """make_alphabets(k) assigns the first k letters disjoint 9-wide offset blocks."""
    assert make_alphabets(1) == {"A": 0}
    assert make_alphabets(6) == {"A": 0, "B": 9, "C": 18, "D": 27, "E": 36, "F": 45}
    assert vocab_size_for(6) == 1 + 6 * 9
    assert num_classes_for(6) == 6 * 9


def test_class_range_disjoint_across_k_alphabets():
    """Every alphabet's class_range is disjoint and together they cover all classes."""
    alphabets = make_alphabets(6)
    ranges = [class_range(letter, alphabets) for letter in alphabets]
    all_indices = set()
    for lo, hi in ranges:
        indices = set(range(lo, hi))
        assert indices.isdisjoint(all_indices)
        all_indices |= indices
    assert all_indices == set(range(num_classes_for(6)))


def test_relabel_and_to_canonical_round_trip_for_k_alphabets():
    """relabel then to_canonical recovers the original board, for any of the K alphabets."""
    alphabets = make_alphabets(6)
    board = [0, 1, 5, 9]
    for letter in alphabets:
        relabeled = relabel(board, letter, alphabets)
        assert to_canonical(relabeled, letter, alphabets) == board


def test_relabel_default_kwarg_still_uses_2_alphabet_scheme():
    """relabel() with no explicit `alphabets=` still uses the original A/B scheme."""
    # no `alphabets=` passed -- must match the original A/B behavior exactly
    assert relabel([0, 1, 9], "B") == [0, 10, 18]
