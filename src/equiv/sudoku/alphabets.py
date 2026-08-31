"""Alphabet relabeling: the vocabulary-renaming transform under study.

Sudoku puzzles/solutions are generated once, in the canonical alphabet
(digits 1-9, 0 = blank). Alphabet "B" is the SAME puzzles with every
non-blank digit shifted by +9 (10-18). Both alphabets share one token
vocabulary (and therefore one embedding table, see model/transformer.py),
so a model trained on only one alphabet still has an (untrained) embedding
row and output-head row for every token in the other alphabet.
"""

from __future__ import annotations

from typing import Sequence

BLANK = 0
A_OFFSET = 0
B_OFFSET = 9
ALPHABETS = {"A": A_OFFSET, "B": B_OFFSET}

VOCAB_SIZE = 1 + 9 + 9  # blank + alphabet A digits (1-9) + alphabet B digits (10-18)
NUM_CLASSES = 18  # output head predicts the literal token id, blank is never a target
# target class index for token t (t is 1..18): class = t - 1


def relabel(board: Sequence[int], alphabet: str) -> list[int]:
    """Map a canonical (1-9/0) board into the given alphabet's token ids."""
    offset = ALPHABETS[alphabet]
    return [BLANK if d == BLANK else d + offset for d in board]


def to_canonical(board: Sequence[int], alphabet: str) -> list[int]:
    """Inverse of relabel: map alphabet tokens back to canonical 1-9/0."""
    offset = ALPHABETS[alphabet]
    return [BLANK if t == BLANK else t - offset for t in board]


def class_range(alphabet: str) -> tuple[int, int]:
    """(lo, hi) exclusive output-head class-index range for this alphabet's
    9 digits. A puzzle's own alphabet is always known unambiguously from its
    given (non-blank) clues, so scoring should always be restricted to this
    range: an unrestricted argmax over all NUM_CLASSES lets the *other*
    alphabet's classes compete, and once transplant() makes both alphabets'
    rows numerically identical for the same digit, argmax's tie-breaking
    (lowest index wins) silently and systematically favors whichever
    alphabet has the lower token offset -- not genuine model behavior."""
    offset = ALPHABETS[alphabet]
    return offset, offset + 9
