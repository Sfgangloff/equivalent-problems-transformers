"""Alphabet relabeling: the vocabulary-renaming transform under study.

Sudoku puzzles/solutions are generated once, in the canonical alphabet
(digits 1-9, 0 = blank). Each alphabet is a disjoint 9-wide block of token
ids: alphabet "A" is the identity (offset 0), "B" is the SAME puzzles with
every non-blank digit shifted by +9 (10-18), "C" by +18, and so on. All
alphabets share one token vocabulary (and therefore one embedding table,
see model/transformer.py), so a model trained on only some alphabets still
has (untrained) embedding/output-head rows for every token in the others.

`ALPHABETS`/`VOCAB_SIZE`/`NUM_CLASSES` remain the original 2-alphabet
(A, B) values for backward compatibility -- every existing call site that
doesn't pass `alphabets=`/`k=` explicitly keeps working unchanged. Use
`make_alphabets(k)`/`vocab_size_for(k)`/`num_classes_for(k)` for the
multi-alphabet (K > 2) experiments.
"""

from __future__ import annotations

from typing import Sequence

BLANK = 0
BLOCK_SIZE = 9
ALPHABET_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # up to 26 alphabets


def make_alphabets(k: int) -> dict[str, int]:
    """First k letters -> disjoint 9-wide offset blocks.
    make_alphabets(2) == {"A": 0, "B": 9} == ALPHABETS, by construction."""
    if not 1 <= k <= len(ALPHABET_LETTERS):
        raise ValueError(f"k must be in [1, {len(ALPHABET_LETTERS)}], got {k}")
    return {ALPHABET_LETTERS[i]: i * BLOCK_SIZE for i in range(k)}


def vocab_size_for(k: int) -> int:
    return 1 + k * BLOCK_SIZE  # blank + k blocks of 9 digits


def num_classes_for(k: int) -> int:
    return k * BLOCK_SIZE  # output head predicts the literal token id, blank is never a target


def offset_for_letter(letter: str) -> int:
    """Offset for a single letter by its fixed position in ALPHABET_LETTERS
    (A=0, B=9, C=18, ...), independent of which other letters are in use --
    lets experiments use a non-contiguous subset (e.g. skip "E" to reserve
    it as a held-out alphabet) while keeping each letter's offset stable."""
    return ALPHABET_LETTERS.index(letter) * BLOCK_SIZE


A_OFFSET = 0
B_OFFSET = 9
ALPHABETS = make_alphabets(2)  # {"A": 0, "B": 9}, unchanged from before

VOCAB_SIZE = vocab_size_for(2)  # 19: blank + alphabet A digits (1-9) + alphabet B digits (10-18)
NUM_CLASSES = num_classes_for(2)  # 18
# target class index for token t (t is 1..NUM_CLASSES): class = t - 1


def relabel(board: Sequence[int], alphabet: str, alphabets: dict[str, int] = ALPHABETS) -> list[int]:
    """Map a canonical (1-9/0) board into the given alphabet's token ids."""
    offset = alphabets[alphabet]
    return [BLANK if d == BLANK else d + offset for d in board]


def to_canonical(board: Sequence[int], alphabet: str, alphabets: dict[str, int] = ALPHABETS) -> list[int]:
    """Inverse of relabel: map alphabet tokens back to canonical 1-9/0."""
    offset = alphabets[alphabet]
    return [BLANK if t == BLANK else t - offset for t in board]


def class_range(alphabet: str, alphabets: dict[str, int] = ALPHABETS) -> tuple[int, int]:
    """(lo, hi) exclusive output-head class-index range for this alphabet's
    9 digits. A puzzle's own alphabet is always known unambiguously from its
    given (non-blank) clues, so scoring should always be restricted to this
    range: an unrestricted argmax over all NUM_CLASSES lets *other*
    alphabets' classes compete, and once transplant() makes two alphabets'
    rows numerically identical for the same digit, argmax's tie-breaking
    (lowest index wins) silently and systematically favors whichever
    alphabet has the lower token offset -- not genuine model behavior."""
    offset = alphabets[alphabet]
    return offset, offset + BLOCK_SIZE
