import random

from src.equiv.sudoku.alphabets import (
    ALPHABETS,
    BLANK,
    VOCAB_SIZE,
    relabel,
    to_canonical,
)
from src.equiv.sudoku.generate import generate_puzzle


def test_relabel_is_a_bijection_round_trip():
    rng = random.Random(3)
    puzzle, solution = generate_puzzle(rng, target_clues=40)
    for alphabet in ALPHABETS:
        for board in (puzzle, solution):
            relabeled = relabel(board, alphabet)
            assert to_canonical(relabeled, alphabet) == board


def test_alphabet_a_is_identity_on_canonical_digits():
    board = [0, 1, 9, 5]
    assert relabel(board, "A") == board


def test_alphabet_b_shifts_by_nine_and_preserves_blanks():
    board = [0, 1, 9, 5]
    assert relabel(board, "B") == [0, 10, 18, 14]


def test_alphabets_share_only_the_blank_token():
    a_tokens = {relabel([d], "A")[0] for d in range(1, 10)}
    b_tokens = {relabel([d], "B")[0] for d in range(1, 10)}
    assert a_tokens.isdisjoint(b_tokens)
    assert BLANK not in a_tokens and BLANK not in b_tokens
    assert len(a_tokens | b_tokens | {BLANK}) == VOCAB_SIZE
