"""Tests for the weight-transplant control (evaluate.py)."""

import torch

from src.equiv.evaluate import transplant
from src.equiv.model.transformer import SudokuTransformer
from src.equiv.sudoku.alphabets import class_range


def _tiny_model() -> SudokuTransformer:
    """A small, deterministically-initialized SudokuTransformer for fast tests."""
    torch.manual_seed(0)
    return SudokuTransformer(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32)


def test_transplant_makes_source_and_target_rows_identical():
    """After transplant(A, B), B's per-digit embedding/output rows equal A's."""
    model = _tiny_model()
    transplant(model, "A", "B")
    for digit in range(1, 10):
        assert torch.equal(
            model.token_embedding.weight[digit], model.token_embedding.weight[digit + 9]
        )
        assert torch.equal(
            model.output_head.weight[digit - 1], model.output_head.weight[digit - 1 + 9]
        )


def test_unrestricted_argmax_after_transplant_is_index_biased():
    """Regression test for a real bug found during local smoke testing:
    transplant() makes the source and target alphabet's rows numerically
    identical, so an UNRESTRICTED argmax over all 18 classes always breaks
    the resulting tie toward the lower class index -- i.e. always toward
    alphabet A, regardless of which alphabet is actually being evaluated.
    That made A->B transplanted cross-eval collapse to ~0 accuracy while
    B->A looked artificially perfect, for a purely index-order reason with
    no bearing on whether the model's reasoning actually transferred.
    """
    model = _tiny_model()
    transplant(model, "A", "B")
    tokens = torch.randint(0, 19, (8, 81))
    logits = model(tokens)[-1]
    unrestricted_pred_class = logits.argmax(dim=-1)
    assert (unrestricted_pred_class < 9).all()


def test_restricted_argmax_matches_across_alphabets_after_transplant():
    """class_range()-restricted argmax (what evaluate.py actually uses) must
    not exhibit the bug above: since transplant makes alphabet B's class
    rows an exact copy of alphabet A's, scoring each alphabet within its own
    class_range() should yield the identical digit-within-alphabet choice
    regardless of which alphabet's window is used."""
    model = _tiny_model()
    transplant(model, "A", "B")
    tokens = torch.randint(0, 19, (8, 81))
    logits = model(tokens)[-1]

    lo_a, hi_a = class_range("A")
    lo_b, hi_b = class_range("B")
    pred_a = logits[..., lo_a:hi_a].argmax(dim=-1)
    pred_b = logits[..., lo_b:hi_b].argmax(dim=-1)
    assert torch.equal(pred_a, pred_b)
