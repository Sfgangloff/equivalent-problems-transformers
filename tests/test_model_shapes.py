"""Tests for the classifier-head model's forward pass (model/transformer.py)."""

import torch

from src.equiv.model.transformer import SudokuTransformer
from src.equiv.sudoku.alphabets import NUM_CLASSES, VOCAB_SIZE


def _random_tokens(batch_size: int) -> torch.Tensor:
    """A batch of random valid token ids, shaped (batch_size, 81)."""
    return torch.randint(0, VOCAB_SIZE, (batch_size, 81))


def test_single_pass_forward_shape():
    """A single-iteration forward pass returns one (B, 81, NUM_CLASSES) logits tensor, no NaNs."""
    model = SudokuTransformer(d_model=32, n_layers=2, n_heads=4, dim_feedforward=64)
    tokens = _random_tokens(4)
    outputs = model(tokens)
    assert len(outputs) == 1
    assert outputs[0].shape == (4, 81, NUM_CLASSES)
    assert not torch.isnan(outputs[0]).any()


def test_iterative_forward_returns_one_logits_tensor_per_step():
    """With num_iterations=3, forward returns exactly 3 logits tensors, one per unrolled step."""
    model = SudokuTransformer(
        d_model=32, n_layers=2, n_heads=4, dim_feedforward=64, num_iterations=3
    )
    tokens = _random_tokens(2)
    outputs = model(tokens)
    assert len(outputs) == 3
    for logits in outputs:
        assert logits.shape == (2, 81, NUM_CLASSES)


def test_given_cells_are_never_overwritten_across_iterations():
    """forward() does not mutate its input tensor in place across unrolled iterations."""
    model = SudokuTransformer(
        d_model=16, n_layers=1, n_heads=2, dim_feedforward=32, num_iterations=2
    )
    tokens = _random_tokens(1)
    tokens[0, 0] = 5  # a "given" (non-blank) cell
    # forward should not mutate the input tensor in place
    original = tokens.clone()
    model(tokens)
    assert torch.equal(tokens, original)
