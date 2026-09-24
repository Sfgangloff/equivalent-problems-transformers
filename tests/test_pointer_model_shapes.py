"""Tests for the pointer-head model's forward pass (model/pointer_transformer.py)."""

import torch

from src.equiv.model.pointer_transformer import SudokuPointerTransformer

VOCAB_SIZE = 19


def _random_batch(batch_size: int):
    """A random (tokens, candidate_tokens, candidate_mask) batch, all 9 candidate slots valid."""
    tokens = torch.randint(0, VOCAB_SIZE, (batch_size, 81))
    # 9 distinct candidate tokens per puzzle (all valid, no padding)
    candidate_tokens = torch.stack([torch.randperm(VOCAB_SIZE)[:9] for _ in range(batch_size)])
    candidate_mask = torch.ones(batch_size, 9, dtype=torch.bool)
    return tokens, candidate_tokens, candidate_mask


def test_single_pass_forward_shape():
    """A single-iteration forward pass returns one (B, 81, 9) logits tensor, no NaNs."""
    model = SudokuPointerTransformer(d_model=32, n_layers=2, n_heads=4, dim_feedforward=64, vocab_size=VOCAB_SIZE)
    tokens, candidate_tokens, candidate_mask = _random_batch(4)
    outputs = model(tokens, candidate_tokens, candidate_mask)
    assert len(outputs) == 1
    assert outputs[0].shape == (4, 81, 9)
    assert not torch.isnan(outputs[0]).any()


def test_iterative_forward_returns_one_logits_tensor_per_step():
    """With num_iterations=3, forward returns exactly 3 logits tensors, one per unrolled step."""
    model = SudokuPointerTransformer(
        d_model=32, n_layers=2, n_heads=4, dim_feedforward=64, vocab_size=VOCAB_SIZE, num_iterations=3
    )
    tokens, candidate_tokens, candidate_mask = _random_batch(2)
    outputs = model(tokens, candidate_tokens, candidate_mask)
    assert len(outputs) == 3
    for logits in outputs:
        assert logits.shape == (2, 81, 9)


def test_masked_candidates_never_win_argmax():
    """Masked-out candidate slots are never chosen as the predicted slot."""
    model = SudokuPointerTransformer(d_model=32, n_layers=2, n_heads=4, dim_feedforward=64, vocab_size=VOCAB_SIZE)
    tokens, candidate_tokens, candidate_mask = _random_batch(4)
    # mask out candidate slots 5-8, leaving only 5 valid candidates
    candidate_mask[:, 5:] = False
    logits = model(tokens, candidate_tokens, candidate_mask)[-1]
    pred_slot = logits.argmax(dim=-1)
    assert (pred_slot < 5).all()


def test_without_query_projection_still_runs():
    """use_query_projection=False skips building query_proj and still runs end-to-end."""
    model = SudokuPointerTransformer(
        d_model=32, n_layers=2, n_heads=4, dim_feedforward=64, vocab_size=VOCAB_SIZE, use_query_projection=False
    )
    assert not hasattr(model, "query_proj")
    tokens, candidate_tokens, candidate_mask = _random_batch(4)
    outputs = model(tokens, candidate_tokens, candidate_mask)
    assert outputs[0].shape == (4, 81, 9)
    assert not torch.isnan(outputs[0]).any()


def test_given_cells_are_never_overwritten_across_iterations():
    """forward() does not mutate its input tensor in place across unrolled iterations."""
    model = SudokuPointerTransformer(
        d_model=16, n_layers=1, n_heads=2, dim_feedforward=32, vocab_size=VOCAB_SIZE, num_iterations=2
    )
    tokens, candidate_tokens, candidate_mask = _random_batch(1)
    tokens[0, 0] = 5  # a "given" cell
    original = tokens.clone()
    model(tokens, candidate_tokens, candidate_mask)
    assert torch.equal(tokens, original)
