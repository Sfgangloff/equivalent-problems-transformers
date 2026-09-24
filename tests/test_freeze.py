"""Tests for the frozen-body finetuning setup (finetune.freeze_for_finetune)."""

import torch

from src.equiv.finetune import freeze_for_finetune
from src.equiv.model.transformer import SudokuTransformer
from src.equiv.sudoku.alphabets import class_range, offset_for_letter, vocab_size_for, num_classes_for


def _model_for_k(k: int) -> SudokuTransformer:
    """A small, deterministically-initialized SudokuTransformer sized for `k` alphabets."""
    torch.manual_seed(0)
    return SudokuTransformer(
        d_model=16, n_layers=1, n_heads=2, dim_feedforward=32,
        vocab_size=vocab_size_for(k), num_classes=num_classes_for(k),
    )


def test_freeze_leaves_non_target_rows_bit_identical_after_one_step():
    """A key regression test: one optimizer step
    must change ONLY target_alphabet's rows -- everything else (the shared
    encoder body, and every other alphabet's embedding/output-head rows)
    must be bit-identical before and after, and weight_decay=0 must be used
    so AdamW's decoupled decay doesn't silently shrink the frozen rows."""
    target = "E"  # 5th letter, k=7 covers A..G so E's rows exist but aren't in training data
    k = 7
    model = _model_for_k(k)
    freeze_for_finetune(model, target)

    before = {name: p.clone() for name, p in model.named_parameters()}

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=0.1, weight_decay=0.0
    )

    lo, hi = class_range(target, {target: offset_for_letter(target)})
    tok_lo, tok_hi = lo + 1, hi + 1

    tokens = torch.randint(0, vocab_size_for(k), (4, 81))
    # E's own tokens must actually appear as input, or its embedding rows are
    # simply unused (correctly zero-gradient for a mundane reason unrelated
    # to masking) and this test can't distinguish "frozen" from "unused".
    tokens[:, :9] = torch.arange(tok_lo, tok_hi)
    tokens[:, 9:] = 0  # rest blank, so loss covers every remaining cell/class
    logits = model(tokens)[-1]
    # force targets into E's class range too, so output_head's E-rows get
    # a guaranteed, non-accidental gradient signal to update against.
    target_classes = torch.randint(lo, hi, (4, 81))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, num_classes_for(k)), target_classes.reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for name, p in model.named_parameters():
        if name == "token_embedding.weight":
            assert not torch.equal(p[tok_lo:tok_hi], before[name][tok_lo:tok_hi]), "target embedding rows should change"
            mask = torch.ones(p.shape[0], dtype=torch.bool)
            mask[tok_lo:tok_hi] = False
            assert torch.equal(p[mask], before[name][mask]), "non-target embedding rows must be untouched"
        elif name == "output_head.weight":
            assert not torch.equal(p[lo:hi], before[name][lo:hi]), "target output rows should change"
            mask = torch.ones(p.shape[0], dtype=torch.bool)
            mask[lo:hi] = False
            assert torch.equal(p[mask], before[name][mask]), "non-target output-head rows must be untouched"
        elif name == "output_head.bias":
            assert not torch.equal(p[lo:hi], before[name][lo:hi]), "target output bias should change"
            mask = torch.ones(p.shape[0], dtype=torch.bool)
            mask[lo:hi] = False
            assert torch.equal(p[mask], before[name][mask]), "non-target output-head bias must be untouched"
        else:
            # encoder + row/col/box positional embeddings: fully frozen
            assert torch.equal(p, before[name]), f"{name} should be fully frozen but changed"


def test_frozen_params_have_requires_grad_false():
    """The shared encoder and positional embeddings are fully frozen (requires_grad=False)."""
    model = _model_for_k(3)
    freeze_for_finetune(model, "A")
    for p in model.encoder.parameters():
        assert not p.requires_grad
    for emb in (model.row_embedding, model.col_embedding, model.box_embedding):
        for p in emb.parameters():
            assert not p.requires_grad
    # token_embedding/output_head stay requires_grad=True overall (masked via hook, not requires_grad)
    assert model.token_embedding.weight.requires_grad
    assert model.output_head.weight.requires_grad
