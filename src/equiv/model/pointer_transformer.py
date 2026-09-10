"""Pointer/copy-style Sudoku transformer: for each blank cell, predict
WHICH of the puzzle's own (up to 9) distinct given token ids belongs
there, instead of classifying into a fixed, global, per-alphabet
vocabulary (see SudokuTransformer in transformer.py).

Motivation (see results/2026-09-09_multi_alphabet_experiments.md and
PLAN.md's Experiment 2 section): SudokuTransformer's output_head is a
Linear(d_model, 9*K) with one row per digit *per known alphabet* -- a
structural ceiling that no amount of pretraining diversity K can fix,
since a genuinely new alphabet always needs new, never-trained output
rows. It also gives the model every incentive to key its predictions to
"which of my K known alphabets is this," which the mixed-alphabet and
frozen-body-vs-scratch results suggest it does, at least partially,
rather than learning each symbol's meaning independently.

This design removes that incentive structurally: there is no per-alphabet
output row at all. A blank cell's prediction is a compatibility score
(learned query projection on the cell's contextual hidden state, dot-
producted against each candidate's raw token_embedding -- reusing the
INPUT embedding table as the pointer target, no separate output
projection) against whatever (<=9) distinct tokens actually appear as
givens in THAT SPECIFIC puzzle instance. The output space is bounded by
the board, not by how many alphabets have ever been trained on.

This is necessary but not sufficient for zero-shot generalization to a
genuinely novel alphabet: it removes the *option* of an absolute-identity
shortcut, but nothing forces the encoder to learn a truly identity-
invariant, similarity-based computation instead of one still subtly
entangled with the known alphabets' specific trained embeddings. That
remains the open empirical question this architecture is built to test,
not a guaranteed outcome -- see the Abstractors/relational-bottleneck
literature (PLAN.md) for the natural next step if it underperforms.

`use_query_projection=False` is a cheap ablation: raw (un-projected) dot
products between the post-encoder hidden state and the input embedding
table, which both Pointer Networks and pointer-generator copy mechanisms
found insufficient (the two live in very different subspaces after
several transformer layers) -- included to let that claim be checked
empirically rather than assumed.
"""

from __future__ import annotations

import torch
from torch import nn


def _box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + (col // 3)


class SudokuPointerTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_iterations: int = 1,
        vocab_size: int = 19,
        use_query_projection: bool = True,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.use_query_projection = use_query_projection

        # Same shared encoder/positional setup as SudokuTransformer -- kept
        # as a separate class (not a refactor of it) so that model stays
        # completely untouched and its checkpoints/tests keep working
        # exactly as validated, with zero regression risk from this new,
        # more novel architecture.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.row_embedding = nn.Embedding(9, d_model)
        self.col_embedding = nn.Embedding(9, d_model)
        self.box_embedding = nn.Embedding(9, d_model)

        rows = torch.tensor([r for r in range(9) for _ in range(9)])
        cols = torch.tensor([c for _ in range(9) for c in range(9)])
        boxes = torch.tensor([_box_index(r, c) for r in range(9) for c in range(9)])
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.register_buffer("boxes", boxes, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if use_query_projection:
            self.query_proj = nn.Linear(d_model, d_model)

    def _positions(self) -> torch.Tensor:
        return (
            self.row_embedding(self.rows)
            + self.col_embedding(self.cols)
            + self.box_embedding(self.boxes)
        )  # (81, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """tokens: (B, 81) token ids (0 = blank). candidate_tokens: (B, 9)
        token ids, one per digit role present as a given in that puzzle,
        padded with an arbitrary value where fewer than 9 are present.
        candidate_mask: (B, 9) bool, True = real candidate, False = padding.
        Returns a list of `num_iterations` logits tensors, each (B, 81, 9)
        -- one score per candidate SLOT (not a global class id); map back
        to a token via candidate_tokens.gather(1, argmax_slot)."""
        blank_mask = tokens == 0
        current = tokens
        pos = self._positions().unsqueeze(0)  # (1, 81, d_model)

        candidate_keys = self.token_embedding(candidate_tokens)  # (B, 9, d_model)
        mask_value = torch.finfo(candidate_keys.dtype).min
        invalid = (~candidate_mask).unsqueeze(1)  # (B, 1, 9), broadcasts over the 81 query positions

        all_logits = []
        for _ in range(self.num_iterations):
            x = self.token_embedding(current) + pos
            x = self.encoder(x)  # (B, 81, d_model)
            query = self.query_proj(x) if self.use_query_projection else x
            logits = torch.einsum("bqd,bkd->bqk", query, candidate_keys)  # (B, 81, 9)
            logits = logits.masked_fill(invalid, mask_value)
            all_logits.append(logits)
            if self.num_iterations > 1:
                pred_slot = logits.argmax(dim=-1)  # (B, 81), index into the 9 candidates
                candidate_tokens_expanded = candidate_tokens.unsqueeze(1).expand(-1, 81, -1)
                pred_token = torch.gather(candidate_tokens_expanded, 2, pred_slot.unsqueeze(-1)).squeeze(-1)
                current = torch.where(blank_mask, pred_token, tokens)
        return all_logits
