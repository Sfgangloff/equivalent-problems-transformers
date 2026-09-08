"""Encoder-only Sudoku transformer.

Input: 81 tokens (0 = blank, 1-9 = alphabet A digits, 10-18 = alphabet B
digits). Output: per-cell 18-way logits over the literal token id
(class index = token_id - 1). The head predicts the literal token rather
than a "digit within my own alphabet" -- a relative/shared head would
implicitly hand the model its own alphabet identity, presupposing the
answer to the question this study asks.

`num_iterations` > 1 unrolls the same encoder with shared weights, feeding
back each step's argmax prediction for still-blank cells (RRN/IREM-style).
forward() always returns one logits tensor per iteration so the training
loop can supervise every step.

`vocab_size`/`num_classes` default to the 2-alphabet (A, B) values but can
be overridden (e.g. via sudoku.alphabets.vocab_size_for(k)/num_classes_for(k))
for multi-alphabet (K > 2) experiments.
"""

from __future__ import annotations

import torch
from torch import nn

from ..sudoku.alphabets import NUM_CLASSES, VOCAB_SIZE


def _box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + (col // 3)


class SudokuTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_iterations: int = 1,
        vocab_size: int = VOCAB_SIZE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.num_iterations = num_iterations

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
        self.output_head = nn.Linear(d_model, num_classes)

    def _positions(self) -> torch.Tensor:
        return (
            self.row_embedding(self.rows)
            + self.col_embedding(self.cols)
            + self.box_embedding(self.boxes)
        )  # (81, d_model)

    def forward(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """tokens: (B, 81) token ids in [0, VOCAB_SIZE).
        Returns a list of `num_iterations` logits tensors, each (B, 81, NUM_CLASSES)."""
        blank_mask = tokens == 0
        current = tokens
        pos = self._positions().unsqueeze(0)  # (1, 81, d_model), broadcasts over batch

        all_logits = []
        for _ in range(self.num_iterations):
            x = self.token_embedding(current) + pos
            x = self.encoder(x)
            logits = self.output_head(x)  # (B, 81, NUM_CLASSES)
            all_logits.append(logits)
            if self.num_iterations > 1:
                pred_token = logits.argmax(dim=-1) + 1  # class index -> token id
                current = torch.where(blank_mask, pred_token, tokens)
        return all_logits
