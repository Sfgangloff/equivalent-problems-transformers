"""Evaluate a trained checkpoint: per-cell accuracy on blanks, full-board
exact-match rate, and valid-Sudoku rate (no duplicate in any row/col/box).

`--transplant` copies the checkpoint's own alphabet's embedding and
output-head rows into the target `--alphabet`'s token slots (using the
known ground-truth digit-renaming map) before evaluating. This isolates
whether the model's internal reasoning transfers across the renaming,
independent of the untrained-embedding confound a naive cross-eval has.
See README.md for the full rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.equiv.model.config import Config  # noqa: E402
from src.equiv.model.transformer import SudokuTransformer  # noqa: E402
from src.equiv.sudoku.alphabets import ALPHABET_LETTERS, class_range, offset_for_letter  # noqa: E402
from src.equiv.sudoku.dataset import SudokuDataset  # noqa: E402
from src.equiv.sudoku.solver import is_valid_board  # noqa: E402
from src.equiv.utils import pick_device  # noqa: E402


def load_checkpoint(path: str, device: torch.device) -> tuple[SudokuTransformer, dict]:
    ckpt = torch.load(path, map_location=device)
    model = SudokuTransformer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def transplant(model: SudokuTransformer, source_alphabet: str, target_alphabet: str) -> None:
    """In-place: copy source_alphabet's per-digit embedding/output-head rows
    into target_alphabet's token slots via the known renaming map. Works for
    any two letters (each letter's offset is fixed by its position in
    ALPHABET_LETTERS, see offset_for_letter), not just the original A/B."""
    src_offset = offset_for_letter(source_alphabet)
    dst_offset = offset_for_letter(target_alphabet)
    with torch.no_grad():
        for digit in range(1, 10):
            src_token = digit + src_offset
            dst_token = digit + dst_offset
            src_class = src_token - 1
            dst_class = dst_token - 1
            model.token_embedding.weight[dst_token] = model.token_embedding.weight[src_token].clone()
            model.output_head.weight[dst_class] = model.output_head.weight[src_class].clone()
            model.output_head.bias[dst_class] = model.output_head.bias[src_class].clone()


@torch.no_grad()
def evaluate(
    model: SudokuTransformer,
    base_path: str,
    eval_alphabet: str,
    split: str,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    alphabets = {eval_alphabet: offset_for_letter(eval_alphabet)}
    dataset = SudokuDataset(base_path, split, eval_alphabet, alphabets=alphabets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    lo, hi = class_range(eval_alphabet, alphabets)

    total_blank = 0
    total_cell_correct = 0
    total_boards = 0
    total_exact_match = 0
    total_valid = 0

    for puzzle, solution, blank_mask in loader:
        puzzle, solution, blank_mask = (
            puzzle.to(device),
            solution.to(device),
            blank_mask.to(device),
        )
        all_logits = model(puzzle)
        # restrict to eval_alphabet's own 9 classes -- see class_range() docstring
        pred_tokens = all_logits[-1][..., lo:hi].argmax(dim=-1) + lo + 1
        filled = torch.where(blank_mask, pred_tokens, puzzle)

        cell_correct = (pred_tokens == solution) & blank_mask
        total_cell_correct += cell_correct.sum().item()
        total_blank += blank_mask.sum().item()

        exact_match = (filled == solution).all(dim=1)
        total_exact_match += exact_match.sum().item()
        total_boards += filled.shape[0]

        for board in filled.tolist():
            if is_valid_board(board):
                total_valid += 1

    return {
        "cell_accuracy": total_cell_correct / total_blank if total_blank else 0.0,
        "exact_match_rate": total_exact_match / total_boards if total_boards else 0.0,
        "valid_rate": total_valid / total_boards if total_boards else 0.0,
        "n_boards": total_boards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="config yaml (used for data.output_path)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--alphabet",
        required=True,
        choices=list(ALPHABET_LETTERS),
        help="alphabet to evaluate on (must exist within the checkpoint's vocab_size)",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--transplant",
        action="store_true",
        help="transplant the checkpoint's own alphabet into --alphabet before evaluating",
    )
    parser.add_argument("--out", default=None, help="append the result as one JSON line to this file")
    args = parser.parse_args()

    config = Config.load(args.config)
    device = pick_device(config.train.device)

    model, ckpt = load_checkpoint(args.checkpoint, device)
    source_alphabet = ckpt.get("alphabet")
    trained_alphabets = ckpt.get("alphabets")  # non-None for a multi-alphabet (K-way) checkpoint

    if trained_alphabets is not None:
        # multi-alphabet checkpoint: no single "source" alphabet, so mode is
        # "known" (eval_alphabet was one of the K training alphabets) or
        # "zero_shot" (it wasn't -- e.g. a held-out letter like "E").
        if args.transplant:
            raise ValueError(
                "--transplant needs a single well-defined source alphabet; "
                f"checkpoint {args.checkpoint} is multi-alphabet ({trained_alphabets})"
            )
        mode = "known" if args.alphabet in trained_alphabets else "zero_shot"
    elif args.transplant:
        transplant(model, source_alphabet, args.alphabet)
        mode = "transplanted_cross"
    elif args.alphabet == source_alphabet:
        mode = "self"
    else:
        mode = "naive_cross"

    metrics = evaluate(model, config.data.output_path, args.alphabet, args.split, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "trained_alphabet": source_alphabet,
        "trained_alphabets": trained_alphabets,
        "eval_alphabet": args.alphabet,
        "mode": mode,
        "split": args.split,
        **metrics,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
