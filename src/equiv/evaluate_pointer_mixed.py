"""Pointer-model analogue of evaluate_mixed.py: evaluate a pointer
checkpoint (train_pointer.py) on "mixed" alphabets -- each digit role
(1-9) uses the symbol from a possibly different already-trained alphabet,
in a 9-symbol combination never presented as a single coherent alphabet
during training. See sudoku.alphabets.mixed_relabel for the full
rationale (this tests compositional generalization over already-known
symbols, distinct from evaluate_pointer.py's zero-shot-to-a-genuinely-new-
alphabet test).

No new training needed: pure evaluation against an existing pointer
checkpoint. Primary metric is valid_rate, same reasoning as
evaluate_pointer.py.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.equiv.evaluate_mixed import (  # noqa: E402
    all_same_digit_to_letter,
    cycling_digit_to_letter,
    random_digit_to_letter,
)
from src.equiv.evaluate_pointer import load_checkpoint  # noqa: E402
from src.equiv.model.config import Config  # noqa: E402
from src.equiv.sudoku.pointer_dataset import MixedAlphabetPointerDataset  # noqa: E402
from src.equiv.sudoku.solver import is_valid_board  # noqa: E402
from src.equiv.utils import pick_device  # noqa: E402


@torch.no_grad()
def evaluate_mixed_pointer(
    model,
    base_path: str,
    digit_to_letter: dict[int, str],
    split: str,
    device: torch.device,
    indices: list[int] | None = None,
    batch_size: int = 128,
) -> dict:
    """Score a pointer `model` on one mixed-alphabet trial (`digit_to_letter`):
    cell accuracy, exact-match rate, and the rate of legal completed boards.
    `indices`, if given, restricts to that puzzle subset of `split`."""
    dataset = MixedAlphabetPointerDataset(base_path, split, digit_to_letter)
    if indices is not None:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_blank = total_cell_correct = total_boards = total_exact_match = total_valid = 0
    for puzzle, solution, blank_mask, candidate_tokens, candidate_mask, target_slot, _fully_represented in loader:
        puzzle, solution, blank_mask = puzzle.to(device), solution.to(device), blank_mask.to(device)
        candidate_tokens, candidate_mask, target_slot = (
            candidate_tokens.to(device),
            candidate_mask.to(device),
            target_slot.to(device),
        )
        all_logits = model(puzzle, candidate_tokens, candidate_mask)
        pred_slot = all_logits[-1].argmax(dim=-1)
        candidate_tokens_expanded = candidate_tokens.unsqueeze(1).expand(-1, 81, -1)
        pred_tokens = torch.gather(candidate_tokens_expanded, 2, pred_slot.unsqueeze(-1)).squeeze(-1)
        filled = torch.where(blank_mask, pred_tokens, puzzle)

        cell_correct = (pred_slot == target_slot) & blank_mask
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
    """CLI entry point: run the fixed control/maximally-mixed/random trial
    protocol against a pointer checkpoint and report each trial's metrics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--letters", required=True, help="comma-separated known/trained alphabet letters, e.g. A,B,C,D,F,G")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-puzzles", type=int, default=1000, help="puzzles per trial (same subset reused across all trials)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    device = pick_device(config.train.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    letters = args.letters.split(",")

    full = MixedAlphabetPointerDataset(config.data.output_path, args.split, all_same_digit_to_letter(letters[0]))
    rng_np = np.random.default_rng(args.seed)
    indices = rng_np.choice(len(full), size=min(args.n_puzzles, len(full)), replace=False).tolist()

    rng = random.Random(args.seed)
    trials = [
        (f"control_all_{letters[0]}", all_same_digit_to_letter(letters[0])),
        ("maximally_mixed_cycle", cycling_digit_to_letter(letters)),
    ]
    for i in range(args.n_trials):
        trials.append((f"random_{i}", random_digit_to_letter(letters, rng)))

    results = []
    for name, digit_to_letter in trials:
        metrics = evaluate_mixed_pointer(model, config.data.output_path, digit_to_letter, args.split, device, indices=indices)
        result = {"checkpoint": str(args.checkpoint), "trial": name, "digit_to_letter": digit_to_letter, **metrics}
        results.append(result)
        print(json.dumps(result))
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    random_accs = [r["cell_accuracy"] for r in results if r["trial"].startswith("random_")]
    random_valid = [r["valid_rate"] for r in results if r["trial"].startswith("random_")]
    if len(random_accs) > 1:
        print(
            f"\nrandom mixes (n={len(random_accs)}): "
            f"mean cell_accuracy={statistics.mean(random_accs):.4f} stdev={statistics.stdev(random_accs):.4f} "
            f"mean valid_rate={statistics.mean(random_valid):.4f} stdev={statistics.stdev(random_valid):.4f}"
        )


if __name__ == "__main__":
    main()
