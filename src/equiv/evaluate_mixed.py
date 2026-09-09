"""Evaluate a multi-alphabet-trained checkpoint on "mixed" alphabets: each
digit role (1-9) uses the symbol from a possibly different already-trained
alphabet, in a 9-symbol combination never presented as a single coherent
alphabet during training. Every individual symbol is known; the specific
combination is not.

Tests whether the model binds each symbol's meaning independently of which
other symbols happen to co-occur (real compositional generalization) as
opposed to recognizing/routing through one of the K known coherent
alphabets as a whole (a block-level shortcut a model trained only on K
disjoint, internally-uniform alphabets could get away with -- see
sudoku.alphabets.mixed_relabel for the full rationale).

No new training needed: pure evaluation against an existing multi-alphabet
checkpoint (see train.py --alphabets).
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

from src.equiv.evaluate import load_checkpoint  # noqa: E402
from src.equiv.model.config import Config  # noqa: E402
from src.equiv.sudoku.alphabets import mixed_class_indices  # noqa: E402
from src.equiv.sudoku.dataset import MixedAlphabetDataset  # noqa: E402
from src.equiv.sudoku.solver import is_valid_board  # noqa: E402
from src.equiv.utils import pick_device  # noqa: E402


def all_same_digit_to_letter(letter: str) -> dict[int, str]:
    """Control: every digit from the SAME known alphabet -- reconstructs
    that alphabet exactly, so this should recover ordinary self-eval
    accuracy. Sanity-checks the mixed-eval machinery itself."""
    return {d: letter for d in range(1, 10)}


def cycling_digit_to_letter(letters: list[str]) -> dict[int, str]:
    """Maximally mixed: every digit from a different known alphabet,
    cycling through `letters` if there are fewer than 9."""
    return {d: letters[(d - 1) % len(letters)] for d in range(1, 10)}


def random_digit_to_letter(letters: list[str], rng: random.Random) -> dict[int, str]:
    return {d: rng.choice(letters) for d in range(1, 10)}


@torch.no_grad()
def evaluate_mixed(
    model,
    base_path: str,
    digit_to_letter: dict[int, str],
    split: str,
    device: torch.device,
    indices: list[int] | None = None,
    batch_size: int = 128,
) -> dict:
    dataset = MixedAlphabetDataset(base_path, split, digit_to_letter)
    if indices is not None:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # class_indices[i] is digit (i+1)'s class index under this mixed
    # alphabet; generally non-contiguous, unlike class_range()'s (lo, hi).
    class_indices = torch.tensor(mixed_class_indices(digit_to_letter), dtype=torch.long, device=device)

    total_blank = total_cell_correct = total_boards = total_exact_match = total_valid = 0
    for puzzle, solution, blank_mask in loader:
        puzzle, solution, blank_mask = puzzle.to(device), solution.to(device), blank_mask.to(device)
        all_logits = model(puzzle)
        restricted = all_logits[-1][..., class_indices]  # (B, 81, 9)
        pred_slot = restricted.argmax(dim=-1)  # 0..8, index into class_indices
        pred_tokens = class_indices[pred_slot] + 1
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--letters", required=True, help="comma-separated known/trained alphabet letters, e.g. A,B,C,D,F,G")
    parser.add_argument("--n-trials", type=int, default=10, help="number of random mixes, in addition to the 2 fixed controls")
    parser.add_argument("--n-puzzles", type=int, default=1000, help="puzzles per trial (same subset reused across all trials)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    device = pick_device(config.train.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    letters = args.letters.split(",")

    # Same underlying puzzle subset for every trial (only the digit_to_letter
    # mapping varies) -- isolates the mixed-alphabet effect from puzzle
    # difficulty variance across trials.
    full = MixedAlphabetDataset(config.data.output_path, args.split, all_same_digit_to_letter(letters[0]))
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
        metrics = evaluate_mixed(model, config.data.output_path, digit_to_letter, args.split, device, indices=indices)
        result = {"checkpoint": str(args.checkpoint), "trial": name, "digit_to_letter": digit_to_letter, **metrics}
        results.append(result)
        print(json.dumps(result))
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    random_accs = [r["cell_accuracy"] for r in results if r["trial"].startswith("random_")]
    if len(random_accs) > 1:
        print(
            f"\nrandom mixes (n={len(random_accs)}): "
            f"mean cell_accuracy={statistics.mean(random_accs):.4f} "
            f"stdev={statistics.stdev(random_accs):.4f} "
            f"min={min(random_accs):.4f} max={max(random_accs):.4f}"
        )


if __name__ == "__main__":
    main()
