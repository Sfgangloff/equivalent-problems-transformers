"""Evaluate a pointer-model checkpoint (train_pointer.py). Primary metric:
`valid_rate` -- does the model produce a fully constraint-satisfying grid
(no repeated symbol in any row/col/box, respecting the givens), regardless
of whether it matches one arbitrarily-chosen ground-truth labeling. This
is the right metric for a genuinely novel alphabet: there's no
"class_range"/offset bookkeeping needed at all, since the pointer head
outputs a literal token id directly (via candidate_tokens), not an
absolute class index -- unlike evaluate.py's classifier-model evaluation.

Also reports `cell_accuracy`/`exact_match_rate` against the synthetic
ground-truth correspondence as a secondary diagnostic ONLY -- flagged
explicitly because it's only computable here because we control the
puzzle generation process; it is not a fair general zero-shot metric (a
genuinely novel real-world alphabet has no knowable ground-truth
correspondence at all, which is exactly why this architecture doesn't
need one).

Evaluates on the natural, UNFILTERED test distribution by default (not
just fully-represented puzzles) -- see pointer_dataset.py's docstring:
restricting to filtered puzzles would cherry-pick and undermine the
generalization claim, even though it means ~0.8% of puzzles are
mechanically unsolvable by this architecture (no candidate exists for
their answer) and should show up as failures, honestly.
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
from src.equiv.model.pointer_transformer import SudokuPointerTransformer  # noqa: E402
from src.equiv.sudoku.alphabets import ALPHABET_LETTERS, offset_for_letter  # noqa: E402
from src.equiv.sudoku.pointer_dataset import PointerSudokuDataset  # noqa: E402
from src.equiv.sudoku.solver import is_valid_board  # noqa: E402
from src.equiv.utils import pick_device  # noqa: E402


def load_checkpoint(path: str, device: torch.device) -> tuple[SudokuPointerTransformer, dict]:
    """Load a pointer-head checkpoint into eval mode; return (model, raw checkpoint dict)."""
    ckpt = torch.load(path, map_location=device)
    model = SudokuPointerTransformer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate(
    model: SudokuPointerTransformer,
    base_path: str,
    eval_alphabet: str,
    split: str,
    device: torch.device,
    filter_fully_represented: bool = False,
    batch_size: int = 128,
) -> dict:
    """Score `model` on `eval_alphabet`'s `split`: overall metrics, plus the
    same metrics stratified by whether each puzzle's givens cover all 9
    digits (see module docstring for why that split matters)."""
    alphabets = {eval_alphabet: offset_for_letter(eval_alphabet)}
    dataset = PointerSudokuDataset(base_path, split, eval_alphabet, alphabets, filter_fully_represented)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Stratified by whether the puzzle's own givens include all 9 digits:
    # "missing" puzzles are mechanically unsolvable by this architecture
    # (no candidate exists for the missing digit's cells, see module
    # docstring) -- separating them out shows how much of the overall
    # failure rate, even on a known alphabet, is this edge case vs.
    # something else.
    buckets = {
        "all": {"blank": 0, "cell_correct": 0, "boards": 0, "exact_match": 0, "valid": 0},
        "fully_represented": {"blank": 0, "cell_correct": 0, "boards": 0, "exact_match": 0, "valid": 0},
        "missing_digit": {"blank": 0, "cell_correct": 0, "boards": 0, "exact_match": 0, "valid": 0},
    }

    for puzzle, solution, blank_mask, candidate_tokens, candidate_mask, target_slot, fully_represented in loader:
        puzzle, solution, blank_mask = puzzle.to(device), solution.to(device), blank_mask.to(device)
        candidate_tokens, candidate_mask, target_slot = (
            candidate_tokens.to(device),
            candidate_mask.to(device),
            target_slot.to(device),
        )
        fully_represented = fully_represented.to(device).bool()

        all_logits = model(puzzle, candidate_tokens, candidate_mask)
        pred_slot = all_logits[-1].argmax(dim=-1)  # (B, 81), index into candidates
        candidate_tokens_expanded = candidate_tokens.unsqueeze(1).expand(-1, 81, -1)
        pred_tokens = torch.gather(candidate_tokens_expanded, 2, pred_slot.unsqueeze(-1)).squeeze(-1)
        filled = torch.where(blank_mask, pred_tokens, puzzle)

        # secondary diagnostic only -- see module docstring. target_slot is
        # -1 for cells whose true digit isn't among the givens (only
        # possible when filter_fully_represented=False); those can never
        # be "correct" by construction, so pred_slot (always >= 0) never
        # equals -1 and they're correctly counted as wrong here.
        cell_correct = (pred_slot == target_slot) & blank_mask
        exact_match = (filled == solution).all(dim=1)
        valid = torch.tensor([is_valid_board(b) for b in filled.tolist()], device=device)

        for name, board_mask in (
            ("all", torch.ones_like(fully_represented)),
            ("fully_represented", fully_represented),
            ("missing_digit", ~fully_represented),
        ):
            cell_mask = board_mask.unsqueeze(1) & blank_mask
            buckets[name]["blank"] += cell_mask.sum().item()
            buckets[name]["cell_correct"] += (cell_correct & cell_mask).sum().item()
            buckets[name]["boards"] += board_mask.sum().item()
            buckets[name]["exact_match"] += (exact_match & board_mask).sum().item()
            buckets[name]["valid"] += (valid & board_mask).sum().item()

    def _metrics(b: dict) -> dict:
        """Turn one bucket's running totals into cell/exact-match/valid rates."""
        return {
            "cell_accuracy": b["cell_correct"] / b["blank"] if b["blank"] else 0.0,
            "exact_match_rate": b["exact_match"] / b["boards"] if b["boards"] else 0.0,
            "valid_rate": b["valid"] / b["boards"] if b["boards"] else 0.0,
            "n_boards": b["boards"],
        }

    return {
        **_metrics(buckets["all"]),
        "fully_represented": _metrics(buckets["fully_represented"]),
        "missing_digit": _metrics(buckets["missing_digit"]),
    }


def main() -> None:
    """CLI entry point: parse args, load the checkpoint, and evaluate on one alphabet/split."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alphabet", required=True, choices=list(ALPHABET_LETTERS))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--filter-fully-represented",
        action="store_true",
        help="restrict to puzzles where every digit appears among givens (default: natural, unfiltered distribution)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = Config.load(args.config)
    device = pick_device(config.train.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    trained_alphabets = ckpt.get("alphabets")
    mode = "known" if trained_alphabets and args.alphabet in trained_alphabets else "zero_shot"

    metrics = evaluate(
        model, config.data.output_path, args.alphabet, args.split, device,
        filter_fully_represented=args.filter_fully_represented,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "trained_alphabets": trained_alphabets,
        "eval_alphabet": args.alphabet,
        "mode": mode,
        "split": args.split,
        "filtered": args.filter_fully_represented,
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
