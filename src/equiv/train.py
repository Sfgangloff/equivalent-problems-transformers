"""Train one model: a single alphabet ("A"/"B"), the mixed-2-alphabet
reference model ("union"), or a K-way multi-alphabet model (`--alphabets`,
e.g. `--alphabets A,B,C,D,F,G` -- letters need not be contiguous from "A",
so a letter like "E" can be reserved as a held-out alphabet for a later
zero-shot/few-shot experiment without ever appearing in this run's data).

To produce the random-init baseline referenced in the eval plan, run with
`--epochs 0` and a `--tag` so it doesn't overwrite the real checkpoint for
that alphabet: the model is built and immediately saved untrained, and
evaluate.py treats it exactly like any other checkpoint (its "self-eval" and
"naive cross-eval" numbers become the floor references for that alphabet).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.equiv.model.config import Config  # noqa: E402
from src.equiv.model.transformer import SudokuTransformer  # noqa: E402
from src.equiv.sudoku.alphabets import (  # noqa: E402
    ALPHABET_LETTERS,
    class_range,
    num_classes_for,
    offset_for_letter,
    vocab_size_for,
)
from src.equiv.sudoku.dataset import (  # noqa: E402
    SudokuDataset,
    multi_alphabet_dataset,
    union_dataset,
)
from src.equiv.utils import pick_device  # noqa: E402


def build_dataset(base_path: str, split: str, alphabet: str):
    """Return the requested split as a single alphabet's dataset, or the A+B union."""
    if alphabet == "union":
        return union_dataset(base_path, split)
    return SudokuDataset(base_path, split, alphabet)


def compute_loss(
    all_logits: list[torch.Tensor], solution: torch.Tensor, blank_mask: torch.Tensor
) -> torch.Tensor:
    """Average cross-entropy over blank cells, across every unrolled iteration's logits."""
    target_classes = solution - 1  # token id -> class index, see sudoku/alphabets.py
    loss = torch.zeros((), device=solution.device)
    for logits in all_logits:
        loss = loss + F.cross_entropy(logits[blank_mask], target_classes[blank_mask])
    return loss / len(all_logits)


@torch.no_grad()
def evaluate_split(
    model: SudokuTransformer, loader: DataLoader, device: torch.device, alphabet: str | None
) -> tuple[float, float]:
    """Run one no-grad pass over `loader`, returning (mean loss, cell accuracy).

    `alphabet` controls whether the argmax is restricted to that alphabet's
    own class range (see the comment below); pass `None` for a union or
    multi-alphabet loader, whose batches mix alphabets.
    """
    # Restricting to a single class_range only makes sense for a loader that
    # is entirely one alphabet. "union" and multi-alphabet (alphabet=None)
    # loaders mix alphabets per batch, so this val accuracy is a rough
    # monitoring signal only; evaluate.py's per-alphabet restricted accuracy
    # (always single-alphabet) is the rigorous one. Any single letter (not
    # just "A"/"B") is restrictable -- offset_for_letter handles any letter.
    restrict = alphabet is not None and alphabet != "union"
    if restrict:
        lo, hi = class_range(alphabet, {alphabet: offset_for_letter(alphabet)})

    model.eval()
    total_loss, total_correct, total_blank, n_batches = 0.0, 0, 0, 0
    for puzzle, solution, blank_mask in loader:
        puzzle, solution, blank_mask = (
            puzzle.to(device),
            solution.to(device),
            blank_mask.to(device),
        )
        all_logits = model(puzzle)
        loss = compute_loss(all_logits, solution, blank_mask)
        total_loss += loss.item()
        n_batches += 1

        if restrict:
            pred_tokens = all_logits[-1][..., lo:hi].argmax(dim=-1) + lo + 1
        else:
            pred_tokens = all_logits[-1].argmax(dim=-1) + 1
        correct = (pred_tokens == solution) & blank_mask
        total_correct += correct.sum().item()
        total_blank += blank_mask.sum().item()
    model.train()
    cell_acc = total_correct / total_blank if total_blank else 0.0
    return total_loss / max(n_batches, 1), cell_acc


def _tagged_path(path: str, identifier: str, tag: str) -> Path:
    """Insert `_{identifier}` (and, if given, `_{tag}`) before a path's extension."""
    p = Path(path)
    suffix = f"_{identifier}" + (f"_{tag}" if tag else "")
    return p.with_name(f"{p.stem}{suffix}{p.suffix}")


def train(
    config: Config,
    alphabet: str | None,
    epochs: int,
    tag: str,
    alphabets: list[str] | None = None,
    n_puzzles: int | None = None,
) -> Path:
    """Train (or, with `epochs=0`, just build and save untrained) one model.

    Either `alphabet` (single letter or "union") or `alphabets` (a
    multi-alphabet K-way list) must be used; see module docstring for the
    `epochs=0` random-init-baseline mode. Returns the saved checkpoint path.
    """
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    device = pick_device(config.train.device)

    if alphabets is not None:
        # NOT len(alphabets): letters need not be contiguous from "A" (e.g.
        # A,B,C,D,F,G deliberately skips E to reserve it held-out). Vocab/
        # output-head size must cover up to the highest-position letter used
        # so that skipped letters' rows still exist (untrained) in the model
        # for later zero-shot/few-shot eval -- len() alone would undersize
        # the model and either truncate or misalign every letter after a gap.
        k = max(ALPHABET_LETTERS.index(letter) for letter in alphabets) + 1
        model_kwargs = {
            **asdict(config.model),
            "vocab_size": vocab_size_for(k),
            "num_classes": num_classes_for(k),
        }
        identifier = "multi" + "".join(alphabets)
    else:
        model_kwargs = asdict(config.model)
        identifier = alphabet

    model = SudokuTransformer(**model_kwargs).to(device)
    checkpoint_path = _tagged_path(config.train.checkpoint_path, identifier, tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if epochs > 0:
        if alphabets is not None:
            train_ds = multi_alphabet_dataset(
                config.data.output_path, "train", alphabets, n_puzzles=n_puzzles, seed=config.train.seed
            )
            val_ds = multi_alphabet_dataset(config.data.output_path, "val", alphabets)
        else:
            train_ds = build_dataset(config.data.output_path, "train", alphabet)
            val_ds = build_dataset(config.data.output_path, "val", alphabet)
        train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

        # For a single-alphabet loader, evaluate_split can restrict argmax to
        # that alphabet's own class range; a union/multi-alphabet loader mixes
        # alphabets per batch, so pass None (no restriction) -- see its docstring.
        eval_alphabet = alphabet if alphabets is None else None

        log_path = _tagged_path(config.train.log_path, identifier, tag)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        step = 0
        with open(log_path, "w") as log_file:
            for epoch in range(epochs):
                model.train()
                for puzzle, solution, blank_mask in train_loader:
                    puzzle, solution, blank_mask = (
                        puzzle.to(device),
                        solution.to(device),
                        blank_mask.to(device),
                    )
                    all_logits = model(puzzle)
                    loss = compute_loss(all_logits, solution, blank_mask)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % config.train.log_every == 0:
                        log_file.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "epoch": epoch,
                                    "split": "train",
                                    "loss": loss.item(),
                                    "time": time.time(),
                                }
                            )
                            + "\n"
                        )
                        log_file.flush()
                    step += 1

                val_loss, val_cell_acc = evaluate_split(model, val_loader, device, eval_alphabet)
                log_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "split": "val",
                            "loss": val_loss,
                            "cell_accuracy": val_cell_acc,
                            "time": time.time(),
                        }
                    )
                    + "\n"
                )
                log_file.flush()
                print(f"epoch {epoch}: val_loss={val_loss:.4f} val_cell_acc={val_cell_acc:.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            # model_kwargs (not asdict(config.model)) so a checkpoint from a
            # K-way multi-alphabet run records its actual vocab_size/num_classes
            # override -- otherwise reloading via SudokuTransformer(**model_config)
            # (see evaluate.py/finetune.py) would reconstruct the wrong-sized model.
            "model_config": model_kwargs,
            "alphabet": alphabet,
            "alphabets": alphabets,
            "n_puzzles": n_puzzles,
            "epochs_trained": epochs,
            "tag": tag,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    """CLI entry point: parse args, load the config, and run one training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--alphabet", choices=["A", "B", "union"], help="single/2-alphabet training")
    group.add_argument(
        "--alphabets",
        help="comma-separated letters for K-way multi-alphabet training, e.g. A,B,C,D,F,G",
    )
    parser.add_argument(
        "--n-puzzles",
        type=int,
        default=None,
        help="only with --alphabets: puzzles per alphabet (default: full train split)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="override config.train.epochs (0 = random-init baseline, no training)"
    )
    parser.add_argument("--tag", default="", help="extra suffix for checkpoint/log filenames")
    args = parser.parse_args()

    config = Config.load(args.config)
    epochs = args.epochs if args.epochs is not None else config.train.epochs
    alphabets = args.alphabets.split(",") if args.alphabets else None
    train(config, args.alphabet, epochs, args.tag, alphabets=alphabets, n_puzzles=args.n_puzzles)


if __name__ == "__main__":
    main()
