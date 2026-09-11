"""Train the pointer/copy Sudoku model (model/pointer_transformer.py),
mirroring train.py's --alphabets multi-alphabet path but kept as a
separate script since the pointer model's data/loss/eval shapes differ
enough from the classifier model's to make sharing code more confusing
than duplicating the (short) training loop -- same isolation reasoning as
finetune.py: zero regression risk to the already-validated classifier path.

Two mutually exclusive training modes:

  --alphabets A,B,C,D,F,G --n-puzzles 60000 --epochs 20
      K FIXED alphabets, each puzzle repeated once per letter (see
      multi_alphabet_pointer_dataset). Letters need not be contiguous from
      "A" (e.g. skip "E" to reserve it held-out) -- vocab sizing follows
      the same non-contiguous-letters fix as train.py (size by highest
      letter position used, not len(alphabets)), so a skipped letter's
      embedding row still exists (untrained) in the model.

  --random-mix-letters A,B,C,D,F,G,H,...,Z --epochs 20
      Data augmentation: every training example gets an INDEPENDENTLY
      RANDOM digit-to-letter mapping drawn fresh from this pool, every
      access (see RandomMixPointerDataset) -- no fixed alphabet
      combination is ever repeated for the model to memorize. Tests
      whether the K=6-fixed-alphabets result's partial transfer gap is a
      training-diversity problem (should close here) or more structural.
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
from src.equiv.model.pointer_transformer import SudokuPointerTransformer  # noqa: E402
from src.equiv.sudoku.alphabets import ALPHABET_LETTERS, vocab_size_for  # noqa: E402
from src.equiv.sudoku.pointer_dataset import (  # noqa: E402
    RandomMixPointerDataset,
    multi_alphabet_pointer_dataset,
)
from src.equiv.utils import pick_device  # noqa: E402


def compute_loss(all_logits: list[torch.Tensor], target_slot: torch.Tensor, blank_mask: torch.Tensor) -> torch.Tensor:
    loss = torch.zeros((), device=target_slot.device)
    for logits in all_logits:
        loss = loss + F.cross_entropy(logits[blank_mask], target_slot[blank_mask])
    return loss / len(all_logits)


@torch.no_grad()
def evaluate_split(model: SudokuPointerTransformer, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_blank, n_batches = 0.0, 0, 0, 0
    for puzzle, _solution, blank_mask, candidate_tokens, candidate_mask, target_slot, _fully_represented in loader:
        puzzle, blank_mask = puzzle.to(device), blank_mask.to(device)
        candidate_tokens, candidate_mask, target_slot = (
            candidate_tokens.to(device),
            candidate_mask.to(device),
            target_slot.to(device),
        )
        all_logits = model(puzzle, candidate_tokens, candidate_mask)
        loss = compute_loss(all_logits, target_slot, blank_mask)
        total_loss += loss.item()
        n_batches += 1

        pred_slot = all_logits[-1].argmax(dim=-1)
        correct = (pred_slot == target_slot) & blank_mask
        total_correct += correct.sum().item()
        total_blank += blank_mask.sum().item()
    model.train()
    cell_acc = total_correct / total_blank if total_blank else 0.0
    return total_loss / max(n_batches, 1), cell_acc


def _tagged_path(path: str, identifier: str, tag: str) -> Path:
    p = Path(path)
    suffix = f"_{identifier}" + (f"_{tag}" if tag else "")
    return p.with_name(f"{p.stem}{suffix}{p.suffix}")


def train(
    config: Config,
    epochs: int,
    tag: str,
    use_query_projection: bool,
    alphabets: list[str] | None = None,
    n_puzzles: int | None = None,
    random_mix_letters: list[str] | None = None,
) -> Path:
    """Exactly one of `alphabets` (K fixed alphabets) or `random_mix_letters`
    (fresh random mix per training example, see RandomMixPointerDataset)
    must be given."""
    if (alphabets is None) == (random_mix_letters is None):
        raise ValueError("pass exactly one of alphabets or random_mix_letters")

    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    device = pick_device(config.train.device)

    letters_for_sizing = alphabets if alphabets is not None else random_mix_letters
    # See train.py's identical fix: NOT len(letters) -- letters need not be
    # contiguous from "A" (e.g. skip "E" to reserve it held-out), so vocab
    # must cover up to the highest letter position used or a skipped
    # letter's embedding row wouldn't exist at all.
    k = max(ALPHABET_LETTERS.index(letter) for letter in letters_for_sizing) + 1
    model_kwargs = {
        **asdict(config.model),
        "vocab_size": vocab_size_for(k),
        "use_query_projection": use_query_projection,
    }
    model = SudokuPointerTransformer(**model_kwargs).to(device)

    if alphabets is not None:
        identifier = "pointerMulti" + "".join(alphabets)
        train_ds = multi_alphabet_pointer_dataset(config.data.output_path, "train", alphabets, n_puzzles=n_puzzles, seed=config.train.seed)
        val_ds = multi_alphabet_pointer_dataset(config.data.output_path, "val", alphabets)
    else:
        identifier = "pointerRandomMix"
        train_ds = RandomMixPointerDataset(config.data.output_path, "train", random_mix_letters, seed=config.train.seed)
        val_ds = RandomMixPointerDataset(config.data.output_path, "val", random_mix_letters, seed=config.train.seed + 1)

    checkpoint_path = _tagged_path(config.train.checkpoint_path, identifier, tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    log_path = _tagged_path(config.train.log_path, identifier, tag)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    with open(log_path, "w") as log_file:
        for epoch in range(epochs):
            model.train()
            for puzzle, _solution, blank_mask, candidate_tokens, candidate_mask, target_slot, _fully_represented in train_loader:
                puzzle, blank_mask = puzzle.to(device), blank_mask.to(device)
                candidate_tokens, candidate_mask, target_slot = (
                    candidate_tokens.to(device),
                    candidate_mask.to(device),
                    target_slot.to(device),
                )
                all_logits = model(puzzle, candidate_tokens, candidate_mask)
                loss = compute_loss(all_logits, target_slot, blank_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % config.train.log_every == 0:
                    log_file.write(
                        json.dumps({"step": step, "epoch": epoch, "split": "train", "loss": loss.item(), "time": time.time()}) + "\n"
                    )
                    log_file.flush()
                step += 1

            val_loss, val_cell_acc = evaluate_split(model, val_loader, device)
            log_file.write(
                json.dumps(
                    {"step": step, "epoch": epoch, "split": "val", "loss": val_loss, "cell_accuracy": val_cell_acc, "time": time.time()}
                )
                + "\n"
            )
            log_file.flush()
            print(f"epoch {epoch}: val_loss={val_loss:.4f} val_cell_acc={val_cell_acc:.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_kwargs,
            # "alphabets" is read by evaluate_pointer.py to label eval
            # results "known" vs "zero_shot" -- for random-mix training
            # that means "one of the letters ever drawn into a mix", which
            # is the right notion of "known" here too (evaluate_pointer.py
            # doesn't need to distinguish "coherent alphabet" from "mix
            # ingredient").
            "alphabets": alphabets if alphabets is not None else random_mix_letters,
            "training_mode": "fixed_alphabets" if alphabets is not None else "random_mix",
            "n_puzzles": n_puzzles,
            "epochs_trained": epochs,
            "tag": tag,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--alphabets", help="comma-separated letters, e.g. A,B,C,D,F,G -- K FIXED alphabets")
    group.add_argument(
        "--random-mix-letters",
        help="comma-separated letters, e.g. A,B,C,D,F,G,H,...,Z -- fresh random mix per training example",
    )
    parser.add_argument("--n-puzzles", type=int, default=None, help="only with --alphabets: puzzles per alphabet (default: full train split)")
    parser.add_argument("--epochs", type=int, default=None, help="override config.train.epochs")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-query-projection", action="store_true", help="ablation: raw dot product, no learned query projection")
    args = parser.parse_args()

    config = Config.load(args.config)
    epochs = args.epochs if args.epochs is not None else config.train.epochs
    alphabets = args.alphabets.split(",") if args.alphabets else None
    random_mix_letters = args.random_mix_letters.split(",") if args.random_mix_letters else None
    train(
        config,
        epochs,
        args.tag,
        use_query_projection=not args.no_query_projection,
        alphabets=alphabets,
        n_puzzles=args.n_puzzles,
        random_mix_letters=random_mix_letters,
    )


if __name__ == "__main__":
    main()
