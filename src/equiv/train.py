"""Train one model: a single alphabet ("A"/"B") or the mixed-alphabet
reference model ("union").

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
from src.equiv.sudoku.alphabets import class_range  # noqa: E402
from src.equiv.sudoku.dataset import SudokuDataset, union_dataset  # noqa: E402
from src.equiv.utils import pick_device  # noqa: E402


def build_dataset(base_path: str, split: str, alphabet: str):
    if alphabet == "union":
        return union_dataset(base_path, split)
    return SudokuDataset(base_path, split, alphabet)


def compute_loss(
    all_logits: list[torch.Tensor], solution: torch.Tensor, blank_mask: torch.Tensor
) -> torch.Tensor:
    target_classes = solution - 1  # token id -> class index, see sudoku/alphabets.py
    loss = torch.zeros((), device=solution.device)
    for logits in all_logits:
        loss = loss + F.cross_entropy(logits[blank_mask], target_classes[blank_mask])
    return loss / len(all_logits)


@torch.no_grad()
def evaluate_split(
    model: SudokuTransformer, loader: DataLoader, device: torch.device, alphabet: str
) -> tuple[float, float]:
    # For "union" the loader mixes both alphabets, so there's no single class
    # range to restrict to -- this val accuracy is a rough monitoring signal
    # only; evaluate.py's per-alphabet restricted accuracy is the rigorous one.
    restrict = alphabet in ("A", "B")
    if restrict:
        lo, hi = class_range(alphabet)

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


def _tagged_path(path: str, alphabet: str, tag: str) -> Path:
    p = Path(path)
    suffix = f"_{alphabet}" + (f"_{tag}" if tag else "")
    return p.with_name(f"{p.stem}{suffix}{p.suffix}")


def train(config: Config, alphabet: str, epochs: int, tag: str) -> Path:
    torch.manual_seed(config.train.seed)
    random.seed(config.train.seed)
    device = pick_device(config.train.device)

    model = SudokuTransformer(**asdict(config.model)).to(device)
    checkpoint_path = _tagged_path(config.train.checkpoint_path, alphabet, tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if epochs > 0:
        train_ds = build_dataset(config.data.output_path, "train", alphabet)
        val_ds = build_dataset(config.data.output_path, "val", alphabet)
        train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

        log_path = _tagged_path(config.train.log_path, alphabet, tag)
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

                val_loss, val_cell_acc = evaluate_split(model, val_loader, device, alphabet)
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
            "model_config": asdict(config.model),
            "alphabet": alphabet,
            "epochs_trained": epochs,
            "tag": tag,
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--alphabet", required=True, choices=["A", "B", "union"])
    parser.add_argument(
        "--epochs", type=int, default=None, help="override config.train.epochs (0 = random-init baseline, no training)"
    )
    parser.add_argument("--tag", default="", help="extra suffix for checkpoint/log filenames")
    args = parser.parse_args()

    config = Config.load(args.config)
    epochs = args.epochs if args.epochs is not None else config.train.epochs
    train(config, args.alphabet, epochs, args.tag)


if __name__ == "__main__":
    main()
