"""Few-shot adaptation to a held-out alphabet, after multi-alphabet
pretraining (see train.py --alphabets).

--mode finetune: load a multi-alphabet checkpoint, freeze everything except
--target-alphabet's own embedding/output-head rows, and train on a small
(--n-puzzles) subset of that alphabet's data.

--mode scratch: ignore --checkpoint, train a completely fresh, fully
trainable model (same vocab_size_for(--vocab-k)/num_classes_for(--vocab-k)
as the pretraining run, so the comparison is apples-to-apples) on the SAME
small subset -- the baseline --mode finetune is compared against, to test
whether multi-alphabet pretraining lets the model adapt to a new alphabet
from less data than training from scratch would.

    python -m src.equiv.finetune --config configs/base.yaml \
        --checkpoint checkpoints/base_multiABCDFG.pt --target-alphabet E \
        --n-puzzles 1000 --mode finetune --seed 0

    python -m src.equiv.finetune --config configs/base.yaml \
        --target-alphabet E --n-puzzles 1000 --mode scratch --vocab-k 7 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.equiv.model.config import Config  # noqa: E402
from src.equiv.model.transformer import SudokuTransformer  # noqa: E402
from src.equiv.sudoku.alphabets import (  # noqa: E402
    class_range,
    num_classes_for,
    offset_for_letter,
    vocab_size_for,
)
from src.equiv.sudoku.dataset import SudokuDataset  # noqa: E402
from src.equiv.train import compute_loss, evaluate_split  # noqa: E402
from src.equiv.utils import pick_device  # noqa: E402


def _row_mask(weight: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    mask = torch.zeros_like(weight)
    mask[lo:hi] = 1.0
    return mask


def freeze_for_finetune(model: SudokuTransformer, target_alphabet: str) -> None:
    """In-place: freeze the shared encoder body entirely (requires_grad_(False),
    the standard whole-parameter freeze -- the optimizer skips .grad is None
    params automatically), and mask token_embedding/output_head gradients via
    register_hook so only target_alphabet's rows can ever update.

    The hook masks the gradient at the moment autograd produces it, so it
    can't be silently bypassed by something added between backward() and
    optimizer.step() later (e.g. grad clipping) -- unlike zeroing .grad
    manually after backward().

    Caller MUST use weight_decay=0.0 on the optimizer: AdamW's decoupled
    weight decay shrinks every parameter element every step regardless of
    its gradient, so masking the gradient to zero on non-target rows does
    NOT stop weight decay from corrupting them over many fine-tuning steps.
    """
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for emb in (model.row_embedding, model.col_embedding, model.box_embedding):
        for p in emb.parameters():
            p.requires_grad_(False)

    lo, hi = class_range(target_alphabet, {target_alphabet: offset_for_letter(target_alphabet)})
    tok_lo, tok_hi = lo + 1, hi + 1  # token ids are class index + 1

    emb_mask = _row_mask(model.token_embedding.weight, tok_lo, tok_hi)
    model.token_embedding.weight.register_hook(lambda g: g * emb_mask)

    w_mask = _row_mask(model.output_head.weight, lo, hi)
    model.output_head.weight.register_hook(lambda g: g * w_mask)

    b_mask = torch.zeros_like(model.output_head.bias)
    b_mask[lo:hi] = 1.0
    model.output_head.bias.register_hook(lambda g: g * b_mask)


def _n_puzzle_subset(dataset: SudokuDataset, n_puzzles: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_puzzles, len(dataset)), replace=False).tolist()
    return Subset(dataset, indices)


def _tagged_path(path: str, identifier: str) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}_{identifier}{p.suffix}")


def finetune(
    config: Config,
    mode: str,
    checkpoint: str | None,
    target_alphabet: str,
    n_puzzles: int,
    epochs: int,
    seed: int,
    vocab_k: int | None,
) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    device = pick_device(config.train.device)

    if mode == "finetune":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --mode finetune")
        ckpt = torch.load(checkpoint, map_location=device)
        model = SudokuTransformer(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        freeze_for_finetune(model, target_alphabet)
        weight_decay = 0.0  # see freeze_for_finetune docstring: required, not optional
    elif mode == "scratch":
        if vocab_k is None:
            raise ValueError("--vocab-k is required for --mode scratch (match the pretraining run's K)")
        model_kwargs = {
            **asdict(config.model),
            "vocab_size": vocab_size_for(vocab_k),
            "num_classes": num_classes_for(vocab_k),
        }
        model = SudokuTransformer(**model_kwargs).to(device)
        weight_decay = config.train.weight_decay
    else:
        raise ValueError(f"unknown mode {mode!r}")

    alphabets = {target_alphabet: offset_for_letter(target_alphabet)}
    train_full = SudokuDataset(config.data.output_path, "train", target_alphabet, alphabets)
    train_ds = _n_puzzle_subset(train_full, n_puzzles, seed)
    # Capped, not the full val split: this experiment deliberately runs many
    # cheap epochs over a TINY train set (n_puzzles), so validating against
    # the full split (10k puzzles at base.yaml scale) every epoch would
    # dominate wall-clock time by orders of magnitude. This is a monitoring
    # signal only -- the rigorous final number comes from evaluate.py's
    # separate pass over the FULL, uncapped test split after training.
    val_full = SudokuDataset(config.data.output_path, "val", target_alphabet, alphabets)
    val_ds = _n_puzzle_subset(val_full, min(500, len(val_full)), seed)
    train_loader = DataLoader(train_ds, batch_size=min(config.train.batch_size, n_puzzles), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.train.lr, weight_decay=weight_decay
    )

    identifier = f"{mode}{target_alphabet}_n{n_puzzles}_seed{seed}"
    checkpoint_path = _tagged_path(config.train.checkpoint_path, identifier)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = _tagged_path(config.train.log_path, identifier)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    with open(log_path, "w") as log_file:
        for epoch in range(epochs):
            model.train()
            for puzzle, solution, blank_mask in train_loader:
                puzzle, solution, blank_mask = puzzle.to(device), solution.to(device), blank_mask.to(device)
                all_logits = model(puzzle)
                loss = compute_loss(all_logits, solution, blank_mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_file.write(
                    json.dumps({"step": step, "epoch": epoch, "split": "train", "loss": loss.item(), "time": time.time()})
                    + "\n"
                )
                step += 1

            val_loss, val_cell_acc = evaluate_split(model, val_loader, device, target_alphabet)
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
            "model_config": {**asdict(config.model), "vocab_size": model.token_embedding.num_embeddings, "num_classes": model.output_head.out_features},
            "alphabet": target_alphabet,
            "alphabets": None,
            "mode": mode,
            "source_checkpoint": checkpoint,
            "n_puzzles": n_puzzles,
            "seed": seed,
            "epochs_trained": epochs,
            "tag": "",
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["finetune", "scratch"])
    parser.add_argument("--checkpoint", default=None, help="required for --mode finetune")
    parser.add_argument("--vocab-k", type=int, default=None, help="required for --mode scratch")
    parser.add_argument("--target-alphabet", required=True)
    parser.add_argument("--n-puzzles", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override config.train.epochs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = Config.load(args.config)
    epochs = args.epochs if args.epochs is not None else config.train.epochs
    finetune(
        config,
        args.mode,
        args.checkpoint,
        args.target_alphabet,
        args.n_puzzles,
        epochs,
        args.seed,
        args.vocab_k,
    )


if __name__ == "__main__":
    main()
