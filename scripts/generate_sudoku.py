#!/usr/bin/env python
"""Generate the base Sudoku dataset (canonical 1-9 alphabet) and a fixed
train/val/test split. Alphabet A, alphabet B, and the union view are all
derived from this single file at load time (see src/equiv/sudoku/dataset.py)
-- this script never writes alphabet-specific data.

Usage:
    python scripts/generate_sudoku.py --config configs/tiny.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.equiv.model.config import Config  # noqa: E402
from src.equiv.sudoku.generate import generate_puzzle  # noqa: E402

_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def make_split_assignment(
    n: int, val_fraction: float, test_fraction: float, rng: random.Random
) -> np.ndarray:
    """Randomly assign each of `n` puzzle indices a split code (0=train, 1=val, 2=test)."""
    order = list(range(n))
    rng.shuffle(order)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    split = np.zeros(n, dtype=np.int8)  # default: train
    for i in order[:n_val]:
        split[i] = _SPLIT_CODE["val"]
    for i in order[n_val : n_val + n_test]:
        split[i] = _SPLIT_CODE["test"]
    return split


def main() -> None:
    """CLI entry point: generate the configured number of puzzles and write
    the base dataset (puzzles, solutions, split) to a single `.npz` file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = Config.load(args.config)
    data_cfg = config.data
    rng = random.Random(data_cfg.seed)

    puzzles = np.zeros((data_cfg.n_puzzles, 81), dtype=np.int8)
    solutions = np.zeros((data_cfg.n_puzzles, 81), dtype=np.int8)
    for i in range(data_cfg.n_puzzles):
        puzzle, solution = generate_puzzle(rng, target_clues=data_cfg.target_clues)
        puzzles[i] = puzzle
        solutions[i] = solution

    split = make_split_assignment(
        data_cfg.n_puzzles, data_cfg.val_fraction, data_cfg.test_fraction, rng
    )

    output_path = Path(data_cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, puzzles=puzzles, solutions=solutions, split=split)

    n_train = int((split == 0).sum())
    n_val = int((split == 1).sum())
    n_test = int((split == 2).sum())
    print(
        f"wrote {data_cfg.n_puzzles} puzzles to {output_path} "
        f"(train={n_train}, val={n_val}, test={n_test})"
    )


if __name__ == "__main__":
    main()
