"""Tests for K-way multi-alphabet training (train.py)."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_DATA = REPO_ROOT / "data" / "sudoku_tiny.npz"


@pytest.fixture(scope="module", autouse=True)
def ensure_tiny_dataset():
    """Generate the tiny fixture dataset once per test module, if not already present."""
    if not TINY_DATA.exists():
        subprocess.run(
            [sys.executable, "scripts/generate_sudoku.py", "--config", "configs/tiny.yaml"],
            cwd=REPO_ROOT,
            check=True,
        )


def test_non_contiguous_alphabets_size_vocab_to_cover_the_gap(tmp_path):
    """Regression test: --alphabets A,B,C,D,F,G (skipping E) must size the
    model to cover positions 0-6 (through G), not just len(letters)=6 --
    otherwise E's reserved (untrained) slot would be missing and G's tokens
    (offset 54-62) would overflow a too-small vocab/output head."""
    from src.equiv.model.config import Config
    from src.equiv.train import train

    config = Config.load(REPO_ROOT / "configs" / "tiny.yaml")
    config.train.checkpoint_path = str(tmp_path / "ckpt.pt")
    config.train.log_path = str(tmp_path / "log.jsonl")

    ckpt_path = train(config, alphabet=None, epochs=1, tag="", alphabets=["A", "B", "C", "D", "F", "G"], n_puzzles=20)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    assert ckpt["model_config"]["vocab_size"] == 1 + 7 * 9  # covers A..G (7 positions), not 6
    assert ckpt["model_config"]["num_classes"] == 7 * 9
    assert ckpt["model_state"]["token_embedding.weight"].shape[0] == 1 + 7 * 9
    assert ckpt["model_state"]["output_head.weight"].shape[0] == 7 * 9
