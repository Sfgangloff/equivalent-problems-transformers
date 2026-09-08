"""YAML-driven configuration for data generation, the model, and training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    n_puzzles: int = 2000
    target_clues: int = 40
    seed: int = 0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    output_path: str = "data/sudoku.npz"


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    num_iterations: int = 1  # >1 unrolls the block with shared weights (RRN/IREM-style)


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 3.0e-4
    weight_decay: float = 0.01  # AdamW's own default; finetune.py's freeze path overrides to 0.0
    epochs: int = 20
    device: str = "auto"  # "auto" picks cuda > mps > cpu
    seed: int = 0
    log_path: str = "logs/run.jsonl"
    checkpoint_path: str = "checkpoints/model.pt"
    log_every: int = 50


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            data=_from_dict(DataConfig, raw.get("data", {})),
            model=_from_dict(ModelConfig, raw.get("model", {})),
            train=_from_dict(TrainConfig, raw.get("train", {})),
        )


def _from_dict(cls: type, raw: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)
