# Equivalent Problems for Transformers: Two-Alphabet Sudoku

Research question: does a transformer's *learned solving procedure* for a
combinatorial problem transfer across presentation-preserving transformations
of that problem? Formally, if `P ~ Q` (same underlying structure, different
surface form) and a model solves `P`, does it also solve `Q`?

This repo's first milestone tests a pure vocabulary-renaming transform on
Sudoku: the *same* underlying puzzles, once labeled with digits 1-9
(alphabet A) and once with the same digits shifted to 10-18 (alphabet B).
One model is trained per alphabet; each is then evaluated on the alphabet it
never saw, three ways:

- **self-eval** — model tested on its own training alphabet (upper bound).
- **naive cross-eval** — model tested directly on the other alphabet. This
  conflates "never learned this token" with "reasoning doesn't transfer."
- **transplanted cross-eval** — before testing on the other alphabet, the
  known digit-renaming map is copied into the embedding/output-head rows for
  the unseen tokens (no retraining). This isolates whether the internal
  reasoning transfers, independent of the untrained embedding rows.

See `slurm/` for how training runs on Leonardo (CINECA EuroHPC).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate puzzles (base dataset, digits 1-9; alphabet B/union are lazy views)
python scripts/generate_sudoku.py --config configs/tiny.yaml

# 2. local smoke test (tiny model, CPU/MPS)
python -m src.equiv.train --config configs/tiny.yaml --alphabet A
python -m src.equiv.train --config configs/tiny.yaml --alphabet B

# 3. evaluate, including the transplant test
python -m src.equiv.evaluate --config configs/tiny.yaml \
    --checkpoint <path-to-A-checkpoint> --alphabet B --transplant
```

Full-scale runs use `configs/base.yaml` and the SLURM scripts in `slurm/`,
which are meant to run on Leonardo, not locally.

## Tests

```bash
pytest tests/
```
