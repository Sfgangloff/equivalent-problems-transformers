# Two-Alphabet Sudoku Equivalence Study — Plan

## Context

The user is exploring a research question from a draft paper excerpt: does a transformer's *learned solving procedure* for a combinatorial problem transfer across presentation-preserving transformations of that problem (renamed variables/relations, changed vocabulary, permuted premises, notation changes, graph isomorphisms, ...)? Formally, if `P ~ Q` (same underlying structure, different surface form) and a model solves `P`, does it also solve `Q`?

The first concrete milestone (this plan's scope) is Sudoku with a pure vocabulary-renaming transform: the *same* underlying puzzles, once labeled with digits 1–9 (alphabet A) and once with the same digits shifted to 10–18 (alphabet B). Train one model per alphabet, then test whether solving ability transfers to the alphabet it never saw. Training will run on the user's Leonardo EuroHPC GPU allocation; I have no SSH/2FA access to Leonardo from this session, so my output is code + SLURM scripts that the user runs themselves, reporting logs/results back for iteration. The broader general "equivalence transform" framework (relation renaming, premise permutation, graph isomorphism, arbitrary `P_0 ~ ... ~ P_k` chains) and other problem domains are explicitly deferred past this milestone — the code should not preclude them, but shouldn't be built out now.

A background design pass (web research + literature check) surfaced one thing worth calling out up front because it shapes the eval design: with a single shared embedding table where each model only ever sees its own alphabet's tokens, the untrained alphabet's embedding rows stay at random init. A naive cross-alphabet eval then conflates two different failure modes — "never learned this token" vs. "the reasoning doesn't transfer" — which is exactly the ambiguity the experiment is trying to resolve. The fix (§5 below) is cheap and is treated as core to the MVP, not a stretch goal.

## Repo setup

- `git init` locally in this directory; `gh repo create Sfgangloff/equivalent-problems-transformers --private --source=. --remote=origin` (gh auth already confirmed working as `Sfgangloff`).
- Standard Python `.gitignore` (venvs, `__pycache__`, checkpoints, generated datasets, `logs/`, `results/` — those are regenerated, not committed; a `results/README.md` placeholder explains that run outputs live on Leonardo and get pulled back selectively).
- `README.md` stating the research question, the P~Q framing, and how to reproduce (generate data → local smoke test → Leonardo SLURM run → eval).

## Data pipeline

- `src/equiv/sudoku/solver.py`: bitmask-based backtracking solver (9-bit row/col/box candidate sets) that counts solutions up to 2 (for uniqueness checks) and can return the unique solution.
- `src/equiv/sudoku/generate.py`: generate a full valid grid via randomized backtracking with MRV cell ordering, then carve a puzzle by removing cells one at a time (checking uniqueness after each removal via the solver) down to a target clue count (~35–45 givens — moderate difficulty, chosen so a small transformer has a realistic shot at learning it; this is a controlled testbed, not a sudoku-solving benchmark).
- `src/equiv/sudoku/alphabets.py`: the relabeling abstraction — constants (`BLANK=0`, alphabet A = identity on 1–9, alphabet B = `+9`, shared vocab size 19) and a pure relabel function. Designed so future variants (digit permutation, letters, etc.) are a one-function addition later.
- `src/equiv/sudoku/dataset.py`: **one** base dataset is generated and stored (digits 1–9, canonical form) with one fixed-seed train/val/test split. `SudokuDataset(base_path, split, alphabet)` applies the alphabet relabeling *lazily* in `__getitem__`, and a union view is just a `ConcatDataset` of the A-view and B-view over the same indices. This avoids ever materializing two datasets that could drift out of sync — A, B, and union are always guaranteed to be the same underlying puzzles.
- `scripts/generate_sudoku.py`: CLI entry point that writes the base dataset + split indices. Pure CPU work; fine to run locally for a smoke-scale set, and as a small CPU job on Leonardo for the full-scale set.

## Model

- `src/equiv/model/transformer.py`: encoder-only transformer over the 81 fixed cell positions (no attention masking needed — fixed length). Token embedding (vocab size 19) + summed learned row/col/3x3-box positional embeddings. Output: an 18-way head predicting the literal token id (not a 9-way "digit within my own alphabet" head — a shared/relative head would implicitly hand the model its own alphabet identity, which would presuppose the answer to what we're testing). Loss (cross-entropy) computed only on cells that were originally blank.
- A `num_iterations` config flag (default 1, single-pass) with an escape hatch to unroll the same block with shared weights for `k` steps, feeding back the previous step's predicted-digit embedding for still-blank cells (RRN/IREM-style), with loss summed across steps. Single-pass is the default per the "simplicity over SOTA" framing and is plausible to work at this difficulty level (a deep single-pass transformer's own layer stack already approximates a few rounds of row/col/box constraint propagation); the literature's known fix if it plateaus near a trivial baseline is iterative refinement, not autoregressive fill order (which would entangle solving-order with presentation — an extra confound this study wants to avoid).
- `src/equiv/model/config.py`: YAML-driven hyperparams. `configs/tiny.yaml` (2 layers, d_model 64 — local Mac smoke test) and `configs/base.yaml` (6–12 layers, d_model 256–512, ~10–30M params — the real Leonardo run).

## Training / eval

- `src/equiv/train.py`: config-driven training loop. Default logging is local JSONL metrics + `torch.save` checkpoints (Leonardo GPU compute nodes commonly restrict outbound internet, so no cloud dependency by default); an optional `--logger wandb` flag exists but that import only happens if explicitly requested.
- `src/equiv/evaluate.py`: computes per-cell accuracy on blanks, full-board exact-match, and valid-Sudoku rate (no duplicate in any row/col/box). Supports a `--transplant` mode: given a checkpoint trained on alphabet A, before evaluating on B's test set, copy the A-token embedding rows and output-head rows/bias into the corresponding B-token slots (`embedding.weight[10:19] = embedding.weight[1:10]`, same for the output head) — this hands the model the known ground-truth renaming map at the input/output layer only, isolating "does the internal reasoning transfer" from "did it ever learn these token embeddings."
- Models trained: alphabet A only, alphabet B only, a union model (mixed A+B, upper-bound reference), and a random-init baseline (untrained, lower-bound reference for the cross-eval numbers).
- Final metrics table per model: self-eval, naive cross-eval, transplanted cross-eval — this three-way split is what makes the headline result interpretable.

## Repo layout

```
src/equiv/
  sudoku/
    generate.py       solver.py       alphabets.py       dataset.py
  model/
    transformer.py     config.py
  train.py              evaluate.py   (incl. --transplant)
scripts/
  generate_sudoku.py
configs/
  tiny.yaml   base.yaml
slurm/
  leonardo_env.sh
  generate_data.slurm   train_alphabet_a.slurm   train_alphabet_b.slurm
  train_union.slurm     eval_cross.slurm
tests/
  test_solver.py   test_alphabets.py   test_dataset.py   test_model_shapes.py
```

## SLURM templates (Leonardo)

Researched current conventions (CINECA docs, Aug 2026): Booster GPU partition `boost_usr_prod`, debug QOS `boost_qos_dbg` (≤30min, ≤8 nodes — good for smoke tests), long QOS `boost_qos_lprod` (≤4 days). Typical GPU job shape: `--gres=gpu:4 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=8` (8 CPU cores/GPU is the documented ratio). Account is set via `#SBATCH -A <project_account>` — this is a project-specific string I can't know or guess, left as a clearly marked placeholder.

What I could **not** verify: exact current module version strings (e.g. the `cineca-ai/<version>` or `cuda/<version>` to load) — these drift and I won't hardcode a guessed version. `leonardo_env.sh` will show both supported setup paths (the `module load profile/deeplrn` → `cineca-ai` prebuilt-PyTorch path, and the manual `python`/`cuda` module + venv path) with a `# TODO` asking the user to run `module av cineca-ai` and `module av python` once on Leonardo and paste the output back so I can pin exact versions before any real job runs. Puzzle generation (`generate_data.slurm`) is CPU-only — also flagged as a TODO to confirm whether it should target Booster without `--gres=gpu` or Leonardo's CPU (DCGP) partition, rather than guessing.

## Implementation & verification order

1. `sudoku/solver.py` + `generate.py` + `tests/test_solver.py` — generate ~100 puzzles locally, verify uniqueness and a few by hand.
2. `sudoku/alphabets.py` + `dataset.py` + `tests/test_alphabets.py` — verify the relabel is an exact bijection and A/B/union share identical indices by construction.
3. `model/transformer.py` (`num_iterations=1`) + `tests/test_model_shapes.py` — forward-pass shape/dtype checks only, no training.
4. `configs/tiny.yaml` end-to-end run on the Mac (CPU/MPS, ~500 puzzles, a few hundred steps): confirm `train.py` runs, loss decreases, checkpoints and JSONL logs are written.
5. `evaluate.py` on the tiny run, including `--transplant` — confirms all three eval paths compute without shape errors (numbers won't be meaningful yet, this is a plumbing check).
6. Only after all local checks pass: finalize SLURM templates. User runs `module av` on Leonardo and reports back so I can pin exact module versions; then one `boost_qos_dbg` smoke-scale run on Leonardo before committing to a full `base.yaml` run.
7. Full runs on Leonardo: train A, train B, train union, random-init baseline → `eval_cross.slurm` producing the complete self/naive-cross/transplanted-cross results table. User reports logs/results back here for interpretation.
