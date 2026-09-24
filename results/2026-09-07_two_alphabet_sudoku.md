# Two-Alphabet Sudoku Equivalence Study — Results (2026-09-07)

Full run on Leonardo (CINECA EuroHPC), account `EUHPC_XXXX_XXX`, single A100
GPU per job. Raw numbers: `results/eval_results.jsonl` (same run).

## Setup

- `configs/base.yaml`: 200,000 Sudoku puzzles, 40 given clues, generated
  once in the canonical alphabet (digits 1-9). Alphabet B is the identical
  set of puzzles with every non-blank digit shifted `+9` (10-18) — a pure
  vocabulary-renaming transform, `P ~ Q`, not a different puzzle
  distribution. Fixed split: 180,000 train / 10,000 val / 10,000 test,
  identical indices across both alphabets.
- Model: 8-layer, `d_model=384` encoder-only transformer, single-pass
  (`num_iterations=1`), 18-way literal-token output head (vocab: blank +
  digits 1-9 + digits 10-18, one shared embedding table).
- Training: 30 epochs, batch size 256, AdamW, lr 3e-4. Jobs: `train_A`
  (52min), `train_B` (52min), `train_union` (1h43min, ~2x data).
- Four checkpoints evaluated on the held-out test split (10,000 puzzles per
  alphabet):
  - `A`, `B`: trained on one alphabet only.
  - `union`: trained on both alphabets mixed (upper-bound reference).
  - `random-init`: same architecture, zero training (floor reference).
- Three eval modes per single-alphabet checkpoint:
  - **self** — evaluated on its own training alphabet.
  - **naive cross** — evaluated directly on the other alphabet.
  - **transplanted cross** — the known digit-renaming map is copied into
    the embedding/output-head rows for the unseen alphabet's tokens (no
    retraining) before evaluating, isolating whether the internal
    reasoning transfers independent of the untrained-embedding confound.

Training-time validation accuracy at epoch 29 (final): A = 99.43%,
B = 99.47%, union = 99.65%.

## Results

| Model | Eval alphabet | Mode | Cell acc | Exact match | Valid rate | n |
|---|---|---|---|---|---|---|
| A | A | self | 99.41% | 92.20% | 92.20% | 10,000 |
| A | B | naive cross | **11.28%** | 0.00% | 0.00% | 10,000 |
| A | B | transplanted cross | 99.41% | 92.20% | 92.20% | 10,000 |
| B | B | self | 99.46% | 92.31% | 92.31% | 10,000 |
| B | A | naive cross | **10.95%** | 0.00% | 0.00% | 10,000 |
| B | A | transplanted cross | 99.46% | 92.31% | 92.31% | 10,000 |
| union | A | (trained on both) | 99.63% | 94.64% | 94.64% | 10,000 |
| union | B | (trained on both) | 99.65% | 95.08% | 95.08% | 10,000 |
| random-init | A | self floor | 11.10% | 0.00% | 0.00% | 10,000 |
| random-init | B | naive-cross floor | 11.17% | 0.00% | 0.00% | 10,000 |

(Chance level for 9-way digit classification: 1/9 ≈ 11.11%.)

## Interpretation

1. **The model genuinely learned to solve Sudoku.** 99.4%+ cell accuracy
   and ~92% full-board exact match on held-out 40-clue puzzles it never
   saw during training — not memorization noise.

2. **Naive cross-alphabet transfer is exactly zero.** A model trained only
   on alphabet A, given a B puzzle, scores 11.28% cell accuracy —
   statistically indistinguishable from the **random-init floor**
   (11.10-11.17%) and from pure chance. Same for B→A (10.95%). The model
   has not learned *anything* transferable about the renamed digits on its
   own; it performs exactly as if it had never been trained at all.

3. **Transplanted cross-eval exactly equals self-eval** (99.41% = 99.41%,
   99.46% = 99.46% — not approximately, identically, by construction: once
   the digit-correspondence is copied into the embedding/output-head rows,
   the forward pass computes the literal same function on the relabeled
   puzzle). This confirms the *only* thing separating 11% from 99% is
   knowing the symbol correspondence — there is no deeper representational
   gap once that's supplied.

4. **The union model (trained on both alphabets) does slightly *better*
   than either single-alphabet model** (99.6%+ vs 99.4%), not worse —
   mixing alphabets causes no negative interference, consistent with 4
   simply having ~2x the effective training data rather than any conflict
   between the two vocabularies.

**Headline finding:** under plain single-alphabet supervised training, the
model's Sudoku-solving competence is bound to the specific tokens it was
trained on. It does not spontaneously discover that a renamed puzzle is
the same underlying problem (`P ~ Q`) — that equivalence has to be handed
to it externally (transplant). Whatever internal "reasoning circuit" the
model learned is not represented in a vocabulary-invariant way.

## Caveats / scope

- One small transformer (~tens of millions of params) trained from scratch
  on one narrow task and format. This is **not** a claim about
  transformers in general, or about large pretrained LLMs, which are
  exposed to vastly more varied surface forms during pretraining and might
  behave very differently on an analogous test.
- Only one transformation type was tested here: a pure digit-offset
  renaming. Other transforms from the original framing (row/column
  permutation, notation changes, premise reordering, graph isomorphisms)
  are untested and could behave differently.
- The result is consistent with "no training pressure toward invariance →
  no invariance for free," which is a narrower and more mechanistic claim
  than "transformers structurally cannot recognize this kind of analogy."
  The next experiment below is designed to distinguish these two readings.

## Suggested next step

Train a model on **many** random digit-relabelings of the puzzle (data
augmentation over the full symmetry group of relabelings, not just two
fixed alphabets), then test zero-shot generalization to a held-out
relabeling never seen during training.

- If that succeeds: supports the "right training signal is enough"
  reading — the invariance is learnable, it just wasn't incentivized by
  single-vocabulary training data. Not a fundamental architectural limit.
- If it still fails: a substantially stronger result, closer to genuine
  evidence of a structural obstacle to spontaneous analogy-recognition in
  this training paradigm.

This would be the natural `P_1` in the `P_0 ~ P_1 ~ ... ~ P_k` sequence
from the original framing.
