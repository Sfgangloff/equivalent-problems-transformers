# Multi-Alphabet Sudoku Experiments — Results (2026-09-09)

Two follow-up experiments to the original two-alphabet study
(`results/2026-09-07_two_alphabet_sudoku.md`), both built on a single
model pretrained on 6 disjoint Sudoku alphabets (A, B, C, D, F, G — E
deliberately reserved held-out). Raw numbers: `results/eval_mixed_results.jsonl`,
`results/eval_finetune_results.jsonl`.

## Setup

- `checkpoints/base_multiABCDFG.pt`: same architecture/data scale as the
  original study (8-layer, `d_model=384`, 200,000 base puzzles), trained
  on 60,000 puzzles per alphabet (same underlying puzzles, 6 different
  relabelings) for 20 epochs. Reached 99.57% cell accuracy across all 6
  trained alphabets during training — real, well-learned competence, not
  a weak baseline.
- Alphabet E (offset 36-44) was never included in this pretraining, but
  the model's vocabulary/output head were sized to include its slots
  (untrained) from the start, so it could be tested later without
  re-architecting anything.

## Experiment A: mixed alphabets (compositional generalization)

**Question** (from a colleague's suggestion): the model was trained on 6
alphabets, each *internally uniform* (every digit in a given puzzle comes
from the same alphabet block). Could it be taking a shortcut — "recognize
which of my 6 known alphabets this is, then apply that alphabet's
routine" — rather than learning what each individual symbol means,
independent of which other symbols happen to appear alongside it? To test
this, construct "mixed" alphabets: for each of the 9 digit roles,
independently pick which of the 6 known alphabets supplies that role's
symbol. Every individual symbol is one the model has seen and learned;
the specific 9-symbol *combination* has never been presented as a single
alphabet during training.

**Results** (1000 test puzzles per trial, same underlying puzzle subset
across all trials):

| Trial | Cell accuracy | Exact match |
|---|---|---|
| control (all digits from alphabet A) | 99.49% | 93.0% |
| maximally mixed (every digit a different known alphabet) | 44.5% | 0% |
| 10 random mixes | 23.2% – 70.8% (mean 45.4%, stdev 12.4%) | 0% |

The control confirms the evaluation mechanism itself is correct (matches
ordinary single-alphabet self-eval exactly). The random/mixed trials are
the finding: **far above chance** (~11% would be chance for 9-way
classification) but **far below the 99% ceiling**, and accuracy clearly
tracks how "coherent" a given mix is — e.g. `random_0` (digit-to-alphabet
map dominated by one alphabet, D, with a few substitutions) scored 70.8%,
while `random_4` (spread evenly across 6 different alphabets) scored
23.2%.

## Experiment B: few-shot adaptation to the held-out alphabet

**Question**: does pretraining on 6 known alphabets make the model adapt
*faster* to alphabet E, given a small amount of labeled E data, than
training a model from scratch on that same small amount of data would?
This is distinct from zero-shot transfer (no data at all) — it asks
whether prior exposure to related-but-different vocabularies buys
sample-efficiency on a new one.

Two conditions, at n ∈ {100, 1000} E-puzzles, 3 random seeds each:
- **finetune**: start from the pretrained 6-alphabet checkpoint, freeze
  the entire shared encoder body, and only let alphabet E's own
  embedding/output-head rows adapt on the small E dataset.
- **scratch**: a fresh, fully-trainable model (nothing pretrained, nothing
  frozen) trained on the identical small E dataset — the baseline.

**Results** (full, uncapped 10,000-puzzle E test split):

| Condition | n=100 | n=1000 |
|---|---|---|
| zero-shot (no fine-tuning at all) | 11.8% | — |
| finetune (frozen body) | 26.4% / 26.8% / 27.1% (mean 26.8%) | 39.5% / 39.4% / 39.5% (mean 39.5%) |
| scratch (fresh, unfrozen) | 25.7% / 26.8% / 26.9% (mean 26.5%) | 68.4% / 70.2% / 68.9% (mean 69.2%) |

At n=100 the two conditions are statistically indistinguishable. At
n=1000, **scratch clearly outperforms finetune** (69.2% vs 39.5%) — and
finetune's number barely moves across the three seeds, indicating a real
ceiling from the frozen body, not noise.

## Interpretation: the two experiments reinforce each other

Both results point to the same underlying conclusion from different
angles: **the shared network body learned from 6-alphabet pretraining is
not a clean, alphabet-agnostic Sudoku-solving procedure** — it still
carries structure specific to the particular alphabets it was trained on.

- If it were a clean abstraction, mixed alphabets (built entirely from
  known, well-learned symbols) should score close to 99%, not 23-71%.
- If it were a clean abstraction, freezing it and only relearning E's
  vocabulary mapping (`finetune`) should be *easier* than relearning
  everything (`scratch`) from the same data — reusing a working reasoning
  engine and only needing a small vocabulary lookup should beat learning
  vocabulary *and* reasoning jointly from scratch. Instead `finetune`
  plateaus well below `scratch` once there's enough data (n=1000) for
  `scratch`'s extra flexibility to matter.

Put plainly: whatever the model learned during 6-alphabet pretraining is
useful (far above chance in both tests) but partial and entangled with
the specific 6 vocabularies seen, not a portable, disentangled "solve
Sudoku regardless of symbol identity" circuit.

## Caveats

- Same scope caveats as the original study apply: one small transformer
  trained from scratch on one narrow task; not a claim about transformers
  or large pretrained LLMs in general.
- The mixed-alphabet accuracy-vs-coherence pattern (dominated-by-one-alphabet
  mixes scoring higher) is based on only 10 random trials — suggestive,
  not a rigorously fit relationship. A follow-up could deliberately vary
  "how many distinct alphabets are mixed in" as a controlled independent
  variable rather than relying on chance draws.
- `finetune`'s frozen-body design (per `finetune.py`) only ever allows
  alphabet E's own embedding/output-head rows to move; it does not test
  intermediate options (e.g. unfreezing just the last encoder layer),
  which could plausibly close some of the gap to `scratch` without giving
  up all the pretraining benefit.

## Suggested next steps

1. **Vary mix coherence deliberately**: construct mixed alphabets with a
   controlled number of distinct source alphabets (1, 2, 3, ... 6) rather
   than fully random draws, to directly test whether accuracy degrades
   smoothly with "how many alphabets are blended."
2. **Partial unfreezing**: repeat the few-shot sweep unfreezing progressively
   more of the encoder (last layer only, last two layers, ...) to find
   where the finetune/scratch gap closes, which would localize where the
   alphabet-specific entanglement actually lives in the network.
3. This still motivates the originally-planned Experiment 2 (pointer/copy
   architecture, see `PLAN.md`): a model that never needs a fixed,
   per-alphabet output vocabulary in the first place would sidestep the
   entanglement this results doc surfaces, rather than trying to work
   around it after the fact.
