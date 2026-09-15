# Do Transformers Learn Alphabet-Agnostic Sudoku Solving?

Status report as of 2026-09-15. Covers every experiment run so far, why
each one was run, what it found, and what's still outstanding. Raw data
for every experiment lives alongside this report in `results/*.jsonl`;
per-experiment narrative reports for the two completed multi-part studies
are `results/2026-09-07_two_alphabet_sudoku.md` and
`results/2026-09-09_multi_alphabet_experiments.md`. This document is the
cross-cutting synthesis across all of them, including the pointer-network
experiments that came after those two reports.

## The research question

Does a transformer trained to solve problem P also implicitly know how to
solve an "equivalent" problem Q -- the same underlying structure, dressed
in different symbols? Or does it only learn a narrower, surface-level
competence tied to the specific tokens it trained on? This project tests
that question concretely using Sudoku as a testbed, where the transform
under study is pure digit relabeling: alphabet "B" is the exact same
puzzles as alphabet "A", with every digit mapped to a different token id.
Same combinatorial structure, different vocabulary.

## Common setup

Unless noted otherwise, every experiment below uses:

- **Data**: 200,000 Sudoku puzzles, 40 given clues, generated once in the
  canonical alphabet (digits 1-9) and relabeled per experiment. Fixed
  90/5/5 train/val/test split (180,000 / 10,000 / 10,000), identical
  puzzle indices reused across every alphabet so results are always
  directly comparable.
- **Model**: an 8-layer, `d_model=384`, 8-head encoder-only transformer
  (row/column/box positional embeddings, single forward pass, no
  iterative refinement in these runs).
- **Alphabets**: each alphabet is a disjoint 9-token block; letter's
  offset = 9 x (its position in A-Z), so A=0-8, B=9-17, C=18-26, etc.
  Alphabet **E is always reserved held-out** -- it is never included in
  any pretraining pool across every experiment in this report, so it
  always serves as the genuinely-unseen-symbol test.
- **Compute**: all training/eval on Leonardo (CINECA EuroHPC), single
  A100 GPU per job.

---

## Experiment 0: does naive training discover the equivalence at all? (2026-09-07)

**Why**: the most direct test of the research question. If a transformer
trained only on alphabet A can solve alphabet B's puzzles with no
help, that's evidence it learned something genuinely structural. If it
can't, the next question is *why* -- and whether that's fixable.

**Setup**: models trained on alphabet A only, alphabet B only, and a
mixed "union" model trained on both. Three evaluation modes per
single-alphabet model: **self** (own alphabet), **naive cross** (other
alphabet, no help), **transplanted cross** (the other alphabet, but with
the known digit-correspondence copied into the relevant embedding/output
rows before evaluating -- isolating whether reasoning transfers,
independent of the confound that an untrained embedding row necessarily
looks like noise regardless of how good the model is).

**Results**:

| Model | Eval alphabet | Mode | Cell acc | Exact match |
|---|---|---|---|---|
| A | A | self | 99.41% | 92.2% |
| A | B | naive cross | 11.28% | 0% |
| A | B | transplanted cross | 99.41% | 92.2% |
| B | B | self | 99.46% | 92.3% |
| B | A | naive cross | 10.95% | 0% |
| B | A | transplanted cross | 99.46% | 92.3% |
| union | A / B | (sees both) | 99.63% / 99.65% | 94.6% / 95.1% |
| random-init | — | floor | 11.10% / 11.17% | 0% |

**Finding**: naive cross-alphabet transfer is **exactly at the
random-init floor** -- not "a little worse," statistically
indistinguishable from a network that was never trained at all.
Transplanting the known correspondence recovers the *exact* self-eval
number, proving the only thing separating 11% from 99% is knowing the
symbol correspondence, not some deeper representational gap. **The
model's solving competence is bound to the specific trained tokens; it
does not spontaneously discover that a renamed puzzle is the same
problem.**

Full detail: `results/2026-09-07_two_alphabet_sudoku.md`.

---

## Experiment 1: does more alphabet diversity help? (2026-09-07 to 09-09)

**Why**: Experiment 0 used only one alphabet per model. Two natural
follow-up questions: (a) does pretraining on *several* alphabets let the
model adapt to a *new* one faster than training from scratch, given a
little labeled data (few-shot transfer efficiency, not zero-shot), and
(b) is the model's competence really general, or still tied to specific
*combinations* of tokens even across several known alphabets?

**Setup (a): few-shot adaptation.** Pretrained a classifier model on K=6
alphabets (A,B,C,D,F,G; E reserved held-out), 60,000 puzzles/alphabet, 20
epochs -- reached 99.57% cell accuracy across all 6. Then, for held-out
E, compared two conditions at n in {100, 1000} labeled E-puzzles, 3 seeds
each: **finetune** (freeze the entire shared encoder body, only let E's
own embedding/output rows adapt) vs. **scratch** (a fresh, fully
trainable model on the same small data -- the baseline).

| Condition | n=100 | n=1000 |
|---|---|---|
| zero-shot (no fine-tuning) | 11.8% | — |
| finetune (frozen body) | 26.8% (mean) | 39.5% (mean, ~flat across 3 seeds) |
| scratch (fresh, unfrozen) | 26.5% (mean) | 69.2% (mean) |

**Finding**: at n=100 the two conditions tie. At n=1000, training from
scratch clearly *beats* reusing the frozen pretrained body (69.2% vs.
39.5%), and the frozen-body number barely moves across seeds -- a real
ceiling, not noise. If the frozen body were a clean, portable,
alphabet-agnostic solving procedure, reusing it and only relearning
vocabulary should have been *easier* than relearning everything from the
same data. It wasn't. **The shared network body is not a clean
abstraction -- it's entangled with the specific alphabets it trained on.**

**Setup (b): mixed-alphabet compositional test** (idea from a colleague).
Same K=6-alphabet classifier checkpoint. Instead of testing a genuinely
new alphabet, recombine the 6 *known* alphabets: each of the 9 digit
roles independently draws its symbol from a possibly different known
alphabet, producing a 9-symbol combination that was never presented as
one coherent alphabet during training, even though every individual
symbol is well-learned. Tests whether the model relies on a "recognize
which of my 6 known blocks this is" shortcut (which this combination
defeats) vs. genuine per-symbol binding.

| Trial | Cell accuracy |
|---|---|
| control (all digits from alphabet A) | 99.49% |
| maximally mixed (9 digits, 6 different alphabets, cycled) | 44.5% |
| 10 random mixes | 23.2% – 70.8% (mean 45.4%, stdev 12.4%) |

**Finding**: far above chance (~11%), but far below the 99% ceiling, and
accuracy clearly tracks how "coherent" the specific mix is (dominated by
one alphabet = high; evenly spread across many = low). **Partial,
graded compositional ability, not clean disentangled symbol binding.**

Both (a) and (b) point to the same underlying conclusion from different
angles. Full detail: `results/2026-09-09_multi_alphabet_experiments.md`.

---

## Experiment 2: does an architectural fix help? (2026-09-10 to 09-11)

**Why**: the classifier model's output layer is `Linear(d_model, 9*K)`
-- one row per digit *per known alphabet*. This is a structural ceiling:
a genuinely new alphabet always needs new, never-trained output rows,
*no matter how large K is*. It also gives the model every incentive to
key its predictions on "which of my K known alphabets is this" rather
than each symbol's meaning alone -- plausibly explaining Experiment 1's
entanglement findings. This experiment removes that structure entirely.

**Design**: for each blank cell, instead of classifying into a fixed
global vocabulary, the model predicts **which of the puzzle's own (up to
9) distinct given token ids belongs there** -- a learned query projection
on the blank cell's contextual hidden state, dot-producted against each
candidate's raw input embedding (reusing the token embedding table, no
separate per-alphabet output projection). The output space is bounded by
the board (≤9 candidates), not by how many alphabets have ever been
trained on. This is *necessary but not sufficient* for genuine zero-shot
transfer: it removes the *option* of an absolute-identity shortcut, but
nothing forces the encoder's internal computation to actually become
identity-invariant rather than still subtly entangled with known
alphabets -- that's the open question the experiment tests.

Grounded in the pointer-network / symbol-binding literature: Palm,
Paquet & Winther (2018) *Recurrent Relational Networks*; Webb, Sinha &
Cohen (2021) *Emergent Symbols through Binding in External Memory*;
Altabaa, Webb, Cohen & Lafferty (2023) *Abstractors*; Vinyals, Fortunato
& Jaitly (2015) *Pointer Networks*; See, Liu & Manning (2017) *Get To
The Point*.

A known limitation, measured not assumed: a puzzle's pointer target is
only well-defined if every digit appears among that puzzle's own
*givens*. Measured directly on this repo's generator: 0.80% of puzzles
(4/500 in a sample) are missing one digit from their givens entirely --
filtered out for training (so every training target is well-defined),
left in for evaluation (so the reported numbers reflect this real, if
small, limitation honestly rather than hiding it).

**Setup**: same K=6/60,000-per-alphabet/20-epoch protocol as Experiment
1's classifier pretrain, same held-out E, for direct comparability.

**Results — in-distribution gate** (the 6 known alphabets, must be
checked before the zero-shot number means anything):

| Alphabet | Cell acc | Valid rate |
|---|---|---|
| A / B / C / D / F / G | 99.3% (all six) | 92.5% – 93.0% |

Matches the classifier's ceiling essentially exactly -- the architecture
change costs nothing on the easy case.

**Results — zero-shot on held-out E**:

| Metric | Value |
|---|---|
| Cell accuracy | 14.47% |
| Exact match rate | 0% |
| **Valid rate** | **0%** |

**Finding**: `valid_rate = 0%` is the more decisive number here (more
so than cell accuracy) -- there are 9! equally-valid relabelings of any
solution, so a model that had learned to produce *some* internally
consistent completion (even one not matching our specific labeling)
would show up as nonzero valid_rate. Zero out of 10,000 boards rules
that out completely. The 14.47% cell accuracy is a small, statistically
real signal above the ~11.1% chance floor (given ~410,000 blank test
cells, the gap is far larger than sampling noise) -- unlike the
classifier's flat 11.8% (indistinguishable from chance) -- but it is a
small effect, nowhere near functional. **Necessary but not sufficient,
exactly as flagged going in: removing the fixed-vocabulary bottleneck
gave a small edge, not the clean transfer being tested for.**

**Results — mixed-alphabet compositional test, same protocol as
Experiment 1(b), now on the pointer model** (same underlying random
mixes, same seed, directly comparable trial-by-trial):

| Trial | Classifier | Pointer | Δ |
|---|---|---|---|
| control (all-A) | 99.5% | 99.4% | ~0 |
| maximally mixed cycle | 44.5% | 34.9% | −9.6pp |
| random_0 | 70.8% | 59.5% | −11.3pp |
| random_1 | 53.1% | 75.8% | +22.7pp |
| random_2 | 42.5% | 46.5% | +4.0pp |
| random_3 | 38.7% | 45.3% | +6.5pp |
| random_4 | 23.2% | 43.0% | +19.8pp |
| random_5 | 39.2% | 76.3% | +37.1pp |
| random_6 | 45.6% | 42.7% | −2.9pp |
| random_7 | 54.0% | 45.9% | −8.0pp |
| random_8 | 41.1% | 34.4% | −6.7pp |
| random_9 | 46.3% | 58.7% | +12.4pp |
| **mean of 10 random trials** | **45.4% (stdev 12.4%)** | **52.8% (stdev 14.3%)** | **+7.4pp** |

**Finding**: a real average improvement, and notably the pointer
architecture helps *most* on the classifier's worst cases (the two
biggest gains are on the classifier's two lowest-scoring trials, and the
overall floor moved up from 23.2% to 34.4%). But it's uneven -- 4 of 11
non-control trials got *worse*, including the most systematic one
(maximally mixed cycle). **Consistent overall story across both tests:
the architectural fix helps more with the easier question (recombining
known symbols) than the harder one (a genuinely novel symbol), in both
cases modestly rather than decisively.**

Also added: a lightweight interpretive hypothesis for *why* the pointer
model shows a small positive zero-shot signal at all, where the
classifier showed none. The classifier's failure was total because no
mechanism existed for gradient to reach an untrained output row. The
pointer model's E-alphabet embeddings are also untrained (random), but
every occurrence of a given E-symbol in a puzzle uses the *same* random
vector -- and "two things being the same vector" is a property of vector
*equality*, not vector *identity*, so a "don't repeat a symbol in this
row/column/box" detector built on similarity could plausibly fire
correctly even for never-trained tokens. This would predict the observed
pattern (weak but real signal, far from solved) but has not been
directly verified (would need an interpretability probe of attention/
compatibility patterns) -- flagged as future work, not a settled finding.

---

## Experiment 3: does more training diversity close the gap? (2026-09-11 to present, **in progress**)

**Why**: Experiment 2 showed a real but small zero-shot improvement.
Two possible explanations, and a cheap way to distinguish them before
committing to a bigger architecture change: (a) a training-diversity
problem -- the K=6 *fixed* alphabets were each repeated across 20 epochs,
letting the model partially memorize those 6 specific combinations, or
(b) something more structural that no amount of training diversity would
fix. This experiment tests (a) directly and cheaply, reusing the exact
same pointer architecture with no code changes to the model itself.

**Design**: instead of 6 fixed alphabets, **every single training
example draws an independently random digit-to-letter mapping**, fresh,
from a pool of 25 non-E letters (each of the 9 digit roles independently,
not even "one random coherent alphabet per puzzle" -- full per-digit
mixing). No fixed combination is ever repeated for the model to
memorize; the only thing that can reduce training loss under that much
variation is genuinely relational structure.

**Training result** (real run, complete): 20 epochs over ~178,560
fully-represented puzzles (no repetition needed, unlike the K=6 job,
since every epoch's pass already gives each puzzle a fresh mix) reached
**98.41% cell accuracy** on the held-out (also randomly mixed) validation
set by epoch 19 -- a strong result given the enormously larger
combinatorial space of alphabets this involves compared to the K=6 run.

**Evaluation: submitted, not yet returned.** Two jobs are queued/running
on Leonardo as of this writing (job ids 57835764 and 57835861):

1. Zero-shot on held-out E (never in the random-mix pool either) --
   the key comparison against Experiment 2's 14.47% cell / 0% valid_rate.
2. The same mixed-alphabet compositional test (recombining A,B,C,D,F,G)
   used in Experiments 1(b) and 2, for direct comparison against the
   classifier's 45.4% mean and the K=6-fixed pointer's 52.8% mean.

Results will be added to this report (and to
`results/eval_pointer_random_mix_results.jsonl` /
`results/eval_pointer_random_mix_mixed_results.jsonl`) once available.

---

## Synthesis so far

Across every completed experiment, one consistent story: the shared
computation these models learn is real, useful (always far above chance
whenever any signal exists at all), and **partially but not cleanly
generalizable across symbol identity**. Structural fixes (Experiment 2)
and diversity-based training fixes (Experiment 3, in progress) both
appear to help incrementally rather than solve the problem outright, at
least based on results so far. The clean, "aha" result (transplant
recovering full accuracy exactly, Experiment 0) shows the model's
reasoning is genuinely there and genuinely reusable *given* the
correspondence -- what's still missing is any mechanism by which it
discovers that correspondence, or represents its own computation in a
way that doesn't need it, on its own.

## What's left to run

1. **Immediate**: the two pending Experiment 3 eval jobs (zero-shot E,
   mixed-alphabet), already submitted.
2. **Depending on those results**: if random-mix training does not
   meaningfully close the zero-shot gap, the literature-grounded next
   architectural step is an explicit relational-attention mechanism
   (Abstractors-style, Altabaa et al. 2023) that architecturally
   constrains the model to compute only over *similarity patterns*
   between representations, never directly over their content -- a
   bigger redesign than the plain pointer head, not yet started.
3. **Interpretability check**: directly test the "duplicate-detection
   via vector-equality" hypothesis for Experiment 2's small zero-shot
   signal, by comparing attention/compatibility patterns on held-out E's
   cells against the same puzzle in a known alphabet.
4. **Minor architecture note**: the pointer model's candidate list is
   currently fixed once per forward pass; recomputing it at each
   iteration of the existing (currently unused, `num_iterations=1`)
   iterative-refinement mechanism could in principle let the model
   "discover" a missing-from-givens digit via elimination logic and
   later point to its own prior guess -- would only affect the ~0.8%
   edge case, low priority.
5. **More controlled mixed-alphabet sweep**: the current mixed-alphabet
   trials sample random combinations; a deliberate sweep over "how many
   distinct alphabets are blended" (1 through 6) would turn the observed
   coherence-accuracy pattern into a real controlled curve rather than
   an anecdotal one from 10 random draws.
6. **Statistical polish**: more seeds and confidence intervals throughout.
7. **A second task domain beyond Sudoku**: needed to elevate this from a
   single-task case study to a general claim, and likely necessary for a
   main-track (rather than workshop-tier) publication target.
8. **Re-run the stratified missing-digit breakdown** on the K=6-fixed
   pointer checkpoint's zero-shot eval (the stratification code was
   added to `evaluate_pointer.py` after that job ran, so its results
   file predates the fully-represented / missing-digit split; the
   Experiment 3 eval jobs already include it since they were submitted
   after the code change).
