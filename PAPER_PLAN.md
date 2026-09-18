# Paper Plan

Working title: **Abstract or Concrete? Probing Transformer Reasoning
Through Alphabet Equivalence in Sudoku** (see `ABSTRACT.md` — title/
abstract may still change as the disentangling experiment and beyond
land).

This is a section-by-section outline: what each section argues, what
material already exists for it (and where), and what's still needed.
Target venue is **ICLR**, main track, Sudoku-only scope (no second task
domain), and the Abstractors-style architecture experiment is future
work rather than required before submission -- all three scope
decisions are now resolved; see **Open decisions** at the end for the
record.

## Narrative arc

The cleanest story, given everything run so far, is a "detective story"
structure — each section asks a question the previous one's result left
open:

1. Does the model discover problem-equivalence on its own? **No** — and
   a control isolates *why*: not because its reasoning fails to reuse,
   but because it never learns to recognize a new instance calls for it.
2. Is that just insufficient training diversity, in the ordinary sense?
   **No** — more known alphabets, and a frozen "reusable" body, don't
   help; recombining even well-known symbols already shows graded
   failure.
3. Is it an architectural bottleneck (a fixed, per-alphabet vocabulary)?
   **Partially** — removing it structurally enables better transfer, but
   only realizes a small fraction of it.
4. Is it *still* just a training-diversity problem, at a bigger scale?
   **Split answer** — training on a much larger, always-mixed space of
   known symbols essentially *solves* recombination, but leaves transfer
   to a genuinely novel symbol completely untouched.
5. What's the actual variable behind that split? (**disentangling
   experiment, running now**) — pool size, or ever having seen
   within-instance mixing at all.
6. (If pursued) Can an explicit relational architecture close the
   remaining genuine-zero-shot gap that no data intervention touched?

Each numbered question becomes a Results subsection.

## Sections

### 1. Abstract
**Status**: drafted (`ABSTRACT.md`), pending the disentangling result and
a decision on whether to include Experiment 3 (see open decisions).

### 2. Introduction
**Argues**: reasoning-generalization claims about neural networks are
often untestable as stated; we give an operational, checkable version
(instance equivalence via relabeling) plus a control (weight transplant)
that separates genuine reasoning-transfer failure from the simpler
never-trained-parameter confound — a reusable recipe, not just a Sudoku
result. Ends with a contributions list.

**Status**: not written as prose. All the content exists across this
conversation and `REPORT.md`'s intro/synthesis sections — needs actual
writing, not new material. ~1 page.

**Needed**: 3-5 sentence motivating frame (cite systematic/compositional
generalization literature — see Related Work list below), the
operationalization paragraph (already drafted for the abstract, expand
slightly), a bulleted contributions list (~4 items mapping to Results
subsections 1-4/5 above).

### 3. Related Work
**Status**: a source list exists (gathered when Experiment 2 was
designed) but never turned into prose or checked for completeness.

Known-relevant, already vetted:
- Palm, Paquet & Winther (2018), *Recurrent Relational Networks* —
  neural Sudoku solving via message passing (this repo's model is in
  that lineage).
- Webb, Sinha & Cohen (2021), *Emergent Symbols through Binding in
  External Memory* — the closest direct precedent for the recognition-
  gap framing.
- Altabaa, Webb, Cohen & Lafferty (2023), *Abstractors* — the likely
  next-architecture citation if we pursue that experiment.
- Vinyals, Fortunato & Jaitly (2015), *Pointer Networks*; See, Liu &
  Manning (2017), *Get To The Point* — the pointer/copy-mechanism
  lineage our Experiment 2 architecture sits in.
- Webb et al. (2025), *Emergent Symbolic Mechanisms Support Abstract
  Reasoning in LLMs* — shows this question is live for pretrained LLMs
  too, good framing material for the intro or discussion.

**Needed**: a literature pass specifically on (a) systematic/
compositional generalization benchmarks (SCAN, COGS, and similar — not
yet searched at all), (b) any existing work that already does
"equivalence via relabeling" as a generalization probe outside this
project (worth checking novelty claims aren't overstated), (c) proper
citation formatting once a venue/style is picked.

### 4. Problem Setup and Method
**Argues**: formal equivalence definition, the transplant control, the
Sudoku testbed, both model architectures, and the metrics (especially
why `valid_rate` is the right primary metric once ground-truth
correspondence can't be assumed known).

**Status**: essentially all written already, scattered across
`REPORT.md`'s "Common setup" and each experiment's "Design" paragraphs,
plus the actual code docstrings (`sudoku/alphabets.py`,
`sudoku/pointer_dataset.py`, `model/pointer_transformer.py`) which are
already written at publication-appropriate precision. Converting this
section is mostly compression + one clean formal-notation pass, not new
content.

**Needed**: 
- A figure contrasting the two architectures (classifier: fixed
  `Linear(d, 9K)` head; pointer: per-instance candidate selection) —
  **no figure exists yet**, this is a clear gap.
- Formal notation for "equivalence" (P ~ Q via relabeling) if the venue
  expects it.

### 5. Experiments and Results
Organized by the six questions in the narrative arc above, each as a
subsection. Per subsection: motivating question, setup, result table,
one-paragraph interpretation.

| Subsection | Source experiment | Status |
|---|---|---|
| 5.1 Naive transfer + transplant control | Exp 0 | Complete, written (`results/2026-09-07_two_alphabet_sudoku.md`, `REPORT.md`) |
| 5.2 Diversity alone doesn't close it (few-shot freeze + mixed-alphabet) | Exp 1 | Complete, written (`results/2026-09-09_multi_alphabet_experiments.md`, `REPORT.md`) |
| 5.3 Removing the architectural bottleneck | Exp 2 | Complete, written (`REPORT.md`) |
| 5.4 Training-diversity split (compositional vs. genuine zero-shot) | Exp 3 | Complete, written (`REPORT.md`) |
| 5.5 Disentangling pool size vs. mixing-exposure | K=6-mixed run | **Running now** (job `58017245` on Leonardo) |

Section 5.6 (explicit relational architecture, Abstractors-style) is
**not** a Results subsection -- resolved as future work, see Discussion
and Open decisions.

**Needed beyond writing**:
- **Figures, currently zero exist.** The single highest-value figure for
  this paper: a plot showing per-trial accuracy across the mixed-
  alphabet eval for all conditions run so far (classifier/K=6-fixed;
  pointer/K=6-fixed; pointer/K=25-random-mix; pointer/K=6-random-mix
  once 5.5 lands) — a box-plot or strip-plot would visually make the
  "variance collapse" finding immediate in a way the current tables
  don't. This is buildable right now from data already in
  `results/*.jsonl`, no new experiments needed.
- A second figure: zero-shot-on-E accuracy/valid_rate across the same
  conditions, showing the flat line at 0% valid_rate throughout.
- Statistical rigor: currently single-seed for most conditions (the
  few-shot sweep used 3 seeds; the mixed-alphabet trials use 10 random
  draws but each condition is one training run). More seeds per training
  condition would strengthen this section for a main-track submission
  but is a real compute/time cost — see open decisions.

### 6. Discussion
**Argues**: synthesizes 5.1-5.5 into the two-part
picture — a "compositional coverage" sub-problem that training diversity
solves, and a "genuine novelty" sub-problem that nothing tried so far
touches — and connects back to the abstract/concrete framing from the
introduction. Discusses the (currently unverified) vector-equality
hypothesis for the small residual zero-shot signal as a direction for
future interpretability work, explicitly flagged as speculative.

**Status**: exists as prose in `REPORT.md`'s Synthesis section, written
at close-to-paper quality already — mostly needs the disentangling
result folded in once available, and expansion connecting back to the
Related Work framing.

### 7. Limitations
**Needed** (not yet written as a dedicated section, though the material
exists as scattered caveats throughout `REPORT.md`):
- Single small transformer trained from scratch, single task domain —
  not a claim about large pretrained models or other domains.
- Only one equivalence transform tested (alphabet relabeling); others
  from the original framing (permutation, notation change, premise
  reordering) untested.
- The pointer architecture's ~0.8%-of-puzzles structural blind spot
  (missing-digit-from-givens case).
- Single seed per major training condition (see Discussion above).

### 8. Conclusion
**Status**: not written; short, should be straightforward once
Discussion is finalized.

### Appendix
Full hyperparameters (already in `configs/base.yaml` and code), compute/
GPU-hour accounting (recoverable from SLURM job logs — not currently
tracked in a single place, worth compiling), the `use_query_projection`
ablation if we run it (flag already exists in `train_pointer.py`, never
actually run), additional per-letter breakdowns already sitting in the
raw `results/*.jsonl` files but not in any report prose yet.

## Figures/tables checklist (nothing built yet)

1. Architecture diagram (classifier vs. pointer head).
2. Mixed-alphabet per-trial accuracy across all conditions (box/strip
   plot) — buildable now from existing data.
3. Zero-shot-on-E accuracy/valid_rate across all conditions (bar or line
   plot) — buildable now from existing data.
4. (If 5.6 is run) The same two plots extended with the relational-
   architecture condition.

## Open decisions

1. ~~Target venue and page budget.~~ **Resolved: ICLR.** Main track,
   double-blind (so the repo/commits will need an anonymization pass
   before submission -- not urgent now, but worth remembering later).
   ICLR's exact page limit varies slightly by year; plan around the
   ~9-page main-text convention until closer to the actual deadline, at
   which point re-check the current year's call for papers.
2. ~~Is the Abstractors-style architecture (5.6) required before
   submission?~~ **Resolved: no, future work.** Section 5.6 becomes a
   forward-looking paragraph in Discussion/Conclusion (the natural next
   architectural step, motivated by 5.4/5.5's finding that no data-side
   intervention touches the genuine-zero-shot gap) rather than a
   completed experiment in Results.
3. ~~Is a second task domain beyond Sudoku required?~~ **Resolved: no,
   not for now.** Scope stays Sudoku-only. The Limitations section
   (Section 7) should state this as a deliberate scope choice (a
   controlled single-domain study) rather than an oversight.
