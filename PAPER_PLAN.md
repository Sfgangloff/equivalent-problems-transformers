# Paper Plan

Title (settled): **Probing Transformer Reasoning Transfer Through
Alphabet Equivalence** — deliberately doesn't name Sudoku: the title
leads with the general question, scope is disclosed honestly in the
abstract and Limitations rather than the title's job.

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
**Status: drafted** (`paper/main.tex`), six paragraphs: neural Sudoku
solving, compositional/systematic generalization (SCAN/COGS), symbol
binding, pointer/copy mechanisms, symbolic mechanisms in pretrained
LLMs, and an explicit positioning paragraph. All 8 citations verified
against actual publication venues (not from memory) and confirmed to
resolve/format correctly via a real `pdflatex`/`bibtex` compile
(`references.bib`):
- Palm, Paquet & Winther (2018), *Recurrent Relational Networks*.
- Webb, Sinha & Cohen (2021), *Emergent Symbols through Binding in
  External Memory* — closest direct precedent for the recognition-gap
  framing.
- Altabaa, Webb, Cohen & Lafferty, *Abstractors* — published at
  **ICLR 2024** (corrected from an earlier "2023" mislabeling, its
  arXiv-preprint year).
- Vinyals, Fortunato & Jaitly (2015), *Pointer Networks*; See, Liu &
  Manning (2017), *Get To The Point* — the pointer/copy-mechanism
  lineage the Section 5.3 architecture sits in.
- Yang, Campbell, Huang, Wang, Cohen & Webb (2025), *Emergent Symbolic
  Mechanisms Support Abstract Reasoning in LLMs* — corrected from an
  earlier "Webb et al. 2025" mislabeling; Webb is the *last* author.
- Lake & Baroni (2018, ICML), the SCAN benchmark, and Kim & Linzen
  (2020, EMNLP), COGS — the systematic/compositional-generalization
  literature pass, now done; used to sharpen how our narrower question
  (same combination, relabeled, vs. novel combinations of familiar
  parts) differs from theirs.

Novelty check on "equivalence via relabeling as a generalization probe"
done via web search (not exhaustive): no direct prior use of this exact
combination (checkable relabeling + weight-transplant control) was
found; the Positioning paragraph states this as a modest claim about
the specific combination, not any single piece in isolation.

### 4. Problem Setup and Method
**Status: drafted** (`paper/main.tex`, `\label{sec:method}`), six
subsections: Task and equivalence (formal definition via alphabets as
injective maps), Data (generation/split), Models (classifier vs. pointer
head, side by side), the weight-transplant control, Metrics (including
why `valid_rate` is primary once ground truth can't be assumed known),
Training protocol. Compiles cleanly (6 pages total so far), no
undefined references.

**Done**: the architecture-contrast figure (Figure~\ref{fig:architectures}
in `main.tex`, placed in the Models subsection right after the
classifier-head/pointer-head paragraphs) is built as a self-contained
TikZ diagram (no external image file), replacing the earlier TODO
comment. Two side-by-side boxes: classifier head (fixed
`Linear(d_model, 9K)`, absolute token ids) vs. pointer head (learned
query projection, dot product against the puzzle's own given-token
embeddings, softmax over ≤9 candidates).

**Still needed**:
- Full hyperparameter table for the Appendix (Training protocol
  subsection currently just gestures at "see Appendix").

### 5. Experiments and Results
**Status: drafted** (`paper/main.tex`, `\label{sec:results}`), five
`\subsection`s with real labels (no more manual ".1"/".2" text-suffix
placeholders -- all earlier cross-references in Introduction/Related
Work/Method updated to point at them properly):

| Subsection | Label | Source experiment |
|---|---|---|
| 5.1 Naive transfer + transplant control | `sec:naive` | Exp 0 |
| 5.2 Diversity alone doesn't close it (few-shot freeze + mixed-alphabet) | `sec:diversity1` | Exp 1 |
| 5.3 Removing the architectural bottleneck | `sec:pointer` | Exp 2 |
| 5.4 Training-diversity split (compositional vs. true zero-shot) | `sec:diversity2` | Exp 3 |
| 5.5 Disentangling pool size vs. mixing-exposure | `sec:disentangle` | K=6-mixed follow-up |

Each subsection: motivating question, setup, result table(s), one-
paragraph interpretation, per the original plan. Compiles cleanly at 9
pages total so far (all sections through Method + Results; Discussion/
Limitations/Conclusion still stubs) -- worth watching against ICLR's
~9-page main-text convention as those fill in and figures are added.

**Done**: both flagged figures built and placed. `paper/figures/
make_figures.py` regenerates both from `results/*.jsonl` (run with
`source .venv/bin/activate && python paper/figures/make_figures.py` from
the repo root). `fig_mixed_alphabet.pdf` (Figure~\ref{fig:mixed-alphabet}
in `main.tex`, end of Section~5.5/\texttt{sec:disentangle}): per-trial
cell accuracy across the four conditions evaluated on the mixed-alphabet
test, makes the variance-collapse finding visually immediate.
`fig_zeroshot_E.pdf` (Figure~\ref{fig:zeroshot-E}, same location): cell
accuracy on held-out E across all five conditions tested, annotated with
the flat 0% valid-rate line. Paper now compiles at 12 pages (up from 11).

**Still needed**:
- Statistical rigor: still single-seed for most conditions beyond the
  few-shot sweep (3 seeds) -- see Open Decisions item 2 in REPORT.md's
  "what's left to run" list for the concrete, motivated instance of this
  (resolving whether the residual zero-shot signal is real or noise).

### 6. Discussion
**Status: drafted** (`paper/main.tex`, `\label{sec:discussion}`).
Synthesizes Sections 5.1-5.5 into the two-part picture (recombination
solved by mixing exposure; true novelty unsolved by anything tried),
connects back to Section 1's abstract/concrete framing (sharpens
"concrete" to mean specifically the recognition step, not the
underlying computation), discusses the vector-equality hypothesis
explicitly flagged as untested, and closes with the Abstractors
architecture framed as future work per the Open Decisions.

### 7. Limitations
**Status: drafted** (`paper/main.tex`, `\label{sec:limitations}`). Four
paragraphs: small-transformer/from-scratch scope, Sudoku-only/single-
transform scope (both stated as deliberate choices, cross-referencing
the Open Decisions rather than presenting them as oversights), the
pointer architecture's 0.80%-of-puzzles structural blind spot, and the
single-seed-per-condition caveat (naming the unresolved residual-signal
question directly, while noting valid_rate, the primary metric, is
unaffected by it).

### 8. Conclusion
**Status: drafted** (`paper/main.tex`). Short, ties back to the
Introduction's contributions list without repeating the Abstract
verbatim; closes on the Abstractors architecture as the best-motivated
remaining direction.

**Full paper compiles cleanly end to end as of this entry**: 12 pages
total (Abstract through Conclusion plus references, now including all
three figures: the architecture diagram and both Results figures), no
undefined references, verified with a real pdflatex/bibtex run. Worth
watching against ICLR's ~9-page main-text convention (excludes
references) once the Appendix is added -- the only item left on the
Figures/tables checklist below.

### Appendix
Full hyperparameters (already in `configs/base.yaml` and code), compute/
GPU-hour accounting (recoverable from SLURM job logs — not currently
tracked in a single place, worth compiling), the `use_query_projection`
ablation if we run it (flag already exists in `train_pointer.py`, never
actually run), additional per-letter breakdowns already sitting in the
raw `results/*.jsonl` files but not in any report prose yet.

## Figures/tables checklist

1. Architecture diagram (classifier vs. pointer head) — **done**,
   Figure~\ref{fig:architectures}, self-contained TikZ in `main.tex`.
2. Mixed-alphabet per-trial accuracy across all conditions (box/strip
   plot) — **done**, `figures/fig_mixed_alphabet.pdf`.
3. Zero-shot-on-E accuracy/valid_rate across all conditions (bar plot) —
   **done**, `figures/fig_zeroshot_E.pdf`.
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
