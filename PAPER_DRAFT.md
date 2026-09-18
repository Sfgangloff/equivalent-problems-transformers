# Paper Draft

Growing draft, section by section, per `PAPER_PLAN.md`. Sections not yet
drafted are marked `[NOT DRAFTED]`. Citations marked `[CITE: ...]` are
placeholders for a literature pass not yet done (see `PAPER_PLAN.md`'s
Related Work notes) -- do not treat these as verified references.

---

## Title

Abstract or Concrete? Probing Transformer Reasoning Through Alphabet
Equivalence in Sudoku

*(working title, may change)*

## Abstract

See `ABSTRACT.md` for the current version (kept separate since it's
being iterated against a hard character limit; will be inlined here once
stable).

## 1. Introduction

A model that has learned to solve a problem can be said to reason
*abstractly* about it if the procedure it applies operates on the
problem's structure, indifferent to how that structure happens to be
dressed up in a particular presentation. It reasons only *concretely* if
what it has learned is tied to the specific surface form it was trained
on, rather than to the structure underneath. This distinction is central
to any claim that a learned system reasons rather than merely fits
patterns in its training distribution, but such claims are frequently
made without a test that could actually distinguish the two
possibilities. [CITE: prior framings of systematic/compositional
generalization in neural networks -- literature pass not yet done.]

We propose a concrete, checkable operationalization of this distinction.
An abstract reasoner should treat two different instances of the same
underlying problem interchangeably: having mastered one, it should
transfer to solving the other, since its procedure does not depend on
which particular instantiation it happens to face. We instantiate
"different instances of the same problem" through the simplest transform
that leaves a problem's structure completely intact while changing its
surface form entirely: renaming its symbol alphabet. Two problems $P$
and $Q$ are *equivalent* under this definition when $Q$ is obtained from
$P$ purely by relabeling which symbols represent which underlying roles
-- nothing about the structure, constraints, or solution changes, only
the names used to present it.

This gives more than a single empirical test: it gives a general recipe
applicable to any symbolic or combinatorial domain with an analogous
notion of relabeling. The recipe has two parts. First, define
equivalence via a checkable relabeling, and measure whether a model
trained on one instance transfers, unaided, to an equivalent one.
Second -- and this is the part that is easy to get wrong -- use a
targeted control to separate two failure modes that a naive transfer
evaluation cannot distinguish on its own. A model may fail on a new
instance either because its underlying reasoning does not generalize, or
simply because the new instance's symbols were never seen during
training, so any parameters associated with them are untrained noise
regardless of how good the model's reasoning otherwise is. We isolate
the second explanation with a *weight-transplant* control: after
training, we copy the parameters associated with each of $P$'s symbols
directly onto the parameters for $Q$'s corresponding symbols, using the
known-by-construction correspondence, with no further training. This
tells us how the model would perform on $Q$ if it already knew the
correspondence between the two instances' symbols -- precisely the
information a naive zero-shot evaluation cannot provide on its own.

We instantiate this recipe concretely using Sudoku: a well-defined,
verifiable, and cheaply generated combinatorial puzzle whose solution is
invariant to any consistent renaming of its 9 symbols, making it a clean
testbed for this question. Training a transformer to solve Sudoku
puzzles written in one symbol alphabet, we find that it transfers to an
equivalent puzzle in a different alphabet at chance level -- performance
indistinguishable from a network that was never trained at all -- even
though the weight-transplant control recovers its original accuracy
*exactly*. This is the paper's central result: the model's procedure
does reuse across equivalent instances once it is told how their symbols
correspond; what it lacks is any mechanism for discovering that
correspondence on its own. We then probe this recognition gap along
three further axes -- pretraining diversity, an architectural change
that removes a specific structural bottleneck, and training-time data
diversity at a much larger scale -- and show that each narrows, but does
not close, the gap, with a further finding that training diversity
resolves the gap into two distinct sub-problems with different answers:
recombining already-known symbols is solved by sufficient diversity,
while transfer to a genuinely novel symbol is not.

**Contributions.**

- An operationalized, reusable methodology for testing knowledge
  transfer between equivalent instances of a symbolic problem, including
  a weight-transplant control that isolates genuine reasoning-transfer
  failure from the confound of never-trained parameters.
- Applied to Sudoku, we show naive transfer across alphabet-equivalent
  puzzles fails completely -- statistically indistinguishable from an
  untrained network -- while the transplant control recovers full
  performance exactly, demonstrating the failure is one of recognition,
  not of reusable computation (Section [X.1]).
- We show this recognition gap persists across two further natural
  interventions: pretraining diversity across several known alphabets,
  and a redesigned architecture that removes a specific structural
  bottleneck that prevents zero-shot generalization by construction --
  each narrows but does not close the gap (Sections [X.2]-[X.3]).
- We show training-time data diversity resolves this gap into two
  distinct sub-problems with different answers: recombining
  already-known symbols in novel combinations is essentially solved by
  sufficient training diversity, while transfer to a genuinely novel,
  never-seen symbol remains unsolved by every intervention tested
  (Section [X.4]).

*(Roadmap paragraph and exact section cross-references to be added once
the Results section numbering is finalized -- currently pending the
disentangling experiment's result, see PAPER_PLAN.md Section 5.5.)*

---

## 2. Related Work

`[NOT DRAFTED]` -- see `PAPER_PLAN.md` for the source list and the
literature-pass items still needed.

## 3. Problem Setup and Method

`[NOT DRAFTED]` -- most content already exists as prose in `REPORT.md`'s
"Common setup" section and in code docstrings; needs compression and a
formal-notation pass, not new material.

## 4. Experiments and Results

`[NOT DRAFTED]` -- content exists in `REPORT.md`; needs conversion to
paper prose plus the figures listed in `PAPER_PLAN.md` (currently zero
exist).

## 5. Discussion

`[NOT DRAFTED]` -- exists as prose in `REPORT.md`'s Synthesis section at
close-to-paper quality; needs the disentangling result folded in and
connection back to Section 1's framing.

## 6. Limitations

`[NOT DRAFTED]`.

## 7. Conclusion

`[NOT DRAFTED]`.
