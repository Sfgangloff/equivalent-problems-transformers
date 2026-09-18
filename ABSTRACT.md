# Draft Abstract

*Target venue: ICLR, main track. Constraints to respect on every future
revision: OpenReview enforces a plain-text abstract limit of ~1920
characters (current draft below: ~1856, verify against the actual
current-year CFP closer to submission); main text ~9 pages, references
and appendix unlimited.*

*Status: draft, based on Experiments 0-2 only (see REPORT.md).
Experiment 3 and its disentangling follow-up are both complete
(training diversity essentially solves compositional generalization
over known symbols via mixing-exposure, not pool size, but leaves
genuine zero-shot transfer to an unseen symbol untouched) but not yet
folded in here -- a decision on how to fit that fourth finding into the
results paragraph without re-bloating it is still open. Numbers were
deliberately removed from the results paragraphs (2026-09-18 revision):
the abstract should convey the shape of each finding (fails completely
vs. recovers exactly; graded and partial vs. clean; helps one problem
substantially and the other only marginally) rather than its magnitude
-- precise numbers belong in the paper body, not a 30-second skim.*

## Working title

**Probing Transformer Reasoning Transfer Through Alphabet Equivalence**

## Abstract

A reasoner is *abstract* if it applies a general procedure to a
problem's structure, indifferent to how that structure is concretely
instantiated; it is *concrete* if what it learned is tied to the
particular instantiation trained on. This gives a testable marker: an
abstract reasoner should transfer mastery of one problem instance to
another. We study equivalence via alphabet renaming, and test transfer
using a weight-transplant control -- copying trained per-symbol weights
onto the new instance, no retraining -- distinguishing genuine reasoning
failure from symbols never trained on. We instantiate this with Sudoku,
where an equivalent puzzle is any relabeling of its digits.

A transformer trained on one such puzzle transfers to an equivalent one
no better than an untrained network -- yet the transplant control
recovers its original performance exactly. Reasoning, in that narrow
sense, is abstract; recognizing that a new instance calls for it is not.

We probe this recognition gap three ways. Freezing the shared network
and adapting only a new alphabet's parameters fares worse than training
an unfrozen model from scratch on the same data: prior exposure to other
alphabets does not ease learning a new one. Recombining already-known
symbols into unfamiliar combinations yields graded, partial success --
above chance, short of the seamless transfer an abstract reasoner should
show, degrading as combinations diverge from training. Replacing the
model's fixed, per-alphabet output vocabulary with a mechanism that
selects among a puzzle's own visible symbols helps substantially with
recombination, but only marginally with transfer to a truly unseen
alphabet.

These results isolate a specific failure to recognize alphabet-
equivalent instances as such, distinct from a general failure to reason,
and show several natural interventions narrow, but do not close, this
gap.
