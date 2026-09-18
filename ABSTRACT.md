# Draft Abstract

*Target venue: ICLR, main track. Constraints to respect on every future
revision: OpenReview enforces a plain-text abstract limit of ~1920
characters (current draft below: ~1856, verify against the actual
current-year CFP closer to submission); main text ~9 pages, references
and appendix unlimited.*

*Status: draft, based on Experiments 0-2 only (see REPORT.md). Experiment
3 (training-diversity augmentation) is complete and found a striking
split result -- training diversity essentially solves compositional
generalization over known symbols, but leaves genuine zero-shot transfer
to an unseen symbol untouched. Deliberately not yet folded in: a
follow-up experiment is needed first to determine whether the relevant
variable behind that split is the SIZE of the training alphabet pool or
simply whether training ever exposes the model to within-puzzle mixing
at all -- the two were confounded in Experiment 3 as run. This
abstract will be revised once that's resolved, so the "further
directions" paragraph states the right independent variable. Given the
character limit, folding in a fourth finding will likely require cutting
one of the existing three probes rather than simply appending.*

## Working title

**Abstract or Concrete? Probing Transformer Reasoning Through Alphabet
Equivalence in Sudoku**

## Abstract

A reasoner is *abstract* if it applies a general procedure to a
problem's structure, indifferent to how that structure is concretely
instantiated; it is *concrete* if what it learned is tied to the
particular instantiation trained on. This admits a marker: an abstract
reasoner should transfer mastery of one problem instance to another. We
study equivalence via alphabet renaming, and give a general recipe for
testing transfer under it: a checkable relabeling, plus a
weight-transplant control (copying trained per-symbol weights onto the
new instance, no retraining) isolating genuine transfer failure from
never-trained parameters. We instantiate this with Sudoku: given puzzle
P, construct equivalent puzzle Q by relabeling P's digits.

A transformer trained on P reaches >99% cell accuracy but transfers to Q
at chance (~11%) unaided -- indistinguishable from an untrained network
-- while the transplant control recovers Q's accuracy exactly. Reasoning,
in that narrow sense, is abstract; recognizing a new instance calls for
it is not.

We probe this gap three ways. Freezing the shared network and adapting
only a new alphabet's parameters reaches 39.5% on 1,000 examples of an
unseen alphabet -- worse than training from scratch on the same data
(69.2%). Recombining known symbols into unfamiliar combinations yields
23-71% accuracy (mean 45%), well above chance but far from clean
transfer. Removing the model's fixed, per-alphabet output vocabulary for
a mechanism that selects among a puzzle's visible symbols raises
unseen-alphabet accuracy only to 14.5%, with zero fully valid solutions,
versus 53% on the recombination task.

These results isolate a specific, quantifiable failure to recognize
alphabet-equivalent instances as such, distinct from a general failure
to reason, and show several natural fixes narrow but do not close the
gap.
