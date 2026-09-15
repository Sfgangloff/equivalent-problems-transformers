# Draft Abstract

*Status: draft, based on completed experiments only (see REPORT.md).
Experiment 3 (training-diversity augmentation) is still running; this
abstract will be revised once its results, and any follow-on
architecture work, are in.*

## Working title

**Do Transformers Learn Alphabet-Agnostic Reasoning? A Case Study on
Symbol Equivalence in Sudoku**

## Abstract

A central question for neural reasoning systems is whether they learn
abstract, structure-based competence or a narrower competence tied to
the specific symbols seen during training. We study this question
concretely using Sudoku, constructing an "equivalent" puzzle Q from any
puzzle P by relabeling P's digit vocabulary: Q has the exact same
combinatorial structure and solution as P, differing only in which
tokens represent which digits. We show that a transformer trained to
solve P with near-perfect accuracy (>99% cell accuracy) transfers to Q
at chance level with no assistance -- performance statistically
indistinguishable from an untrained network -- even though supplying the
model with the known symbol correspondence (via a targeted weight
transplant, with no retraining) recovers its original accuracy exactly.
This shows the model's competence is real but is represented in a way
entangled with the specific token identities it trained on, not
abstracted away from them.

We probe this entanglement from three further directions. First,
pretraining across several known alphabets does not yield a portable,
reusable representation: a model with its shared body frozen and only a
new alphabet's own parameters allowed to adapt performs *worse*, given
the same small amount of data, than an unfrozen model trained from
scratch. Second, recombining already-individually-known symbols into
combinations never presented as one coherent alphabet during training
produces graded, partial degradation -- well above chance, well below
ceiling -- rather than clean compositional generalization, with accuracy
tracking how "coherent" the specific recombination is. Third, an
architectural change that removes the fixed, per-alphabet output
vocabulary entirely -- replacing it with a mechanism that selects among a
puzzle's own visible symbols rather than classifying into a global,
alphabet-indexed vocabulary -- yields a measurable but modest
improvement on both the recombination task and genuine zero-shot
transfer to an unseen alphabet, without resolving either.

Together these results characterize a specific, quantifiable failure of
symbol-invariant generalization in a controlled combinatorial reasoning
setting, isolate it from the confound of simply-never-having-seen a
given token (via the transplant control), and identify architectural and
training interventions that partially, but do not fully, address it --
motivating further work on training regimes and architectures that
explicitly separate relational structure from symbol identity.
