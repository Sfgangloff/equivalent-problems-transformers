# Draft Abstract

*Status: draft, based on completed experiments only (see REPORT.md).
Experiment 3 (training-diversity augmentation) is still running; this
abstract will be revised once its results, and any follow-on
architecture work, are in.*

## Working title

**Abstract or Concrete? Probing Transformer Reasoning Through Alphabet
Equivalence in Sudoku**

## Abstract

A reasoner is *abstract* if it applies a general procedure to a
problem's structure, indifferent to how that structure happens to be
concretely instantiated; it is *concrete* if what it has learned is tied
to the particular instantiation it was trained on. This admits a
natural empirical marker: an abstract reasoner should treat different
concrete instances of the same problem interchangeably, so mastering one
should transfer to solving another. We study this marker on a specific
notion of instance equivalence: two problems are *equivalent* when one
is obtained from the other purely by renaming its symbol alphabet,
everything else unchanged. We test this concretely using Sudoku: given a
puzzle P, we construct an equivalent puzzle Q by relabeling P's digit
vocabulary.

A transformer trained to solve P reaches >99% cell accuracy, but
transfers to Q at chance (~11%) with no assistance -- indistinguishable
from an untrained network. Yet copying P's trained per-digit weights
directly onto Q's corresponding symbols, with no retraining, recovers
Q's accuracy exactly. The model's procedure evidently does reuse across
instances once told how their symbols correspond; what it lacks is any
mechanism for discovering that correspondence itself. Reasoning, in that
narrow sense, is abstract; recognizing that a new instance calls for it
is not.

We probe this recognition gap three ways. Pretraining across several
known alphabets does not close it: adapting only a new alphabet's own
parameters, with the shared network frozen, reaches 39.5% on 1,000
examples of an unseen alphabet -- *worse* than an unfrozen model trained
from scratch on the same data (69.2%). Recombining already-known symbols
into unfamiliar combinations, never presented as one coherent alphabet
during training, yields 23-71% accuracy (mean 45%) -- well above chance,
well below the clean transfer an abstract reasoner should show. Removing
the model's fixed, per-alphabet output vocabulary -- replacing it with a
mechanism that selects among a puzzle's own visible symbols -- raises
unseen-alphabet accuracy only to 14.5% (still near chance) and produces
zero fully valid solutions, though it raises the recombination-task mean
to 53%.

These results isolate a specific, quantifiable failure to recognize
alphabet-equivalent instances of the same problem as such -- distinct
from a general failure to reason -- and show that several natural fixes
narrow, but do not close, the gap. This motivates training regimes and
architectures that explicitly separate a problem's relational structure
from the symbols used to present it.
