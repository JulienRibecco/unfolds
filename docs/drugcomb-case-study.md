# DrugComb: when validation answers the wrong question

In a drug-combination experiment, I worked with 246,776 measurements across
49 matched cell lines. Each row combined descriptors for two drugs with
gene-expression features for the cell line. The target was measured Bliss
synergy.

The intended question was: **does the model generalize to cell lines it has
never seen?**

## The hidden mismatch

That question mattered inside feature selection as well as in the outer
evaluation. The feature-engineering pipeline used random internal folds to
score candidate transformations. Those folds could contain measurements from
the same cell line on both sides.

This gave the selector a shortcut. Gene-expression patterns could distinguish
cell lines already represented in training without helping predict outcomes
for an unseen cell line. Features that looked useful under internal validation
could hurt performance under the intended evaluation.

I changed feature selection to score on an explicit validation set containing
held-out cell lines, and disabled its internal random-fold scoring. Training,
selection-validation, and final test cell lines were separated.

## What the original investigation recorded

The [retained research notes](../examples/drugcomb/historical/research-notes.md)
report the following held-out-cell-line results under the revised procedure:

| Configuration | R² |
|---|---:|
| Raw-feature ridge baseline | 0.113 |
| Ridge with selected within-domain and cross-domain features | 0.133 |

Earlier exploratory runs recorded that features selected using random
internal folds could hurt R² by approximately 0.01. Those runs used different
sample sizes and configurations. They document the investigation, but are
**not a controlled before/after comparison of splitting strategies**.

These are historical results, not scores reproduced by the new examples.
The original preparation also selected the 200 highest-variance genes across
all matched cell lines before splitting. That exposed held-out expression
statistics to feature selection. The new real-data audit corrects this by
selecting genes separately on each training fold and retaining all genes in
the prepared dataset.

## The engineering lesson

**A validation boundary needs to survive every stage that makes a modeling
decision.** A grouped outer split does not automatically make random inner
feature-selection folds appropriate for unseen-entity generalization.

`unfolds` provides building blocks for implementing that discipline:

- Grouped index generators express which entities must stay together.
- Fold-safe preprocessing estimates statistics from training data.
- Experiment objects expose development folds and explicit holdout access.
- Recorded predictions make evaluation inspectable and repeatable.

The framework cannot infer what generalization means for a dataset. Custom
feature-selection code must preserve the chosen protocol. This example uses
the public index generators to audit two procedures against one shared test
set; it deliberately reports both frozen choices at the end.

Random-row evaluation can be useful when future observations come from
already-seen entities. Grouped evaluation addresses unseen entities. The
mistake is claiming one result answers the other question.

## Explore it

The [synthetic demo](../examples/entity_validation.py) isolates the mechanism
with 40 entities and 2,000 rows. It holds the data, feature candidates, ridge
settings, and final test entities fixed; only the inner validation split
changes. It is a controlled illustration, not simulated biological evidence.

The [real-data companion](../examples/drugcomb/README.md) provides upstream
downloads, verified input hashes, a compact preparation format, and a new
ridge audit on DrugComb and GDSC2. It also includes the original scripts for
inspection. The new audit uses a smaller, explicit candidate set and has no
dependency on the original feature-engineering library.

## Data references

- [TDC DrugComb description](https://tdcommons.ai/multi_pred_tasks/drugsyn/#drugcomb)
- [TDC drug-response datasets](https://tdcommons.ai/multi_pred_tasks/drugres/)
- [DrugComb, original publication](https://doi.org/10.1093/nar/gkz337)

The public TDC DrugComb release contains 297,098 measurements across 59 cell
lines. The historical matching to GDSC2 retains 246,776 measurements across
49 cell lines; the preparation manifest records the matching and exclusions.
