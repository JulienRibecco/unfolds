# unfolds — ML Experiments with Leakage Guardrails

A framework for building, inspecting, and repeating tabular ML experiments.
Born from research where useful-looking features failed to generalize to
unseen entities, it brings explicit data boundaries and leakage guardrails
to the experiment workflow.

Research-tooling alpha: APIs may change as the workflow develops.

Pure numpy/scipy/sklearn. No deep learning frameworks.

## Install

From the repository root, with Python 3.9 or newer:

```bash
python -m pip install --upgrade pip
pip install -e .
```

## Try the validation audit

```bash
python -m examples.entity_validation
```

No download or extra dependencies. This small synthetic example asks whether
a model can predict outcomes for **entities it has never seen**. It compares
two fixed feature sets using random-row versus grouped inner validation,
then evaluates both choices on the same held-out entities.

Example output (seed 42):

| Inner validation | Selected features | Validation R² | Unseen-entity test R² |
|---|---|---:|---:|
| Random rows | Signal + entity fingerprints | 0.980 | -1.812 |
| Grouped entities | Signal only | 0.214 | 0.336 |

The fingerprints help recover offsets for familiar entities, but those offsets
do not transfer. Random-row validation answers a different question from
unseen-entity evaluation. A negative test R² means worse than predicting the
test-set mean; the grouped model still has substantial unexplained variance.

The run saves candidate scores, selected feature names, environment versions,
split assignments, and predictions under `artifacts/`. Use `--seed` to explore
another generated dataset and `--output` to choose the result path. Results
illustrate this construction, not a universal performance gap.

- [Read the DrugComb investigation that inspired it](docs/drugcomb-case-study.md).
- [Run the optional audit on real DrugComb data](examples/drugcomb/README.md),
  or inspect the original research scripts there.

## Why this exists

ML experiment code can leak information from test data into training,
often in ways that are hard to spot:

- **Normalizing before splitting** — fit statistics include test samples
- **Feature selection on the full dataset** — selected features are biased
  toward the test set
- **Dedup after splitting** — near-duplicate rows straddle train/test
- **Peeking at the holdout** — "just one more evaluation" erodes the sanctified set

unfolds provides guardrails through fold-safe preprocessing, explicit
development/holdout separation, and one-time holdout access per dataset
instance:

```
raw data → validate → sanctify → fold → record → unlock holdout (once per instance)
```

Validation checks and lifecycle guards catch common mistakes. Users remain
responsible for preprocessing performed outside the framework and for avoiding
model selection or tuning based on holdout results.

## The data lifecycle

### 1. Validate

```python
from unfolds import ValidatedDataset, dedup_average

result = dedup_average(X, y)          # merge duplicate rows
vd = ValidatedDataset.validate(       # checks: no dups, no constant cols,
    result['X'], result['y'])         # no all-NaN cols
```

Validation runs automatically. You can't skip it by accident — `ValidatedDataset`
can only be created through `validate()`. If your data has exact duplicates,
it blocks. Dedup first, then validate.

**Why dedup before validate, not after?** Because dedup changes the sample
composition. If you validate first and dedup later, validation ran on data
you won't actually use. Dedup is a data preparation step; validation is a
gate that confirms preparation was done correctly.

### 2. Sanctify

```python
from unfolds import SanctifiedDataset

sd = SanctifiedDataset(
    X, y,
    feature_names=['age', 'income', ...],
    seed=42,
    sanctified_fraction=0.15,
)
```

This separates 15% of your data into a holdout. The public `.X`, `.y`
attributes contain only the dev portion. Holdout arrays are stored internally;
`final_evaluate()` provides explicit access and raises on a second call on
the same dataset instance.

**Why "sanctified" instead of "test set"?** Because in practice, test sets
get looked at repeatedly. Every time you check a number on the test set and
then change your model, the test set leaks into your decisions. The
sanctified set makes that boundary explicit, so accessing it is a deliberate
step at the end of development.

### 3. Experiment

```python
from unfolds import ExperimentConfig, Experiment

config = ExperimentConfig(seed=42, sanctified_fraction=0.15, k=5)
exp = Experiment(sd, config)

for fold in exp.folds(k=5):
    model = train(fold.X_train, fold.y_train)
    fold.record(model.predict(fold.X_val))
```

Fold arrays (`X_train`, `y_train`, `X_val`, `y_val`) are write-protected to
catch accidental modification. Dev-relative indices are also available as
`tr_idx` and `te_idx` for aligning auxiliary data.
Grouped k-fold keeps samples sharing a supplied group label in the same fold:

```python
for fold in exp.folds(k=5, group_by='groups'):
    ...
```

Set `ExperimentConfig(group_by='groups')` to use the same policy in
`exp.folds()`, `exp.holdout()`, `exp.run()`, and the research bench. Explicit
`group_by=None` requests random rows. Use `'source'` for source labels.
These options require the corresponding labels on the dataset.

For ordered data, construct `SanctifiedDataset(..., temporal=True,
temporal_gap=5)`. Temporal splitting takes precedence over grouping: development
folds use forward windows, development holdouts use a chronological tail,
and the final holdout also excludes the configured gap from training.
Rows must already be chronological. Gap rows are omitted from the relevant
training/evaluation pair, so their sizes need not sum to the original count.
Configure temporal settings on the dataset itself; `ExperimentConfig` fields
are also available for loaders to forward when constructing it.

**Why read-only arrays?** Because `X_train[0] = 999` is a real bug that
happens in research code. Write-protected views catch it immediately.

### 4. Final evaluation

```python
final = exp.final_evaluate()   # one-shot: raises on second call
model = train(final.X_dev, final.y_dev)
exp.record_final(model.predict(final.X_sanct))
results = exp.recap()
```

`final_evaluate()` guards access once per dataset instance. Choose your model
and settings using development results before calling it. The returned arrays
include holdout labels for scoring and error analysis; the guard does not
prevent reuse of those arrays or access through a new dataset instance.

### 5. Research bench

For multi-experiment workflows with CLI, history tracking, and automatic
research notes:

```python
from unfolds import Research, ExperimentConfig

config = ExperimentConfig(seed=42, sanctified_fraction=0.15, k=5)
research = Research("My Project", loader_fn, config,
                    save_dir="data/my-project",
                    notes_path="research/RESEARCH_NOTES.md")

research.new_experiment("baseline", baseline_factory)
research.new_experiment("cascade", cascade_factory)
research.main()   # CLI: --quick, --exp baseline, --folds 3
```

Run history is persisted to JSON. Compare experiments:

```python
research.runs().compare("baseline", "cascade")
```

The research bench unlocks the holdout once and evaluates all registered
experiments on it. Use development metrics to select models; holdout
comparisons should be final reporting, not feedback for further tuning.

## Fold-safe preprocessing

The fold-safe preprocessing helpers compute statistics from the supplied
training arrays or training indices, then apply them to validation data:

```python
from unfolds import fold_normalize, fold_impute

Xn_tr, Xn_te, means, stds = fold_normalize(X_train, X_test)
X_tr_imp, X_te_imp, medians = fold_impute(X, train_idx, test_idx)
```

**Why these helpers?** They package training-statistic computation and
validation-data transformation into one call. `StandardScaler` fitted only
on training data is also valid. In either case, callers must supply the
correct split and avoid fitting preprocessing on the full dataset beforehand.

If a column is entirely missing in the training fold, median imputation raises
a clear error. Drop that column based on training data, or apply a justified
fill value first. Validation values cannot supply the missing statistic.

## Model composables

Build complex models from simple pieces:

```python
import numpy as np

from unfolds import NLModel, EnsembleModel, StackedModel, RoutedModel

# Ensemble: same architecture, different seeds
ensemble = EnsembleModel(base=NLModel((8,)), n_seeds=5)

# Stacking: stage 1 predictions augment stage 2 input
stacked = StackedModel(
    first=NLModel((8,)),
    second=NLModel((4,)),
    augment='append',
    oof_folds=5,          # each predicted row is excluded from its inner fit
)

# Routing: partition input space, specialize per region
routed = RoutedModel(
    router=NLModel((4,)),
    experts={'low': NLModel((4,)), 'high': NLModel((8,))},
    route_fn=lambda X, pred: np.where(pred < 50, 'low', 'high'),
)
```

All implement `fit(X, y)` / `predict(X)` / `clone()`. Composable with
the experiment framework — pass any of these as a model factory.

The default stacking strategy uses row-based inner folds. For unseen-entity
evaluation, choose grouped inner splits too and pass labels aligned with each
training fold:

```python
sd = SanctifiedDataset(X, y, groups=groups)
exp = Experiment(sd, ExperimentConfig(k=3, group_by='groups'))

def make_stack():
    return StackedModel(NLModel((8,)), NLModel((4,)),
                        oof_folds=3, oof_strategy='groups')

for fold in exp.folds(group_by='groups'):
    model = make_stack()
    model.fit(fold.X_train, fold.y_train, groups=sd.groups[fold.tr_idx])
    fold.record(model.predict(fold.X_val))

final = exp.final_evaluate()
model = make_stack()
model.fit(final.X_dev, final.y_dev, groups=sd.groups)
exp.record_final(model.predict(final.X_sanct))
```

Grouped stacking requires this explicit loop: `Experiment.run()` and `Research`
do not forward group metadata to model fits, and missing labels raise an error.
Choose an inner fold count no larger than the number of training groups.

For chronological rows, use `oof_strategy='temporal', oof_gap=5`. Each inner
fit uses only earlier rows with the requested gap. Stage two trains only on
rows with forward predictions; the initial prefix and uncovered gap rows are
excluded. `model.oof_train_idx_` records those stage-two training indices.
Set both the dataset's temporal policy and the stack's inner policy explicitly.
Outer grouped/temporal evaluation does not configure a model's inner splits.

### Declarative cascades

```python
from unfolds import CascadeConfig, BinConfig, build_cascade

config = CascadeConfig(
    router=EnsembleModel(base=NLModel((8,)), n_seeds=3),
    thresholds=[10, 50],
    bins=[
        BinConfig("low",  NLModel((4,)),  train=(0, 15)),
        BinConfig("mid",  NLModel((8,)),  train=(8, 60)),
        BinConfig("high", NLModel((4,)),  train=(40, 200)),
    ],
)
model = build_cascade(config, seed=42)
```

Overlapping training ranges (`train=(0, 15)` and `train=(8, 60)`) are
intentional — experts see samples near their boundaries, preventing
hard-cutoff artifacts.

## Pipeline

Chain feature engineering steps with models:

**Current scope:** feature steps run jointly on training data and a supplied
evaluation batch. A fitted pipeline with steps cannot transform a new raw batch
through `predict(X_new)`; supply `X_val` during `fit()` and use `predict()`.
This API supports experiment evaluation. For reusable inference, use a fitted
scikit-learn pipeline or a model that owns its fitted transformations.
Separate step `fit`/`transform` support is deferred to a future API revision.

```python
from unfolds import Pipeline, Step

class MyStep(Step):
    def execute(self, X_train, y, feature_names, X_val=None):
        # transform X_train and X_val
        return {'X_train': ..., 'X_val': ...,
                'feature_names': ..., 'meta': {}}

pipe = Pipeline()
pipe.add(MyStep())
pipe.set_model(NLModel((8,)))

for fold in exp.folds(k=5):
    pipe.fit(fold.X_train, fold.y_train, exp.feature_names,
             X_val=fold.X_val)
    fold.record(pipe.predict())
```

Domain libraries can register step types for string-based lookup:

```python
from unfolds import register_step_type
register_step_type('mystep', MyStep)

pipe.add('mystep', param1=10)  # works after registration
```

## Hierarchical ridge

For cascades with discrete routing variables:

```python
from unfolds import HierarchicalRidge

hr = HierarchicalRidge(alpha=0.1, shrinkage=0.2, min_samples=[10, 5])
hr.fit(X_norm, y, [coarse_groups, fine_groups])
pred = hr.predict(X_norm_test, [coarse_test, fine_test])
```

N-level hierarchy: global → per-group → per-(group × subgroup).
Prediction falls back to coarser levels when a group is unseen or
has too few samples. Shrinkage blends local and parent estimates.

## Design decisions

**No configuration files.** Everything is code. Configuration objects
(`ExperimentConfig`, `CascadeConfig`) are dataclasses — inspectable,
diffable, version-controllable.

**Explicit persistence.** Manual experiments record predictions with
`fold.record()`. `Experiment.run()` and `Research` record them automatically.
The research bench keeps run history in memory and, when `save_dir` is set,
writes history and model snapshots to that directory. Notes are appended when
you call `research.note()`.

**No distributed computing.** Single-machine, single-process. The
framework runs experiments sequentially because reproducibility
matters more than speed. If you need parallelism, parallelize at
the experiment level (different seeds), not inside.

**Practical guardrails.** Read-only fold arrays and explicit, one-time
holdout access help catch accidental misuse while keeping research workflows
flexible. Holdout labels remain available after access for error analysis.
These protections support evaluation discipline; they do not prevent
deliberate bypass or enforce how results influence later decisions.

## Dependencies

- numpy ≥ 1.21
- scipy ≥ 1.7
- scikit-learn ≥ 1.0

No deep learning frameworks. The `NLModel` is a from-scratch sigmoid
network in pure numpy — small, fast, no GPU needed.

## Development

Install the package with `pip install -e .`, then run the tests:

```bash
python -m unittest discover -s tests -v
```

The suite uses Python's built-in test runner. It covers splitting,
preprocessing, holdout access, model snapshots, and examples. The California
Housing example is exercised with synthetic data, so tests need no download.
GitHub Actions runs the core suite on Python 3.9 and 3.13, and tests optional
DrugComb preparation with small local fixtures on Python 3.10.
Core regression tests also run outside the checkout against the installed
package, including gradient accuracy, split-policy propagation, grouped and
temporal stacking, cascade settings, and scikit-learn composition.

## License

[CC BY 4.0](LICENSE), matching the repository's original license declaration.
