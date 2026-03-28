# unfolds — Leakage-Free Experiment Framework

A framework for running ML experiments on tabular data without accidentally
contaminating your results. Built from years of watching subtle data leakage
silently inflate published numbers.

Pure numpy/scipy/sklearn. No deep learning frameworks.

## Install

```bash
pip install -e .
```

## Why this exists

Most ML experiment code leaks information from test data into training,
usually in ways that are hard to spot:

- **Normalizing before splitting** — fit statistics include test samples
- **Feature selection on the full dataset** — selected features are biased
  toward the test set
- **Dedup after splitting** — near-duplicate rows straddle train/test
- **Peeking at the holdout** — "just one more evaluation" erodes the sanctified set

unfolds makes these mistakes structurally impossible by enforcing a strict
data lifecycle:

```
raw data → validate → sanctify → fold → record → final evaluate (once)
```

Each stage has a contract. Break the contract, get an exception.

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

This physically separates 15% of your data into a vault. The public `.X`, `.y`
properties return only the dev portion. The sanctified data is private — you
literally cannot access it except through `final_evaluate()`, and that method
can only be called once.

**Why "sanctified" instead of "test set"?** Because in practice, test sets
get looked at repeatedly. Every time you check a number on the test set and
then change your model, the test set leaks into your decisions. The
sanctified set enforces the discipline: you get one shot.

### 3. Experiment

```python
from unfolds import ExperimentConfig, Experiment

config = ExperimentConfig(seed=42, sanctified_fraction=0.15, k=5)
exp = Experiment(sd, config)

for fold in exp.folds(k=5):
    model = train(fold.X_train, fold.y_train)
    fold.record(model.predict(fold.X_val))
```

Folds are read-only views. You cannot modify the arrays (they're write-protected).
You cannot access fold indices directly — just `X_train`, `y_train`, `X_val`, `y_val`.
Grouped k-fold (preventing group leakage) is built in:

```python
for fold in exp.folds(k=5, group_by='groups'):
    ...
```

**Why read-only arrays?** Because `X_train[0] = 999` is a real bug that
happens in research code. Write-protected views catch it immediately.

### 4. Final evaluation

```python
final = exp.final_evaluate()   # one-shot: raises on second call
model = train(final.X_dev, final.y_dev)
exp.record_final(model.predict(final.X_sanct))
results = exp.recap()
```

`final_evaluate()` is guarded. Call it twice, get an exception. This forces
you to commit to a model before seeing the sanctified results.

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

## Fold-safe preprocessing

All preprocessing functions take train indices and compute statistics
from training data only:

```python
from unfolds import fold_normalize, fold_impute

Xn_tr, Xn_te, means, stds = fold_normalize(X_train, X_test)
X_tr_imp, X_te_imp, medians = fold_impute(X, train_idx, test_idx)
```

**Why not just `StandardScaler`?** Because `scaler.fit(X_train)` is easy
to forget when you're deep in a loop. These functions make it impossible
to accidentally fit on test data — train statistics are always computed
from the first argument.

## Model composables

Build complex models from simple pieces:

```python
from unfolds import NLModel, EnsembleModel, StackedModel, RoutedModel

# Ensemble: same architecture, different seeds
ensemble = EnsembleModel(base=NLModel((8,)), n_seeds=5)

# Stacking: stage 1 predictions augment stage 2 input
stacked = StackedModel(
    first=NLModel((8,)),
    second=NLModel((4,)),
    augment=True,
    oof_folds=5,          # inner OOF prevents stacking leakage
)

# Routing: partition input space, specialize per region
routed = RoutedModel(
    router=NLModel((4,)),
    experts={'low': NLModel((4,)), 'high': NLModel((8,))},
    route_fn=lambda pred: 'low' if pred < 50 else 'high',
)
```

All implement `fit(X, y)` / `predict(X)` / `clone()`. Composable with
the experiment framework — pass any of these as a model factory.

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

**No automatic logging.** The framework records results when you
explicitly call `fold.record()`. No hidden side effects, no surprise
files, no "where did my metrics go?"

**No distributed computing.** Single-machine, single-process. The
framework runs experiments sequentially because reproducibility
matters more than speed. If you need parallelism, parallelize at
the experiment level (different seeds), not inside.

**Strict over convenient.** `final_evaluate()` could silently allow
multiple calls. `fold.X_train` could be writable. `SanctifiedDataset`
could expose the sanctified data. These would all be more convenient
and all undermine the guarantees.

## Dependencies

- numpy ≥ 1.21
- scipy ≥ 1.7
- scikit-learn ≥ 1.0

No deep learning frameworks. The `NLModel` is a from-scratch sigmoid
network in pure numpy — small, fast, no GPU needed.

## License

CC BY 4.0
