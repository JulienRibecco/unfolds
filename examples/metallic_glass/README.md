# Metallic glass: from a chemistry rule to a repeatable comparison

Predict glass-transition temperature **Tg, in kelvin**, from alloy composition.
This example packages a small materials investigation into a short
[benchmark](benchmark.py): one loader, four model factories, and one routing
rule. It demonstrates the convenience of research tooling built around
ordinary Python and sklearn models.

## Run it

From the repository root, after `pip install -e .`:

```bash
pip install -r examples/metallic_glass/requirements.txt
python -m examples.metallic_glass.prepare
python -m examples.metallic_glass.benchmark --quick
```

Preparation downloads **2.8 MB**, verifies the archive and workbook checksums,
and caches prepared data under `data/metallic-glass/`. Subsequent benchmark runs
need no network, pandas, or Excel reader. To prepare offline, supply the pinned
archive with `--archive /path/to/descriptors.zip`. Both commands accept
`--data-dir /path/to/cache`.

```bash
# Five folds and a longer NN training budget.
python -m examples.metallic_glass.benchmark

# Use the built-in experiment selector.
python -m examples.metallic_glass.benchmark --quick --exp ridge
```

Quick mode uses three folds and 300 NN epochs; full mode uses five folds and
3,000 epochs. Both use seed 42 and three networks per ensemble. The tree and
ridge settings stay fixed. Quick mode is a workflow check with undertrained
neural networks, not a quality benchmark.

## What the framework takes care of

The domain-specific rule is deliberately small:

```python
def chemistry_route(X, _):
    return np.where(X[:, -1] > 30, "fe_co_ni_rich", "other")

def chemistry_cascade(ctx):
    return RoutedModel(
        router=None, route_fn=chemistry_route,
        experts={"fe_co_ni_rich": flat_ensemble(ctx), "other": flat_ensemble(ctx)},
    )
```

The last feature is the combined atomic percentage of Fe, Co, and Ni. This
fixed rule routes directly from composition; each expert trains only on its
training-fold subset. Each expert has the same `(8, 8)` architecture, seed
schedule, and epoch budget as the flat ensemble. The routed system contains
twice as many networks in total, so this is not a parameter-matched comparison.

Registering the factories supplies the experiment workflow:

```python
from examples.metallic_glass.benchmark import (
    load_glass, ridge, flat_ensemble, chemistry_cascade, boosted_trees,
)
from unfolds import ExperimentConfig, Research

research = Research(
    "Metallic-glass Tg (K)", load_glass,
    ExperimentConfig(seed=42, k=5, sanctified_fraction=0.15),
    save_dir="artifacts/metallic-glass",
)
for name, factory in [
    ("ridge", ridge), ("flat_ensemble", flat_ensemble),
    ("chemistry_cascade", chemistry_cascade), ("boosted_trees", boosted_trees),
]:
    research.new_experiment(name, factory)
research.run(quick=True)
```

`Research` creates fresh models on shared folds, trains final models on the
development pool, evaluates them together on the shared holdout, prints a
comparison, and saves history and snapshots. Neural models normalize their
training inputs internally; ridge uses a sklearn scaler fitted inside each
fold. Adding another ordinary `fit`/`predict` model means registering another
factory.

Results go to `artifacts/metallic-glass/`:

- `runs.json`: appended run settings, scores, and timing.
- `snapshots/`: the latest fitted models and OOF/holdout predictions for each
  experiment. A later run replaces those snapshots.

The separate `data/metallic-glass/manifest.json` records the source revision,
input and prepared-data hashes, preparation-code hash, and preparation versions.

For notebook use and a different output location:

```python
from examples.metallic_glass.benchmark import make_research

research = make_research(save_dir="artifacts/my-glass-run")
research.run(quick=True)
snapshot = research.load_snapshot("chemistry_cascade")
model = snapshot["model"]
# model.predict(X_new) expects the same 82 prepared descriptors.
```

## Data and scope

The two source workbooks are from
[ZHOU-Ziqing/ML_Metallicglass_GFA](https://github.com/ZHOU-Ziqing/ML_Metallicglass_GFA/tree/54bceddee4980217ea55998da0ee56272a81b1bf),
the authors' repository for
[Rational design of chemically complex metallic glasses by hybrid modeling
guided machine learning](https://www.nature.com/articles/s41524-021-00607-4).
This example uses their composition and property tables for a Tg experiment;
it does not reproduce that paper's models or results. The older local notes
attributed these workbooks to `GAN_BMG`; byte-for-byte verification identifies
`ML_Metallicglass_GFA/descriptors.zip` as their source.

The pinned workbook has 9,894 rows, of which **843 contain Tg**. Preparation
normalizes composition fractions as exact rational numbers and sorts elements,
then merges repeated compositions before any split. This leaves **838 unique
compositions**; all merged measurements agree on Tg. Averaging repeated Tg
measurements is the explicit policy if the preparation function receives
replicates with different values.

The 82 inputs comprise element count, ten statistics for each of eight
elemental properties, and the Fe/Co/Ni percentage. Feature calculation uses
composition and a fixed property lookup only; other measured temperatures,
publication identifiers, and target-derived features are excluded. To retain
the original investigation's descriptor recipe, missing property cells become
zero and the source carbon atomic-mass value of 811 is corrected to 12.011.
The weighted entropy feature uses the original fraction-scaled definition;
it is not entropy of a renormalized weighted distribution. See
[prepare.py](prepare.py) for the complete recipe.

The random, target-stratified split leaves **717 development compositions and
121 holdout compositions**. It evaluates new compositions drawn from the
represented mixture of alloy families, not transfer to unseen families or
independent laboratories. Related alloys can remain similar across splits.
The routing threshold comes from earlier exploration of this dataset; these
results illustrate a workflow rather than independently confirm that rule.
Do not tune against the displayed holdout scores or treat repeated CLI runs
as fresh test sets.

Source workbooks and prepared rows are not bundled in this repository. The
upstream repository does not declare a data license; this repository's code
license does not grant rights to the upstream data.

## Reference run

The checked-in [reference.json](reference.json) records a full run, including
source hashes, software versions, settings, and all four models' scores.
On the reference machine, quick mode took about **7 seconds** and the full
comparison about **31 seconds**, using one BLAS thread. Runtime depends on
hardware and numerical-library settings.

| Model | Dev MAE, mean ± fold SD (K) | Holdout MAE (K) |
|---|---:|---:|
| Ridge | 26.35 ± 0.98 | 28.12 |
| Flat neural ensemble | 18.81 ± 3.03 | 20.46 |
| Chemistry-routed ensemble | 15.94 ± 1.98 | 15.27 |
| Boosted trees | 13.17 ± 1.62 | 14.46 |

Routing improves on this flat ensemble, while boosted trees do better still.
The practical result is a compact workflow for trying a domain hypothesis
alongside conventional baselines, with the same evaluation and reporting.

The comparison is illustrative, with no hyperparameter search or claim of
state-of-the-art performance. Earlier research scores used different
preparation and neural-network code and are not carried over here.
