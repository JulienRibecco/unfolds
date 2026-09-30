# Real DrugComb data: an optional validation audit

The [quick synthetic demo](../entity_validation.py) takes seconds and needs
no download. This companion uses the real DrugComb/GDSC2 sources behind the
[case study](../../docs/drugcomb-case-study.md).

There are two distinct artifacts:

- **New runnable audit:** a controlled comparison of inner split strategies,
  with ridge selecting between drug descriptors and drug descriptors plus
  genes. All fitted preprocessing uses training data only.
- **[Original research archive](historical/README.md):** the notes, preparation
  script, and grouped Soup experiment for inspection. Its feature search,
  preprocessing, and historical scores differ from the new audit.

The new audit requires no `signalfault` checkout or `potage` installation.
It is a reproduction of the evaluation question on real data, not an exact
replication of the original Soup result.

## Install and prepare

Run from the repository root. Python 3.10 is the tested optional environment.
Use a virtual environment so the optional dependencies stay separate:

```bash
python3.10 -m venv .venv-drugcomb
source .venv-drugcomb/bin/activate
python -m pip install -e .
python -m pip install -r examples/drugcomb/requirements.txt
python -m examples.drugcomb.prepare --download
```

Preparation downloads three pinned files from Harvard Dataverse, using the
file IDs in [TDC's registry](https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py):

| File | File ID | Size |
|---|---:|---:|
| DrugComb measurements | 4215720 | 34.7 MiB |
| GDSC2 expression/response data | 4165727 | 111.2 MiB |
| GDSC gene symbols | 5255026 | 0.14 MiB |

The downloader verifies SHA-256 before reading upstream pickle files. It
reuses verified files and never downloads during normal package import or tests.
Source URLs, checksums, versions, cell-line matching, and exclusions are
recorded in `data/drugcomb/prepared.json`. Dataset attribution and terms remain
those of the upstream providers; data files are not distributed in this repo.

If you already have these files in a TDC cache:

```bash
python -m examples.drugcomb.prepare --raw-dir /path/to/tdc-cache
```

Preparation retains 246,776 measurements across 49 matched cell lines and
all 17,737 expression features. It stores molecular descriptors once per drug
and expression once per cell line: the prepared archive is about **8.6 MiB**.
It does not select genes, impute values, normalize, or remove constant columns.
RDKit molecular descriptors are deterministic per-molecule computations.

## Run a small real-data audit

```bash
OPENBLAS_NUM_THREADS=1 python -m examples.drugcomb.audit \
  --max-rows 5000 --output artifacts/drugcomb-small.json
```

This limits model training to a deterministic 5,000-row subset; the upstream
download and preparation still use the full sources. The full run is:

```bash
OPENBLAS_NUM_THREADS=1 python -m examples.drugcomb.audit \
  --max-rows 0 --output artifacts/drugcomb-full.json
```

`OPENBLAS_NUM_THREADS=1` limits BLAS parallelism where supported; it is optional.
Both commands use seed 42, three inner folds, ridge alpha 10, and up to 200
training-selected genes. Use `--help` to inspect the available settings.
Changing seeds/settings after inspecting test scores turns that test set
into development feedback; these commands do not prevent such reuse.

## What the protocol holds fixed

1. Reserve 20% of cell lines for the outer test set, before model selection.
2. On the remaining rows, compare the same two candidates with either random
   row folds or folds that keep entire cell lines together.
3. In **each training fold**, rank genes by variance across distinct training
   cell lines. Fit finite/constant-column filtering, median imputation, and
   normalization only on training rows. The random-row protocol deliberately
   permits shared cell identities; the grouped protocol does not.
4. Choose each protocol's candidate using pooled out-of-fold R². Both choices
   are fixed before test evaluation. Refit on the full development set,
   recomputing gene selection there, and score on the same unseen cell lines.

This asks about **new cell lines**, not new drugs. Drugs and drug pairs may
appear on both sides. Metrics give every measurement row equal weight;
they are not cell-line-macro averages or estimates of clinical efficacy.

## Observed results

On the full prepared dataset (201,284 development rows, 45,492 test rows;
40 development cell lines, 9 test cell lines):

| Inner validation | Selected candidate | Validation R² | Final test R² | Final test MAE |
|---|---|---:|---:|---:|
| Random rows | Drug descriptors + genes | 0.1495 | 0.0853 | 3.6741 |
| Grouped cell lines | Drug descriptors | 0.1300 | 0.0865 | 3.6694 |

Within grouped development CV, adding genes scored 0.1222 versus 0.1300 for
drug descriptors alone. Random-row CV preferred adding genes, 0.1495 versus
0.1391. **The selection decision changes; the final test difference is small.**
One split does not establish statistical significance or general superiority.

On the 5,000-row subset, both procedures selected drug descriptors alone,
with the same test R² of -0.0171. The smaller audit exercises the workflow;
it does not guarantee the same selection or scores as the full dataset.

Reference environment: Python 3.10.2, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2, pandas 2.2.3, RDKit 2024.9.6, macOS ARM64.
Preparation took about 28 seconds from cached source files. The final
reference runs measured:

| Run | Audit time | Peak process memory |
|---|---:|---:|
| Synthetic illustration | 0.07 s | 135 MiB |
| DrugComb, 5,000 rows | 1.02 s | 485 MiB |
| DrugComb, all 246,776 rows | 54.41 s | 3,349 MiB (3.27 GiB) |

Audit time excludes interpreter startup and data loading; download time is
additional. Memory is the process high-water mark, including loading.
These are measurements from one machine, not resource guarantees. The matrix
expansion and ridge fitting need substantially more RAM than the compact
file's size. Use the small audit first on limited hardware. Reports include
peak process RSS on macOS/Linux for measuring your own run.

## Inspect and reproduce a run

Each run writes two files:

- **JSON:** all candidate CV scores, chosen candidates, final metrics and
  feature names, sample/group counts, versions, source hashes, code hashes,
  elapsed time, and peak process RSS where available.
- **NPZ:** development/test row indices, inner validation-fold assignments,
  out-of-fold predictions, final predictions, test targets, entity IDs,
  and mappings back to prepared and upstream source rows.

Load the NPZ with `numpy.load(path, allow_pickle=False)`. It provides enough
information to inspect overlap and recompute reported scores. Generated
data and results are ignored by Git. Checked-in [reference reports](reference/)
contain aggregate results and feature names; the measurement-level arrays
are regenerated locally.

For an exact environment match, install these core versions in the optional
Python 3.10 environment before running:

```bash
python -m pip install numpy==2.2.6 scipy==1.15.3 scikit-learn==1.7.2
```

Small floating-point differences across numerical libraries/platforms are
possible. A checksum mismatch means the input differs from the pinned
release; do not silently substitute it and compare against these results.

## Sources

- [TDC DrugComb description](https://tdcommons.ai/multi_pred_tasks/drugsyn/#drugcomb)
- [TDC drug-response datasets](https://tdcommons.ai/multi_pred_tasks/drugres/)
- [DrugComb publication](https://doi.org/10.1093/nar/gkz337)
- [GDSC](https://www.cancerrxgene.org/)
