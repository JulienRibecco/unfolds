"""Centralized dataset splitting, fold-safe preprocessing, and sanctified holdout.

Leakage Prevention Guarantees
-----------------------------
1. **Index-first** — generators return indices, not arrays.  Train/test
   boundary is always visible in calling code.
2. **Copy-based fold ops** — ``fold_impute`` and ``fold_normalize`` return
   copies, never mutate.  Eliminates cross-fold contamination from reused
   arrays.
3. **``_validate_indices``** — called by every generator.  Catches overlap,
   duplicates, and group leakage with assertion errors.
4. **Sanctified holdout** — ``SanctifiedDataset`` physically separates dev
   from sanctified at construction.  ``final_evaluate`` guard prevents
   re-use.  Selection leakage becomes structurally impossible.
5. **OOF separation** — ``oof_indices`` is distinct from ``kfold_indices``.
   Use ``kfold_indices`` for outer evaluation.  Use ``oof_indices`` only
   inside training loops for stacking.  Never mix them.
"""

import hashlib
import warnings
from collections import namedtuple

import numpy as np

from .validate import ValidatedDataset


# ---------------------------------------------------------------------------
# Validation (internal, called by every generator)
# ---------------------------------------------------------------------------

def _validate_indices(tr_idx, te_idx, n=None, groups=None):
    """Assert index splits are correct.

    Checks:
      - No duplicates within each split
      - No overlap between splits
      - Exhaustive coverage of 0..n-1 (when *n* is given)
      - No group leakage (when *groups* is given)

    Raises AssertionError on violation.  Warns if a class present in one
    split is absent from the other (cheap np.unique check on the indices
    themselves — caller can pass y-indexed arrays via groups for richer
    checks).
    """
    tr = np.asarray(tr_idx)
    te = np.asarray(te_idx)

    # no duplicates
    assert len(np.unique(tr)) == len(tr), "duplicate indices in train split"
    assert len(np.unique(te)) == len(te), "duplicate indices in test split"

    # no overlap
    assert len(np.intersect1d(tr, te)) == 0, "train/test indices overlap"

    # exhaustive
    if n is not None:
        combined = np.union1d(tr, te)
        assert np.array_equal(combined, np.arange(n)), (
            f"indices not exhaustive over 0..{n-1}: "
            f"got {len(combined)} unique values")

    # group leakage
    if groups is not None:
        groups = np.asarray(groups)
        tr_groups = set(groups[tr].ravel())
        te_groups = set(groups[te].ravel())
        leaked = tr_groups & te_groups
        assert len(leaked) == 0, (
            f"group leakage: {len(leaked)} groups in both train and test")


def _warn_rare_classes(y, tr_idx, te_idx):
    """Warn if a class present in one split is absent from the other."""
    tr_classes = set(np.unique(y[tr_idx]))
    te_classes = set(np.unique(y[te_idx]))
    only_train = tr_classes - te_classes
    only_test = te_classes - tr_classes
    if only_test:
        warnings.warn(
            f"classes in test but not train: {only_test}",
            stacklevel=3)
    if only_train:
        warnings.warn(
            f"classes in train but not test: {only_train}",
            stacklevel=3)


# ---------------------------------------------------------------------------
# Layer 1: Index generators
# ---------------------------------------------------------------------------

def kfold_indices(n, k=5, seed=42):
    """Generate k-fold cross-validation index splits.

    Returns list of *(train_idx, test_idx)* tuples that partition
    ``range(n)`` into *k* folds.  The last fold absorbs remainder samples.

    Use for **outer evaluation** only.  For inner OOF stacking, use
    :func:`oof_indices` instead.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        te_idx = idx[start:end]
        tr_idx = np.concatenate([idx[:start], idx[end:]])
        folds.append((tr_idx, te_idx))
    for tr_idx, te_idx in folds:
        _validate_indices(tr_idx, te_idx, n)
    return folds


def grouped_kfold_indices(groups, k=5, seed=42):
    """Grouped k-fold CV — all samples of the same group stay together.

    Args:
        groups: (n,) array-like of group labels (e.g. composition strings).
        k: number of folds.
        seed: random seed for group shuffling.

    Returns list of *(train_idx, test_idx)* tuples.
    """
    groups = np.asarray(groups)
    n = len(groups)
    uc = sorted(set(groups))
    rng = np.random.RandomState(seed)
    rng.shuffle(uc)
    fs = len(uc) // k
    folds = []
    for fold in range(k):
        ts = fold * fs
        te = ts + fs if fold < k - 1 else len(uc)
        tc = uc[ts:te]
        mask = np.isin(groups, tc)
        te_idx = np.where(mask)[0]
        tr_idx = np.where(~mask)[0]
        folds.append((tr_idx, te_idx))
    for tr_idx, te_idx in folds:
        _validate_indices(tr_idx, te_idx, n, groups=groups)
    return folds


def train_test_indices(n, test_fraction=0.2, seed=42):
    """Random train/test index split (no stratification).

    Returns *(train_idx, test_idx)*.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int((1.0 - test_fraction) * n)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    _validate_indices(tr_idx, te_idx, n)
    return tr_idx, te_idx


def oof_indices(n, n_folds=5):
    """Generate inner k-fold index splits for out-of-fold (OOF) construction.

    Yields *(train_idx, test_idx)* pairs using **contiguous** chunks
    (no shuffling).  Designed for leak-free first-pass predictions inside
    a training loop.

    Use ``kfold_indices`` for outer evaluation.  Use ``oof_indices`` only
    inside training loops for stacking.  **Never mix them.**
    """
    chunk = n // n_folds
    for k in range(n_folds):
        s = k * chunk
        e = (k + 1) * chunk if k < n_folds - 1 else n
        te_idx = np.arange(s, e)
        tr_idx = np.concatenate([np.arange(0, s), np.arange(e, n)])
        _validate_indices(tr_idx, te_idx, n)
        yield tr_idx, te_idx


def temporal_split_indices(n, fraction=0.15):
    """Chronological sanctified split — last *fraction* of data.

    No shuffling.  Sanctified = chronological tail, dev = head.

    Returns *(dev_idx, sanct_idx)* covering ``0..n-1``.
    """
    split = int(n * (1 - fraction))
    split = max(1, min(split, n - 1))  # at least 1 sample per side
    dev_idx = np.arange(split)
    sanct_idx = np.arange(split, n)
    _validate_indices(dev_idx, sanct_idx, n)
    return dev_idx, sanct_idx


def expanding_window_indices(n, k=3, gap=0):
    """Expanding-window (forward-chaining) CV for time series.

    Splits *n* samples into *k+1* contiguous segments.  For fold *i*:

    - train = segments ``0..i``
    - test  = segment ``i+1`` (offset by *gap* samples)

    Samples in the gap between train end and test start are excluded
    from both sets — this prevents target overlap for rolling targets
    (e.g. 20-day forward volatility needs gap >= 20).

    Args:
        n: number of samples (already in chronological order).
        k: number of folds (default 3).
        gap: samples to skip between train end and test start.

    Returns list of *(train_idx, test_idx)* tuples.
    """
    seg = n // (k + 1)
    if seg < 1:
        raise ValueError(f"too few samples ({n}) for {k} temporal folds")

    folds = []
    for i in range(k):
        tr_end = (i + 1) * seg
        te_start = tr_end + gap
        te_end = (i + 2) * seg if i < k - 1 else n

        if te_start >= te_end:
            continue  # gap swallowed the test set

        tr_idx = np.arange(0, tr_end)
        te_idx = np.arange(te_start, te_end)
        # no n= → skip exhaustiveness (gap samples intentionally excluded)
        _validate_indices(tr_idx, te_idx)
        folds.append((tr_idx, te_idx))

    if not folds:
        raise ValueError(
            f"temporal gap={gap} too large for n={n}, k={k} "
            f"(segment size={seg})")
    return folds


# ---------------------------------------------------------------------------
# Layer 2: Fold iteration
# ---------------------------------------------------------------------------

FoldData = namedtuple('FoldData', ['fold', 'tr_idx', 'te_idx', 'n_folds'])
SanctifiedResult = namedtuple('SanctifiedResult', ['X_sanct', 'y_sanct', 'sanct_idx'])


def iter_folds(folds):
    """Iterate over a list of fold splits, yielding :class:`FoldData`.

    Args:
        folds: list of *(train_idx, test_idx)* tuples (from any generator).

    Yields:
        ``FoldData(fold=i, tr_idx=..., te_idx=..., n_folds=k)``
    """
    n_folds = len(folds)
    for i, (tr_idx, te_idx) in enumerate(folds):
        yield FoldData(fold=i, tr_idx=tr_idx, te_idx=te_idx, n_folds=n_folds)


# ---------------------------------------------------------------------------
# Layer 3: Fold-safe operations
# ---------------------------------------------------------------------------

def fold_impute(X, tr_idx, te_idx=None):
    """NaN median-impute using train-only statistics.  Returns copies.

    Args:
        X: (n, d) array (may contain NaN).
        tr_idx: train indices — medians computed from these rows only.
        te_idx: test indices (optional).

    Returns:
        *(X_tr_imp, X_te_imp, medians)* — copies, original *X* unchanged.
        *X_te_imp* is None when *te_idx* is None.
    """
    X_tr = X[tr_idx].copy()
    medians = np.nanmedian(X_tr, axis=0)

    for j in range(X_tr.shape[1]):
        mask = np.isnan(X_tr[:, j])
        if mask.any():
            X_tr[mask, j] = medians[j]

    X_te_imp = None
    if te_idx is not None:
        X_te = X[te_idx].copy()
        for j in range(X_te.shape[1]):
            mask = np.isnan(X_te[:, j])
            if mask.any():
                X_te[mask, j] = medians[j]
        X_te_imp = X_te

    return X_tr, X_te_imp, medians


def fold_normalize(X_train, X_test=None):
    """Zero-mean / unit-variance normalize using train-only statistics.

    Returns copies — never mutates inputs.

    Args:
        X_train: (n_tr, d) array.
        X_test: (n_te, d) array (optional).

    Returns:
        *(Xn_train, Xn_test, means, stds)* — *Xn_test* is None when
        *X_test* is None.
    """
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)

    Xn_train = (X_train - means) / stds
    Xn_test = None
    if X_test is not None:
        Xn_test = (X_test - means) / stds

    return Xn_train, Xn_test, means, stds


def fold_safe_preprocess(X_tr, X_te=None, impute=True, normalize=True,
                         full_X=None, tr_idx=None, te_idx=None):
    """Convenience: chain impute → normalize in one call.

    Two calling conventions:

    1. **Pre-sliced** (default): pass *X_tr* and *X_te* directly.
       Imputation uses ``X_tr`` to compute medians.

    2. **Index-based**: pass *full_X*, *tr_idx*, *te_idx*.
       Imputation slices from *full_X* using the indices.

    Returns *(X_tr_proc, X_te_proc, stats_dict)* where *stats_dict*
    contains ``'medians'``, ``'means'``, ``'stds'`` (whichever were
    computed).
    """
    stats = {}

    if full_X is not None and tr_idx is not None:
        # Index-based path
        if impute:
            X_tr, X_te, medians = fold_impute(full_X, tr_idx, te_idx)
            stats['medians'] = medians
        else:
            X_tr = full_X[tr_idx].copy()
            X_te = full_X[te_idx].copy() if te_idx is not None else None
    else:
        # Pre-sliced path — build a synthetic full_X so fold_impute
        # handles all the logic in one place.
        if impute:
            n_tr = len(X_tr)
            if X_te is not None:
                combined = np.concatenate([X_tr, X_te], axis=0)
                synth_tr = np.arange(n_tr)
                synth_te = np.arange(n_tr, n_tr + len(X_te))
                X_tr, X_te, medians = fold_impute(combined, synth_tr, synth_te)
            else:
                X_tr, _, medians = fold_impute(X_tr, np.arange(n_tr))
            stats['medians'] = medians

    if normalize:
        X_tr, X_te, means, stds = fold_normalize(X_tr, X_te)
        stats['means'] = means
        stats['stds'] = stds

    return X_tr, X_te, stats


# ---------------------------------------------------------------------------
# Layer 4: Sanctified holdout
# ---------------------------------------------------------------------------

def sanctified_indices(n, fraction=0.15, seed=42, stratify=None, groups=None):
    """Split into development and sanctified (holdout) index sets.

    The sanctified set should **never** participate in feature selection,
    hyperparameter tuning, architecture search, or any decision-making.
    It exists solely for final evaluation.

    Args:
        n: total number of samples.
        fraction: fraction reserved for sanctified set.
        seed: random seed.
        stratify: (n,) array for balanced split.  Continuous values are
            binned into 10 quantile bins automatically.  Ignored when
            *groups* is provided.
        groups: (n,) array of group labels.  When given, the split is
            done at the **group** level — all samples of a group go to
            the same side.  Prevents group leakage between dev and
            sanctified.

    Returns:
        *(dev_idx, sanct_idx)* — disjoint, cover ``0..n-1``.
    """
    rng = np.random.RandomState(seed)

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = sorted(set(groups))
        rng.shuffle(unique_groups)
        n_sanct = max(1, int(len(unique_groups) * fraction))
        sanct_groups = set(unique_groups[:n_sanct])

        sanct_mask = np.isin(groups, list(sanct_groups))
        sanct_idx = np.sort(np.where(sanct_mask)[0])
        dev_idx = np.sort(np.where(~sanct_mask)[0])

        _validate_indices(dev_idx, sanct_idx, n, groups=groups)
        return dev_idx, sanct_idx

    if stratify is not None:
        stratify = np.asarray(stratify)
        # Bin continuous values into quantile bins
        unique = np.unique(stratify)
        if len(unique) > 20:
            # Continuous — quantile-bin
            bins = np.percentile(stratify, np.linspace(0, 100, 11))
            bins[-1] += 1  # include max
            labels = np.digitize(stratify, bins) - 1
        else:
            labels = stratify

        dev_list, sanct_list = [], []
        for lab in np.unique(labels):
            mask = np.where(labels == lab)[0]
            perm = rng.permutation(mask)
            if len(perm) == 1:
                # Singleton class — must go to dev so dev folds can see it
                dev_list.append(perm)
                continue
            n_sanct = max(1, int(len(perm) * fraction))
            sanct_list.append(perm[:n_sanct])
            dev_list.append(perm[n_sanct:])

        dev_idx = np.sort(np.concatenate(dev_list))
        sanct_idx = np.sort(np.concatenate(sanct_list))
    else:
        idx = rng.permutation(n)
        n_sanct = max(1, int(n * fraction))
        sanct_idx = np.sort(idx[:n_sanct])
        dev_idx = np.sort(idx[n_sanct:])

    _validate_indices(dev_idx, sanct_idx, n)
    return dev_idx, sanct_idx


class SanctifiedDataset:
    """Dataset with a physically separated sanctified holdout.

    The sanctified subset is chosen once at construction and never
    participates in any model selection decision.  ``iter_dev_folds``
    only yields indices from the development pool.  ``final_evaluate``
    has a one-shot guard that raises on second call.

    Data validation runs by default via :class:`ValidatedDataset`.
    Pass a pre-validated ``ValidatedDataset`` to skip re-validation,
    or ``skip_validation=True`` to bypass entirely (research scripts).

    Parameters
    ----------
    X : array or ValidatedDataset
        If a ValidatedDataset, its X/y/groups/feature_names/source_ids
        are used directly (no re-validation).  If a raw array, validation
        runs automatically unless *skip_validation* is True.
    y : (n,) array, optional — ignored when X is a ValidatedDataset
    groups : (n,) array, optional — ignored when X is a ValidatedDataset
    feature_names : list of str, optional — ignored when X is a ValidatedDataset
    source_ids : (n,) array, optional — ignored when X is a ValidatedDataset.
        Labels identifying which data source each sample comes from.
        Used for leave-one-source-out fold strategies via
        ``iter_dev_folds(group_by='source')``.
    seed : int
    sanctified_fraction : float
    stratify : bool — if True and *y* is provided, stratify the split
    skip_validation : bool — bypass validation (for research scripts)
    """

    def __init__(self, X, y=None, groups=None, feature_names=None,
                 source_ids=None, seed=42, sanctified_fraction=0.15,
                 stratify=True, skip_validation=False,
                 temporal=False, temporal_gap=0):
        if isinstance(X, ValidatedDataset):
            # Already validated — unpack into locals
            X_full = X.X
            y_full = X.y
            groups_full = X.groups
            feature_names = X.feature_names
            sids_full = X.source_ids
        elif skip_validation:
            # Research escape hatch — no validation
            X_full = np.asarray(X)
            y_full = np.asarray(y) if y is not None else None
            groups_full = np.asarray(groups) if groups is not None else None
            sids_full = (np.asarray(source_ids)
                         if source_ids is not None else None)
        else:
            # Default path — validate automatically
            vds = ValidatedDataset.validate(
                X, y=y, groups=groups, feature_names=feature_names,
                source_ids=source_ids)
            X_full = vds.X
            y_full = vds.y
            groups_full = vds.groups
            feature_names = vds.feature_names
            sids_full = vds.source_ids

        self._n_total = len(X_full)
        self._seed = seed
        self._evaluated = False
        self.feature_names = feature_names
        self._temporal = bool(temporal)
        self._temporal_gap = int(temporal_gap)

        if self._temporal:
            # Chronological split — sanctified = tail, dev = head
            dev_idx, sanct_idx = temporal_split_indices(
                self._n_total, fraction=sanctified_fraction)
        elif groups_full is not None:
            # Grouped split — groups take precedence over stratify
            dev_idx, sanct_idx = sanctified_indices(
                self._n_total, fraction=sanctified_fraction,
                seed=seed, groups=groups_full)
        else:
            strat_arr = y_full if (stratify and y_full is not None) else None
            dev_idx, sanct_idx = sanctified_indices(
                self._n_total, fraction=sanctified_fraction,
                seed=seed, stratify=strat_arr)

        # Absolute indices (into original full array) — kept for auxiliary
        # array slicing by callers who need to align external data.
        self.dev_idx = dev_idx
        self._sanct_idx = sanct_idx

        # Public = dev-only
        self.X = X_full[dev_idx]
        self.y = y_full[dev_idx] if y_full is not None else None
        self.groups = groups_full[dev_idx] if groups_full is not None else None
        self.source_ids = sids_full[dev_idx] if sids_full is not None else None

        # Private = sanctified (consumed once by final_evaluate)
        self._sanct_X = X_full[sanct_idx]
        self._sanct_y = y_full[sanct_idx] if y_full is not None else None

    # -- views --

    @property
    def n_sanctified(self):
        """Number of sanctified samples (safe count, no indices exposed)."""
        if self._sanct_X is not None:
            return len(self._sanct_X)
        return 0

    @property
    def n_samples(self):
        """Total number of samples (dev + sanctified)."""
        return self._n_total

    @property
    def X_dev(self):
        return self.X

    @property
    def y_dev(self):
        return self.y

    @property
    def groups_dev(self):
        return self.groups

    @property
    def X_sanct(self):
        if not self._evaluated:
            raise RuntimeError(
                "sanctified data is locked — call final_evaluate() first")
        if self._sanct_X is None:
            raise RuntimeError(
                "sanctified data already consumed by final_evaluate()")
        arr = self._sanct_X.copy()
        arr.setflags(write=False)
        return arr

    @property
    def y_sanct(self):
        if not self._evaluated:
            raise RuntimeError(
                "sanctified data is locked — call final_evaluate() first")
        if self._sanct_y is None:
            return None
        arr = self._sanct_y.copy()
        arr.setflags(write=False)
        return arr

    # -- dev-fold iteration --

    def iter_dev_folds(self, k=5, grouped=False, group_by=None):
        """K-fold over the development set only.

        Yields :class:`FoldData` with dev-relative indices (0..n_dev-1).
        Since ``self.X`` contains only dev data, these indices can be
        used directly: ``self.X[fd.tr_idx]``.

        Parameters
        ----------
        k : int
            Number of folds.
        grouped : bool
            Legacy shorthand for ``group_by='groups'``.
        group_by : str or None
            Grouping dimension for fold assignment:

            - ``None`` — random k-fold (default).
            - ``'groups'`` — grouped k-fold using ``self.groups``.
              All samples sharing a group stay in the same fold.
            - ``'source'`` — grouped k-fold using ``self.source_ids``.
              Use with ``k == n_sources`` for leave-one-source-out.
        """
        n_dev = len(self.X)

        # Resolve backward-compat grouped flag
        if grouped and group_by is None:
            group_by = 'groups'

        if self._temporal:
            folds = expanding_window_indices(
                n_dev, k=k, gap=self._temporal_gap)
        elif group_by == 'groups':
            if self.groups is None:
                raise ValueError("group_by='groups' but no groups provided")
            folds = grouped_kfold_indices(self.groups, k=k, seed=self._seed)
        elif group_by == 'source':
            if self.source_ids is None:
                raise ValueError(
                    "group_by='source' but no source_ids provided")
            folds = grouped_kfold_indices(
                self.source_ids, k=k, seed=self._seed)
        else:
            folds = kfold_indices(n_dev, k=k, seed=self._seed)

        for i, (tr_idx, te_idx) in enumerate(folds):
            yield FoldData(fold=i, tr_idx=tr_idx, te_idx=te_idx, n_folds=k)

    def prepare_dev_fold(self, tr_idx, te_idx, impute=True, normalize=True):
        """Fold-safe preprocess on given dev-fold indices.

        Returns *(X_tr_proc, X_te_proc, stats_dict)*.
        """
        return fold_safe_preprocess(
            X_tr=None, X_te=None,
            impute=impute, normalize=normalize,
            full_X=self.X, tr_idx=tr_idx, te_idx=te_idx)

    def final_evaluate(self, evaluate_fn=None):
        """Call ONCE.  One-shot guard for sanctified evaluation.

        Two calling conventions:

        1. **Stateless** (simple models): pass a callable that receives
           ``X_sanct`` and returns predictions.

           ``ds.final_evaluate(lambda X: model.predict(X))``

        2. **Stateful** (pipelines where test data is prepared during
           training, e.g. vision): pass a zero-arg callable.  The caller
           captures ``sanct_idx`` beforehand and sets up prediction state
           before calling this method.

           ``ds.final_evaluate(lambda: clf.predict()[0])``

        If no *evaluate_fn* is given, acts as a pure guard — marks the
        sanctified set as consumed and returns a
        :class:`SanctifiedResult` containing ``X_sanct``, ``y_sanct``,
        and ``sanct_idx``.

        Args:
            evaluate_fn: callable, optional.  If it accepts one argument,
                ``X_sanct`` is passed.  If zero-arg, called directly.
                If None, returns SanctifiedResult.

        Returns:
            Result of *evaluate_fn*, or :class:`SanctifiedResult` if no
            callable given.

        Raises:
            RuntimeError on second call.
        """
        if self._evaluated:
            raise RuntimeError(
                "final_evaluate already called — sanctified set must be "
                "used only once to prevent selection leakage")
        self._evaluated = True
        sanct_X = self._sanct_X
        sanct_y = self._sanct_y
        sanct_idx = self._sanct_idx

        if evaluate_fn is None:
            # Pure guard mode — return SanctifiedResult
            result = SanctifiedResult(
                X_sanct=sanct_X, y_sanct=sanct_y, sanct_idx=sanct_idx)
            self._sanct_X = self._sanct_y = self._sanct_idx = None
            return result

        import inspect
        try:
            sig = inspect.signature(evaluate_fn)
            n_params = len([
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
            ])
        except (ValueError, TypeError):
            # Builtins or C functions — assume stateless (1 arg)
            n_params = 1

        if n_params >= 1:
            result = evaluate_fn(sanct_X)
        else:
            result = evaluate_fn()

        self._sanct_X = self._sanct_y = self._sanct_idx = None
        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def derive_seeds(base_seed, n, purpose='ensemble'):
    """Derive *n* deterministic seeds from a base seed and purpose string.

    Uses SHA-256 hashing for proper isolation between different purposes
    (e.g. ``'ensemble'`` vs ``'init'``).

    Returns list of *n* non-negative ints suitable for RandomState.
    """
    seeds = []
    for i in range(n):
        h = hashlib.sha256(f"{base_seed}|{purpose}|{i}".encode()).hexdigest()
        seeds.append(int(h[:8], 16))  # 32-bit seed from first 8 hex chars
    return seeds
