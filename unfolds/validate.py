"""Data validation for signalfault.classify modules.

Sentinel-gated ValidatedDataset — can only be constructed via
ValidatedDataset.validate().  Checks for common data quality issues
before they silently degrade model performance: exact duplicates,
degenerate columns, near-duplicate rows, and label conflicts.
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Sentinel (prevents direct construction)
# ---------------------------------------------------------------------------

_SENTINEL = object()


# ---------------------------------------------------------------------------
# Error checks (block construction)
# ---------------------------------------------------------------------------

def _check_exact_duplicates(X):
    """Raise if X contains exact duplicate rows.

    Uses row hashing via tobytes() for O(n) detection.
    """
    seen = {}
    duplicates = []
    for i in range(X.shape[0]):
        key = X[i].tobytes()
        if key in seen:
            duplicates.append((seen[key], i))
        else:
            seen[key] = i
    if duplicates:
        pairs = ', '.join(f'({a}, {b})' for a, b in duplicates[:5])
        n_extra = len(duplicates) - 5
        msg = f"exact duplicate rows found: {pairs}"
        if n_extra > 0:
            msg += f" ... and {n_extra} more"
        raise ValueError(msg)


def dedup_exact(X, y=None, warn_frac=0.10, **extra_arrays):
    """Drop all copies of exact-duplicate rows (drop-both strategy).

    When multiple rows share identical feature vectors, ALL of them are
    removed — no averaging, no keep-first.  This is the safest default
    because duplicates with conflicting labels inject noise.

    Args:
        X: (n, d) array.
        y: (n,) array, optional.
        warn_frac: warn if more than this fraction of rows are dropped.
            Default 0.10 (10%).
        **extra_arrays: additional (n,) arrays to filter in parallel
            (e.g. ``groups=groups, sources=sources``).

    Returns:
        dict with keys ``'X'``, ``'y'`` (if provided), ``'n_dropped'``,
        ``'n_groups'``, plus any extra_arrays keys — all filtered to
        unique-only rows.
    """
    from collections import Counter
    X = np.asarray(X, dtype=np.float64)
    keys = [X[i].tobytes() for i in range(len(X))]
    counts = Counter(keys)
    keep = np.array([counts[k] == 1 for k in keys])

    n_groups = sum(1 for c in counts.values() if c > 1)
    n_dropped = (~keep).sum()

    if n_dropped > 0:
        frac = n_dropped / len(X)
        msg = (f"dedup_exact: {len(X)} → {keep.sum()} "
               f"({n_groups} duplicate groups, {n_dropped} rows dropped)")
        if frac > warn_frac:
            warnings.warn(
                f"{msg} — {frac:.1%} of data removed (>{warn_frac:.0%} "
                f"threshold). Check for data quality issues.")

    result = {
        'X': X[keep],
        'n_dropped': n_dropped,
        'n_groups': n_groups,
    }
    if y is not None:
        result['y'] = np.asarray(y)[keep]
    for name, arr in extra_arrays.items():
        result[name] = np.asarray(arr)[keep]
    return result


def dedup_average(X, y, warn_frac=0.10, **extra_arrays):
    """Merge exact-duplicate feature rows by averaging targets.

    Keeps one copy of each unique feature vector; its y value is the
    mean of all occurrences.  Best when duplicates represent repeated
    measurements rather than data errors.

    Args:
        X: (n, d) array.
        y: (n,) array (required — averaging needs targets).
        warn_frac: warn if more than this fraction of rows are merged.
        **extra_arrays: additional (n,) arrays to filter in parallel.
            For merged groups, keeps the value from the first occurrence.

    Returns:
        dict with keys ``'X'``, ``'y'``, ``'n_merged'`` (number of
        duplicate groups), ``'n_before'``, plus any extra_arrays keys.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    row_map = {}
    for i in range(len(X)):
        key = X[i].tobytes()
        if key not in row_map:
            row_map[key] = []
        row_map[key].append(i)

    unique_idx = []
    y_avg = []
    n_merged = 0
    for indices in row_map.values():
        unique_idx.append(indices[0])
        y_avg.append(float(np.mean(y[indices])))
        if len(indices) > 1:
            n_merged += 1

    order = np.argsort(unique_idx)
    unique_idx = np.array(unique_idx)[order]
    y_avg = np.array(y_avg)[order]

    if n_merged > 0:
        frac = n_merged / len(row_map)
        msg = (f"dedup_average: {len(X)} → {len(unique_idx)} "
               f"({n_merged} duplicate groups merged)")
        if frac > warn_frac:
            warnings.warn(
                f"{msg} — {frac:.1%} of unique rows had duplicates "
                f"(>{warn_frac:.0%} threshold).")

    result = {
        'X': X[unique_idx],
        'y': y_avg,
        'n_merged': n_merged,
        'n_before': len(X),
    }
    for name, arr in extra_arrays.items():
        result[name] = np.asarray(arr)[unique_idx]
    return result


def dedup_keep_first(X, y=None, warn_frac=0.10, **extra_arrays):
    """Keep the first occurrence of each unique feature row.

    Args:
        X: (n, d) array.
        y: (n,) array, optional.
        warn_frac: warn if more than this fraction of rows are dropped.
        **extra_arrays: additional (n,) arrays to filter in parallel.

    Returns:
        dict with keys ``'X'``, ``'y'`` (if provided), ``'n_dropped'``,
        ``'n_before'``, plus any extra_arrays keys.
    """
    X = np.asarray(X, dtype=np.float64)

    seen = set()
    keep = []
    for i in range(len(X)):
        key = X[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep = np.array(keep)

    n_dropped = len(X) - len(keep)
    if n_dropped > 0:
        frac = n_dropped / len(X)
        msg = (f"dedup_keep_first: {len(X)} → {len(keep)} "
               f"({n_dropped} duplicates dropped)")
        if frac > warn_frac:
            warnings.warn(
                f"{msg} — {frac:.1%} of data removed "
                f"(>{warn_frac:.0%} threshold).")

    result = {
        'X': X[keep],
        'n_dropped': n_dropped,
        'n_before': len(X),
    }
    if y is not None:
        result['y'] = np.asarray(y)[keep]
    for name, arr in extra_arrays.items():
        result[name] = np.asarray(arr)[keep]
    return result


def apply_dedup(X, y=None, strategy='average', **extra_arrays):
    """Apply a dedup strategy and return cleaned arrays.

    Central dispatch for all dedup strategies — called by domain loaders
    with ``config.dedup``.

    Args:
        X: (n, d) array.
        y: (n,) array, optional.
        strategy: ``'average'``, ``'keep-first'``, ``'exact'``, or
            ``None`` (no dedup, return inputs unchanged).
        **extra_arrays: additional (n,) arrays to filter in parallel.

    Returns:
        dict with ``'X'``, ``'y'`` (if provided), plus strategy-specific
        metadata and any extra_arrays keys.
    """
    if strategy is None:
        result = {'X': np.asarray(X, dtype=np.float64)}
        if y is not None:
            result['y'] = np.asarray(y, dtype=np.float64)
        for name, arr in extra_arrays.items():
            result[name] = np.asarray(arr)
        return result
    if strategy == 'average':
        return dedup_average(X, y=y, **extra_arrays)
    if strategy == 'keep-first':
        return dedup_keep_first(X, y=y, **extra_arrays)
    if strategy == 'exact':
        return dedup_exact(X, y=y, **extra_arrays)
    raise ValueError(
        f"unknown dedup strategy: {strategy!r} "
        f"(use 'average', 'keep-first', 'exact', or None)")


def _check_degenerate_columns(X):
    """Raise on all-NaN or zero-variance columns (after NaN exclusion)."""
    all_nan = []
    zero_var = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(np.isnan(col)):
            all_nan.append(j)
            continue
        finite = col[~np.isnan(col)]
        if len(finite) > 0 and np.ptp(finite) == 0:
            zero_var.append(j)
    if all_nan:
        raise ValueError(f"all-NaN columns: {all_nan}")
    if zero_var:
        raise ValueError(f"zero-variance columns: {zero_var}")


# ---------------------------------------------------------------------------
# Warning checks (inform, don't block)
# ---------------------------------------------------------------------------

def _warn_near_duplicates(X, tol=1e-8):
    """Detect near-duplicate rows.

    Uses sorted-norm trick for O(n log n) candidate filtering,
    with pairwise fallback for n < 2000.

    Returns:
        list of (i, j) near-duplicate pairs.
    """
    n = X.shape[0]
    Xc = np.where(np.isnan(X), 0.0, X)
    norms = np.linalg.norm(Xc, axis=1)
    pairs = []

    if n < 2000:
        # Pairwise fallback — fine for small datasets
        for i in range(n):
            for j in range(i + 1, n):
                if abs(norms[i] - norms[j]) > tol:
                    continue
                if np.linalg.norm(Xc[i] - Xc[j]) <= tol:
                    pairs.append((i, j))
    else:
        # Sorted-norm trick: only compare neighbors within tol in norm
        order = np.argsort(norms)
        sorted_norms = norms[order]
        for idx in range(len(order)):
            i = order[idx]
            for jdx in range(idx + 1, len(order)):
                if sorted_norms[jdx] - sorted_norms[idx] > tol:
                    break
                j = order[jdx]
                if np.linalg.norm(Xc[i] - Xc[j]) <= tol:
                    pairs.append((min(i, j), max(i, j)))

    if pairs:
        shown = ', '.join(f'({a}, {b})' for a, b in pairs[:5])
        n_extra = len(pairs) - 5
        msg = f"near-duplicate rows (tol={tol}): {shown}"
        if n_extra > 0:
            msg += f" ... and {n_extra} more"
        warnings.warn(msg)

    return pairs


def _warn_label_conflicts(X, y, tol=1e-8):
    """Among near-duplicate row pairs, warn if y values differ."""
    # Suppress inner near-dup warnings (we report conflicts separately)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = _warn_near_duplicates(X, tol=tol)
    conflicts = [(i, j) for i, j in pairs if y[i] != y[j]]
    if conflicts:
        shown = ', '.join(
            f'({a}, {b}): y={y[a]} vs {y[b]}' for a, b in conflicts[:5])
        n_extra = len(conflicts) - 5
        msg = f"label conflicts among near-duplicate rows: {shown}"
        if n_extra > 0:
            msg += f" ... and {n_extra} more"
        warnings.warn(msg)


# ---------------------------------------------------------------------------
# Entity fingerprint check
# ---------------------------------------------------------------------------

def fingerprint_check(X, groups, test_fraction=0.2, seed=42):
    """Test if features encode entity identity (memorization risk).

    Splits each group's samples into train/test, fits a nearest-centroid
    classifier, and compares accuracy to chance (1/n_groups).  If features
    strongly identify which group a sample belongs to, models may memorize
    entity identity rather than learning the target — especially under
    non-grouped validation.

    Pure numpy, zero hyperparameters.  If even nearest-centroid achieves
    high accuracy, the fingerprint is undeniable.

    Args:
        X: (n, d) array.
        groups: (n,) array of group labels.
        test_fraction: fraction of each group's samples held out.
        seed: random seed for the train/test split.

    Returns:
        dict with keys:
            accuracy: float, fingerprint classification accuracy.
            chance: float, 1 / n_unique_groups.
            ratio: float, accuracy / chance.
            risk: str, 'low', 'medium', or 'high' (or 'skip').
            n_groups: int, number of unique groups.
            n_test: int, number of test samples.
    """
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 3:
        return {'accuracy': np.nan, 'chance': 1.0 / max(n_groups, 1),
                'ratio': np.nan, 'risk': 'skip',
                'n_groups': n_groups, 'n_test': 0}

    # Within-group train/test split — every group in both sets
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for g in unique_groups:
        mask = np.where(groups == g)[0]
        if len(mask) < 2:
            train_idx.extend(mask)
            continue  # singletons go to train only
        n_test = max(1, int(len(mask) * test_fraction))
        perm = rng.permutation(len(mask))
        test_idx.extend(mask[perm[:n_test]])
        train_idx.extend(mask[perm[n_test:]])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    if len(test_idx) < 3:
        return {'accuracy': np.nan, 'chance': 1.0 / n_groups,
                'ratio': np.nan, 'risk': 'skip',
                'n_groups': n_groups, 'n_test': len(test_idx)}

    # Normalize using train stats only
    X_train = X[train_idx]
    X_test = X[test_idx]
    means = np.nanmean(X_train, axis=0)
    stds = np.nanstd(X_train, axis=0)
    stds[stds == 0] = 1.0
    X_train_n = np.where(np.isnan(X_train - means), 0.0,
                         (X_train - means) / stds)
    X_test_n = np.where(np.isnan(X_test - means), 0.0,
                        (X_test - means) / stds)

    # Compute group centroids from train set
    g_train = groups[train_idx]
    centroid_labels = []
    centroid_list = []
    for g in unique_groups:
        mask = g_train == g
        if mask.any():
            centroid_labels.append(g)
            centroid_list.append(X_train_n[mask].mean(axis=0))

    if len(centroid_labels) < 2:
        return {'accuracy': np.nan, 'chance': 1.0 / n_groups,
                'ratio': np.nan, 'risk': 'skip',
                'n_groups': n_groups, 'n_test': len(test_idx)}

    centroid_matrix = np.array(centroid_list)  # (n_centroids, d)

    # Classify test samples to nearest centroid
    g_test = groups[test_idx]
    dists = np.linalg.norm(
        X_test_n[:, None, :] - centroid_matrix[None, :, :], axis=2)
    predicted = np.array([centroid_labels[i] for i in dists.argmin(axis=1)])

    accuracy = float((predicted == g_test).mean())
    chance = 1.0 / n_groups
    ratio = accuracy / chance

    if ratio < 3:
        risk = 'low'
    elif ratio < 10:
        risk = 'medium'
    else:
        risk = 'high'

    return {
        'accuracy': accuracy,
        'chance': chance,
        'ratio': ratio,
        'risk': risk,
        'n_groups': n_groups,
        'n_test': len(test_idx),
    }


def _warn_entity_fingerprint(X, groups):
    """Warn if features encode entity identity."""
    result = fingerprint_check(X, groups)
    if result['risk'] == 'medium':
        warnings.warn(
            f"entity fingerprint: nearest-centroid identifies groups at "
            f"{result['accuracy']:.1%} ({result['ratio']:.1f}x chance, "
            f"{result['n_groups']} groups) — consider grouped validation")
    elif result['risk'] == 'high':
        warnings.warn(
            f"entity fingerprint: nearest-centroid identifies groups at "
            f"{result['accuracy']:.1%} ({result['ratio']:.1f}x chance, "
            f"{result['n_groups']} groups) — features strongly encode "
            f"entity identity; grouped validation strongly recommended")


# ---------------------------------------------------------------------------
# ValidatedDataset
# ---------------------------------------------------------------------------

class ValidatedDataset:
    """Dataset that has passed validation checks.

    Cannot be constructed directly — use ValidatedDataset.validate() instead.

    Attributes:
        X: (n, d) float64 array.
        y: (n,) array or None.
        groups: (n,) array or None.
        feature_names: list of str or None.
    """

    def __init__(self, _token=None):
        if _token is not _SENTINEL:
            raise TypeError(
                "Use ValidatedDataset.validate(X, y, ...)")

    @classmethod
    def validate(cls, X, y=None, groups=None, feature_names=None,
                 source_ids=None):
        """Validate data and return a ValidatedDataset.

        Args:
            X: (n, d) array-like.
            y: (n,) array-like, optional.
            groups: (n,) array-like, optional (e.g. for grouped k-fold).
            feature_names: list of str, optional.
            source_ids: (n,) array-like, optional.  Labels identifying which
                data source each sample comes from (e.g. paper, lab,
                measurement technique).  Used for leave-one-source-out
                fold strategies.

        Returns:
            ValidatedDataset with validated X, y, groups, feature_names,
            source_ids.

        Raises:
            ValueError: on exact duplicates, all-NaN columns, or
                zero-variance columns.
        """
        X = np.asarray(X, dtype=np.float64)
        if y is not None:
            y = np.asarray(y)
            assert len(y) == len(X), "X/y length mismatch"

        # errors (block construction)
        _check_exact_duplicates(X)
        _check_degenerate_columns(X)

        # warnings (don't block)
        _warn_near_duplicates(X, tol=1e-8)
        if y is not None:
            _warn_label_conflicts(X, y, tol=1e-8)
        if groups is not None:
            _warn_entity_fingerprint(X, np.asarray(groups))

        obj = cls(_token=_SENTINEL)
        obj.X = X
        obj.y = y
        obj.groups = np.asarray(groups) if groups is not None else None
        obj.feature_names = feature_names
        obj.source_ids = np.asarray(source_ids) if source_ids is not None else None
        return obj

    def transform(self, fn, feature_names=None):
        """Apply a transform to X and re-validate the result.

        Catches degenerate outputs that transforms silently introduce
        (constant columns from downsampling, duplicates from collisions,
        all-NaN from bad feature engineering).

        Args:
            fn: callable, receives (n, d) array, returns (n, d') array.
                Row count must be preserved.
            feature_names: new feature names for the transformed X.
                If None and column count is unchanged, inherits from self.

        Returns:
            New ValidatedDataset with transformed X (y/groups/source_ids
            carry through).
        """
        X_new = np.asarray(fn(self.X), dtype=np.float64)
        if X_new.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"transform changed row count: {self.X.shape[0]} → "
                f"{X_new.shape[0]}")

        # Inherit feature_names only if shape is unchanged
        if feature_names is None:
            if (self.feature_names is not None
                    and X_new.shape[1] == self.X.shape[1]):
                feature_names = self.feature_names

        return ValidatedDataset.validate(
            X_new, y=self.y, groups=self.groups,
            feature_names=feature_names,
            source_ids=self.source_ids)
