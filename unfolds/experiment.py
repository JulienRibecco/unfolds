"""Experiment orchestrator for signalfault.classify.

Owns the full data lifecycle: validate → sanctified holdout → fold
iteration → result collection → final evaluation.

Sanctified data is structurally locked — Pipeline and model code
never see it. Arrays passed to fold consumers are immutable
(write-protected views, zero-copy).

Usage::

    from unfolds.experiment import Experiment, ExperimentConfig

    config = ExperimentConfig(
        seed=42, k=5, sanctified_fraction=0.15,
        brackets=[(0, 10, "low"), (10, 50, "mid"), (50, 200, "high")],
    )
    exp = Experiment(X, y, config=config, feature_names=names)

    for fold in exp.folds():
        pipe.fit(fold.X_train, fold.y_train,
                 exp.feature_names, X_val=fold.X_val)
        fold.record(pipe.predict())

    final = exp.final_evaluate()
    pipe.fit(final.X_dev, final.y_dev,
             exp.feature_names, X_val=final.X_sanct)
    exp.record_final(pipe.predict())
    exp.recap()
"""

from collections import namedtuple
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from .validate import ValidatedDataset
from .data import SanctifiedDataset, SanctifiedResult, train_test_indices


# ---------------------------------------------------------------------------
# Immutability helper
# ---------------------------------------------------------------------------

def _freeze(arr):
    """Read-only view of an array.  Zero-copy."""
    arr = np.asarray(arr)
    out = arr.view()
    out.setflags(write=False)
    return out


# ---------------------------------------------------------------------------
# Scoring helpers (thin wrappers over sklearn.metrics)
# ---------------------------------------------------------------------------

def _scores(y_true, y_pred):
    """Standard regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        'n': len(y_true),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _bracket_scores(y_true, y_pred, brackets):
    """Per-bracket regression metrics.

    Parameters
    ----------
    brackets : list of (lo, hi) or (lo, hi, name)
    """
    results = []
    for b in brackets:
        if len(b) == 3:
            lo, hi, name = b
        else:
            lo, hi = b
            name = f"{lo}-{hi}"
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n > 1:
            s = _scores(y_true[mask], y_pred[mask])
            s['name'] = name
        else:
            s = {'name': name, 'n': n, 'mae': None, 'r2': None, 'rmse': None}
        results.append(s)
    return results


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Experiment-level configuration.

    Bundles settings that would otherwise be scattered across protocol
    files and function arguments.  Pass to :class:`Experiment` via
    ``config=``.

    Parameters
    ----------
    seed : int
        Random seed for sanctified split and fold shuffling.
    sanctified_fraction : float
        Fraction of data held out for one-shot final evaluation.
    k : int
        Default number of folds for :meth:`Experiment.folds`.
    stratify : bool
        Stratify the sanctified split by target value.
    dedup : str or None
        Dedup strategy passed to :func:`_validate.apply_dedup`:
        ``'average'``, ``'keep-first'``, ``'exact'``, or ``None``
        (skip dedup, validation still runs).
    group_by : str or None
        Fold grouping strategy forwarded to :meth:`Experiment.folds`:
        ``None`` (random), ``'groups'`` (grouped k-fold),
        ``'source'`` (source-aware).
    dev_strategy : str
        How dev data is used for evaluation: ``'kfold'`` (default)
        for k-fold cross-validation, ``'holdout'`` for a single
        train/val split.
    holdout_fraction : float
        Fraction of dev data used for validation when
        ``dev_strategy='holdout'``.  Ignored for kfold.
    brackets : list of (lo, hi) or (lo, hi, name), optional
        Target-value brackets for per-regime evaluation and
        distribution display.
    """
    seed: int = 42
    sanctified_fraction: float = 0.15
    k: int = 5
    stratify: bool = True
    dedup: Optional[str] = None
    group_by: Optional[str] = None
    dev_strategy: str = 'kfold'
    holdout_fraction: float = 0.20
    temporal: bool = False
    temporal_gap: int = 0
    brackets: Optional[List] = None


# ---------------------------------------------------------------------------
# Fold — single train/val split with immutable data
# ---------------------------------------------------------------------------

class Fold:
    """A single train/val fold with immutable data.

    All array attributes are read-only views.  Attempting to write
    raises ``ValueError``.

    Feature names are not carried on the Fold — they are schema-level
    metadata owned by ``_data.SanctifiedDataset``.  Access them via
    ``exp.feature_names``.

    Attributes
    ----------
    fold : int
        Fold index (0-based).
    n_folds : int
        Total number of folds.
    tr_idx : ndarray
        Dev-relative train indices (0..n_dev-1).
        Use to slice dev-only auxiliary arrays.
    te_idx : ndarray
        Dev-relative test indices.
    X_train : ndarray (n_train, d), read-only
    X_val : ndarray (n_val, d), read-only
    y_train : ndarray (n_train,), read-only
    y_val : ndarray (n_val,), read-only
    """

    __slots__ = ('fold', 'n_folds', 'tr_idx', 'te_idx',
                 'X_train', 'X_val', 'y_train', 'y_val',
                 '_predictions')

    def __init__(self, fold, n_folds, tr_idx, te_idx,
                 X_train, X_val, y_train, y_val):
        self.fold = fold
        self.n_folds = n_folds
        self.tr_idx = _freeze(tr_idx)
        self.te_idx = _freeze(te_idx)
        self.X_train = _freeze(X_train)
        self.X_val = _freeze(X_val)
        self.y_train = _freeze(y_train)
        self.y_val = _freeze(y_val)
        self._predictions = None

    def record(self, predictions):
        """Record predictions for this fold's validation set.

        Parameters
        ----------
        predictions : array-like (n_val,)
            Model predictions aligned with ``X_val`` / ``y_val``.

        Raises
        ------
        ValueError
            If length doesn't match validation set.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        if len(predictions) != len(self.te_idx):
            raise ValueError(
                f"predictions length {len(predictions)} != "
                f"validation set size {len(self.te_idx)}")
        self._predictions = predictions.copy()

    @property
    def recorded(self):
        """Whether predictions have been recorded for this fold."""
        return self._predictions is not None


# ---------------------------------------------------------------------------
# FinalData — one-shot sanctified access
# ---------------------------------------------------------------------------

FinalData = namedtuple('FinalData', [
    'X_dev', 'y_dev', 'dev_idx',
    'X_sanct', 'y_sanct', 'sanct_idx',
])

RunResult = namedtuple('RunResult', [
    'recap', 'final_model', 'final_predictions', 'final_data',
])


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """Top-level experiment orchestrator.

    Accepts a :class:`SanctifiedDataset` (the only valid input) and
    orchestrates: fold iteration → result collection → final evaluation.

    Sanctified data is structurally unreachable until
    :meth:`final_evaluate` is called.

    Parameters
    ----------
    ds : SanctifiedDataset
        Pre-validated, pre-split dataset.  Domain loaders (e.g.
        ``load_superconductor``) produce this directly.
    config : ExperimentConfig, optional
        Experiment-level settings (k, brackets, etc.).
    """

    def __init__(self, ds, config=None):

        if not isinstance(ds, SanctifiedDataset):
            raise TypeError(
                f"Experiment requires a SanctifiedDataset, "
                f"got {type(ds).__name__}. Use your domain loader "
                f"or construct SanctifiedDataset directly.")

        self._ds = ds
        self._config = config
        self._feature_names = ds.feature_names
        self._seed = ds._seed
        self._folds = []        # completed Fold objects
        self._finalized = False
        self._final_predictions = None
        self._sanct_idx = None      # cached after final_evaluate
        self._sanct_y = None        # cached after final_evaluate

    # ---- properties ----

    @property
    def n_samples(self):
        """Total samples (dev + sanctified)."""
        return self._ds.n_samples

    @property
    def n_dev(self):
        """Number of development samples."""
        return len(self._ds.X)

    @property
    def n_sanctified(self):
        """Number of sanctified (holdout) samples."""
        if self._sanct_idx is not None:
            return len(self._sanct_idx)
        return self._ds.n_sanctified

    @property
    def n_features(self):
        """Number of features."""
        return self._ds.X.shape[1]

    @property
    def feature_names(self):
        """Feature names (or None)."""
        return self._feature_names

    # ---- fold iteration ----

    def target_distribution(self, brackets=None):
        """Count samples per target-value bracket.

        Parameters
        ----------
        brackets : list of (lo, hi) or (lo, hi, name), optional
            Defaults to ``config.brackets`` if a config was provided.

        Returns
        -------
        list of dict
            Each with ``'name'``, ``'n'``, ``'frac'``, ``'mean'``,
            ``'std'``.
        """
        if brackets is None and self._config is not None:
            brackets = self._config.brackets
        if brackets is None:
            raise ValueError(
                "no brackets — pass brackets or set config.brackets")

        y = self._ds.y
        dist = []
        for b in brackets:
            if len(b) == 3:
                lo, hi, name = b
            else:
                lo, hi = b
                name = f"{lo}-{hi}"
            mask = (y >= lo) & (y < hi)
            n = int(mask.sum())
            dist.append({
                'name': name,
                'n': n,
                'frac': n / len(y) if len(y) > 0 else 0,
                'mean': float(y[mask].mean()) if n > 0 else 0,
                'std': float(y[mask].std()) if n > 1 else 0,
            })
        return dist

    def folds(self, k=None, group_by=None):
        """Iterate development folds with immutable data.

        Sanctified data is never exposed.  All yielded arrays are
        read-only views.

        Parameters
        ----------
        k : int, optional
            Number of folds.  Defaults to ``config.k`` if a config
            was provided, otherwise 5.
        group_by : str or None
            Grouping dimension for fold assignment:

            - ``None`` — random k-fold (default).
            - ``'groups'`` — grouped k-fold using *groups* passed at
              construction.  All samples sharing a group stay together.
            - ``'source'`` — grouped k-fold using *source_ids*.
              Use ``k == n_sources`` for leave-one-source-out.

        Yields
        ------
        Fold
            With read-only ``X_train``, ``X_val``, ``y_train``,
            ``y_val``, and absolute ``tr_idx`` / ``te_idx``.
        """
        if k is None:
            k = self._config.k if self._config is not None else 5

        if self._finalized:
            raise RuntimeError(
                "cannot iterate folds after final_evaluate()")

        X = self._ds.X
        y = self._ds.y
        self._folds = []

        for fd in self._ds.iter_dev_folds(k=k, group_by=group_by):
            fold = Fold(
                fold=fd.fold,
                n_folds=fd.n_folds,
                tr_idx=fd.tr_idx,
                te_idx=fd.te_idx,
                X_train=X[fd.tr_idx],
                X_val=X[fd.te_idx],
                y_train=y[fd.tr_idx],
                y_val=y[fd.te_idx],
            )
            self._folds.append(fold)
            yield fold

    def holdout(self, fraction=None):
        """Single train/val split on dev data.

        Alternative to :meth:`folds` for experiments that don't use
        k-fold CV.  Yields a single :class:`Fold` with ``n_folds=1``.

        Parameters
        ----------
        fraction : float, optional
            Fraction of dev data for validation.  Defaults to
            ``config.holdout_fraction`` (0.20).
        """
        if fraction is None:
            fraction = 0.20
            if self._config is not None:
                fraction = self._config.holdout_fraction

        if self._finalized:
            raise RuntimeError(
                "cannot iterate after final_evaluate()")

        X = self._ds.X
        y = self._ds.y
        n = len(X)
        self._folds = []

        tr_idx, te_idx = train_test_indices(
            n, test_fraction=fraction, seed=self._seed)

        fold = Fold(
            fold=0, n_folds=1,
            tr_idx=tr_idx, te_idx=te_idx,
            X_train=X[tr_idx], X_val=X[te_idx],
            y_train=y[tr_idx], y_val=y[te_idx],
        )
        self._folds.append(fold)
        yield fold

    # ---- final evaluation ----

    def final_evaluate(self):
        """One-shot access to sanctified data for final evaluation.

        Returns ``FinalData`` with dev and sanctified arrays (all
        read-only).  Can only be called once — raises on second call.

        Typical usage::

            final = exp.final_evaluate()
            model.train(final.X_dev, final.y_dev)
            pred = model.predict(final.X_sanct)
            mae = np.abs(pred - final.y_sanct).mean()

        Returns
        -------
        FinalData
            Named tuple with ``X_dev``, ``y_dev``, ``dev_idx``,
            ``X_sanct``, ``y_sanct``, ``sanct_idx``.

        Raises
        ------
        RuntimeError
            On second call.
        """
        # Delegate to SanctifiedDataset's one-shot guard
        result = self._ds.final_evaluate()
        self._finalized = True

        # Cache for record_final / recap (sanct arrays are consumed)
        self._sanct_idx = result.sanct_idx.copy()
        self._sanct_y = result.y_sanct.copy()

        return FinalData(
            X_dev=_freeze(self._ds.X),
            y_dev=_freeze(self._ds.y),
            dev_idx=_freeze(self._ds.dev_idx),
            X_sanct=_freeze(result.X_sanct),
            y_sanct=_freeze(result.y_sanct),
            sanct_idx=_freeze(result.sanct_idx),
        )

    # ---- sanctified recording ----

    def record_final(self, predictions):
        """Record sanctified predictions after :meth:`final_evaluate`.

        Parameters
        ----------
        predictions : array-like (n_sanctified,)

        Raises
        ------
        RuntimeError
            If ``final_evaluate()`` has not been called yet.
        """
        if not self._finalized:
            raise RuntimeError(
                "call final_evaluate() before record_final()")
        predictions = np.asarray(predictions, dtype=np.float64)
        n_sanct = len(self._sanct_idx)
        if len(predictions) != n_sanct:
            raise ValueError(
                f"predictions length {len(predictions)} != "
                f"sanctified size {n_sanct}")
        self._final_predictions = predictions.copy()

    # ---- external sanctified injection (used by Research) ----

    def _set_sanctified(self, sanct_idx, sanct_y):
        """Inject sanctified metadata externally (used by Research).

        Allows ``record_final()`` and ``recap()`` to include sanctified
        metrics without calling ``final_evaluate()`` on the underlying
        dataset.  Research calls ``ds.final_evaluate()`` once and shares
        the result across all experiments.

        Parameters
        ----------
        sanct_idx : array-like
            Absolute indices of the sanctified samples.
        sanct_y : array-like
            True target values for the sanctified samples.
        """
        self._finalized = True
        self._sanct_idx = np.asarray(sanct_idx).copy()
        self._sanct_y = np.asarray(sanct_y, dtype=np.float64).copy()

    # ---- reporting ----

    def summary(self):
        """Print dev CV metrics (compact).  See :meth:`recap` for full."""
        r = self.recap(brackets=None, verbose=False)
        if r is None:
            if self._folds:
                print(f"  {len(self._folds)} folds iterated, "
                      f"none recorded (call fold.record())")
            return
        dev = r['dev']
        k = r['data']['k']
        print(f"Experiment: {self.n_samples} samples, "
              f"{self.n_features} features")
        print(f"  Dev: {self.n_dev}  Sanctified: {self.n_sanctified}")
        print(f"\n  Dev CV ({k}-fold):")
        print(f"    MAE:  {dev['mae']['mean']:.4f} "
              f"+/- {dev['mae']['std']:.4f}")
        print(f"    R2:   {dev['r2']['mean']:.4f} "
              f"+/- {dev['r2']['std']:.4f}")
        for i, (m, r2) in enumerate(zip(dev['mae']['per_fold'],
                                         dev['r2']['per_fold'])):
            print(f"    Fold {i}: MAE={m:.4f}  R2={r2:.4f}")

    def recap(self, brackets=None, verbose=True):
        """Structured experiment summary.

        Computes dev CV metrics from recorded folds and (if
        ``record_final`` was called) sanctified metrics.  Optionally
        breaks down by target-value brackets.

        Parameters
        ----------
        brackets : list of (lo, hi) or (lo, hi, name), optional
            Target-value brackets for per-bracket breakdown.
            Defaults to ``config.brackets`` if a config was provided.
        verbose : bool
            Print formatted summary (default True).

        Returns
        -------
        dict or None
            Nested dict with keys ``'data'``, ``'dev'``,
            ``'sanctified'``, ``'comparison'``, ``'brackets'``.
            None if no folds have been recorded.
        """
        if brackets is None and self._config is not None:
            brackets = self._config.brackets

        recorded = [f for f in self._folds if f.recorded]
        if not recorded:
            if verbose:
                print("  No folds recorded.")
            return None

        k = recorded[0].n_folds

        # ---- per-fold scores ----
        fold_scores = [
            _scores(np.asarray(f.y_val), f._predictions)
            for f in recorded
        ]
        dev = {}
        for metric in ('mae', 'r2', 'rmse'):
            vals = [s[metric] for s in fold_scores]
            dev[metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'per_fold': vals,
            }

        # ---- sanctified ----
        sanctified = None
        if self._final_predictions is not None:
            sanctified = _scores(self._sanct_y, self._final_predictions)

        # ---- comparison ----
        comparison = None
        if sanctified is not None:
            comparison = {
                'mae_gap': dev['mae']['mean'] - sanctified['mae'],
                'r2_gap': dev['r2']['mean'] - sanctified['r2'],
            }

        # ---- brackets ----
        bracket_results = None
        target_dist = None
        if brackets is not None:
            oof_y = np.concatenate(
                [np.asarray(f.y_val) for f in recorded])
            oof_pred = np.concatenate(
                [f._predictions for f in recorded])
            bracket_results = {
                'dev': _bracket_scores(oof_y, oof_pred, brackets),
            }
            if sanctified is not None:
                bracket_results['sanctified'] = _bracket_scores(
                    self._sanct_y, self._final_predictions, brackets)
            target_dist = self.target_distribution(brackets)

        result = {
            'data': {
                'n_samples': self.n_samples,
                'n_features': self.n_features,
                'n_dev': self.n_dev,
                'n_sanctified': self.n_sanctified,
                'k': k,
                'target_distribution': target_dist,
            },
            'dev': dev,
            'sanctified': sanctified,
            'comparison': comparison,
            'brackets': bracket_results,
        }

        if verbose:
            _print_recap(result)

        return result

    # ---- automated run ----

    def run(self, model_fn, *, verbose=True, group_by=None):
        """Run full experiment: k-fold CV + final sanctified evaluation.

        Automates the fold loop, final retrain, and recap reporting.
        Equivalent to manually iterating :meth:`folds`, calling
        :meth:`final_evaluate`, :meth:`record_final`, and :meth:`recap`.

        Parameters
        ----------
        model_fn : callable
            Zero-arg factory returning a fresh model with ``fit(X, y)``
            and ``predict(X)``.  Called once per fold and once for the
            final retrain on the full dev set.

            For :class:`Pipeline` objects (which need ``feature_names``
            and ``X_val``), ``model_fn`` should return a configured
            Pipeline.  The method detects this and calls the
            appropriate fit/predict signatures automatically.

            Examples::

                # BaseMLModel
                exp.run(lambda: NLModel(hidden_sizes=(12, 6)))

                # CascadeConfig via build_cascade
                exp.run(lambda: build_cascade(TC_CASCADE, seed=42))

        verbose : bool
            Print per-fold MAE and full recap (default True).
        group_by : str or None
            Fold grouping strategy, forwarded to :meth:`folds`.

        Returns
        -------
        RunResult
            Named tuple with ``recap`` (dict), ``final_model`` (fitted
            model), ``final_predictions`` (sanctified predictions
            array), ``final_data`` (:class:`FinalData`).
        """
        from .pipeline import Pipeline as _Pipeline

        is_pipeline = None

        # ---- k-fold CV ----
        for fold in self.folds(group_by=group_by):
            model = model_fn()

            if is_pipeline is None:
                is_pipeline = isinstance(model, _Pipeline)

            if is_pipeline:
                model.fit(fold.X_train, fold.y_train,
                          self.feature_names, X_val=fold.X_val)
                pred = model.predict()
            else:
                model.fit(fold.X_train, fold.y_train)
                pred = model.predict(fold.X_val)

            fold.record(pred)

            if verbose:
                mae = float(np.abs(pred - np.asarray(fold.y_val)).mean())
                print(f"  Fold {fold.fold}: MAE={mae:.4f}")

        # ---- final sanctified evaluation ----
        final = self.final_evaluate()
        final_model = model_fn()

        if is_pipeline:
            final_model.fit(final.X_dev, final.y_dev,
                            self.feature_names, X_val=final.X_sanct)
            final_predictions = np.asarray(final_model.predict(),
                                           dtype=np.float64)
        else:
            final_model.fit(np.asarray(final.X_dev), np.asarray(final.y_dev))
            final_predictions = np.asarray(
                final_model.predict(np.asarray(final.X_sanct)),
                dtype=np.float64)

        self.record_final(final_predictions)

        recap_dict = self.recap(verbose=verbose)

        return RunResult(
            recap=recap_dict,
            final_model=final_model,
            final_predictions=final_predictions,
            final_data=final,
        )


# ---------------------------------------------------------------------------
# Recap formatter
# ---------------------------------------------------------------------------

def _print_recap(r):
    """Print a formatted experiment recap."""
    d = r['data']
    W = 54

    print(f"\n{'=' * W}")
    print(f"  EXPERIMENT RECAP")
    print(f"{'=' * W}")
    print(f"  {d['n_samples']} samples x {d['n_features']} features")
    if d['k'] == 1:
        print(f"  Dev: {d['n_dev']} (holdout)  "
              f"Sanctified: {d['n_sanctified']}")
    else:
        print(f"  Dev: {d['n_dev']} ({d['k']}-fold CV)  "
              f"Sanctified: {d['n_sanctified']}")

    # Target distribution
    tdist = d.get('target_distribution')
    if tdist is not None:
        print(f"\n  Target distribution")
        print(f"  {'-' * 40}")
        print(f"  {'':>12s}  {'N':>6s}  {'%':>5s}  "
              f"{'mean':>7s}  {'std':>7s}")
        for b in tdist:
            print(f"  {b['name']:>12s}  {b['n']:>6d}  "
                  f"{b['frac']:>5.1%}  "
                  f"{b['mean']:>7.1f}  {b['std']:>7.1f}")

    # Dev metrics
    dev = r['dev']
    if d['k'] == 1:
        dev_header = 'Dev holdout'
    else:
        dev_header = 'Dev CV'
    print(f"\n  {dev_header:<12s}  {'mean':>8s}  {'+/- std':>9s}")
    print(f"  {'-' * 33}")
    for metric in ('mae', 'r2', 'rmse'):
        m = dev[metric]
        label = 'R2' if metric == 'r2' else metric.upper()
        print(f"  {label:<12s}  {m['mean']:>8.4f}  +/- {m['std']:.4f}")

    if d['k'] > 1:
        print(f"  Per-fold MAE: ", end="")
        print("  ".join(
            f"F{i}={v:.2f}" for i, v in enumerate(dev['mae']['per_fold'])))

    # Sanctified
    sanct = r['sanctified']
    if sanct is not None:
        print(f"\n  {'Sanctified':<12s}")
        print(f"  {'-' * 33}")
        for metric in ('mae', 'r2', 'rmse'):
            label = 'R2' if metric == 'r2' else metric.upper()
            print(f"  {label:<12s}  {sanct[metric]:>8.4f}")

    # Comparison
    comp = r['comparison']
    if comp is not None:
        print(f"\n  Dev vs Sanctified")
        print(f"  {'-' * 33}")
        gap = comp['mae_gap']
        direction = "dev worse" if gap > 0 else "dev better"
        pct = (abs(gap) / sanct['mae'] * 100) if sanct['mae'] > 0 else 0
        print(f"  MAE gap:  {gap:+.4f} ({direction}, {pct:.1f}%)")
        print(f"  R2 gap:   {comp['r2_gap']:+.4f}")

    # Brackets
    bk = r['brackets']
    if bk is not None:
        has_sanct = 'sanctified' in bk
        print(f"\n  Brackets")
        print(f"  {'-' * (50 if has_sanct else 33)}")
        hdr = f"  {'':>12s}  {'N':>5s}  {'MAE':>7s}"
        if has_sanct:
            hdr += f"  {'N(s)':>5s}  {'MAE(s)':>7s}"
        print(hdr)
        dev_bk = bk['dev']
        sanct_bk = bk.get('sanctified', [])
        for i, bd in enumerate(dev_bk):
            if bd['n'] == 0:
                continue
            row = f"  {bd['name']:>12s}  {bd['n']:>5d}  {bd['mae']:>7.2f}"
            if has_sanct and i < len(sanct_bk) and sanct_bk[i]['n'] > 0:
                bs = sanct_bk[i]
                row += f"  {bs['n']:>5d}  {bs['mae']:>7.2f}"
            print(row)

    print(f"{'=' * W}")
