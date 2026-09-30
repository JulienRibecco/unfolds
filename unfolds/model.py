"""Base model protocol for unfolds.

ABC with ``RegressorMixin`` for ``score()``.
Adds ``analyze()`` for interpretability, ``clone()`` for safe
duplication, and the convention that models own their normalization
(raw arrays in, predictions out).

Convention: ``__init__`` stores hyperparameters only (no computation).
Fitted state uses trailing underscore (``means_``, ``params_``).
``check_is_fitted`` detects these automatically.

Usage::

    from unfolds import NLModel

    model = NLModel(hidden_sizes=(32,), epochs=5000)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print(model.analyze())
    print(model.score(X_val, y_val))   # R² — free from RegressorMixin

    # Works with Experiment:
    for fold in exp.folds(k=5):
        model = NLModel(hidden_sizes=(32,))
        model.fit(fold.X_train, fold.y_train)
        fold.record(model.predict(fold.X_val))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseMLModel(RegressorMixin, BaseEstimator, ABC):
    """Abstract base for unfolds regression models.

    Subclasses must implement ``fit(X, y)``, ``predict(X)``, and
    ``clone()``.  Override ``analyze()`` for model-specific diagnostics.
    """

    @abstractmethod
    def fit(self, X, y):
        """Train the model on raw arrays.

        Must set at least one fitted attribute (name ending with
        underscore) so that ``check_is_fitted`` works.

        Returns self for chaining.
        """
        ...

    @abstractmethod
    def predict(self, X):
        """Predict targets from raw arrays.

        Returns (n,) float64 array.
        """
        ...

    @abstractmethod
    def clone(self):
        """Return an unfitted copy with the same hyperparameters."""
        ...

    def analyze(self):
        """Model diagnostics for interpretability.

        Override in subclasses to return weights, feature importances,
        per-bracket stats, routing distributions, etc.
        """
        return {'type': type(self).__name__}


# ---------------------------------------------------------------------------
# NLModel — wraps _nn.py train_nl / predict_nl
# ---------------------------------------------------------------------------

class NLModel(BaseMLModel):
    """N-layer sigmoid NN with Huber loss and momentum SGD.

    Thin wrapper around ``unfolds.nn`` functions.
    Owns normalization internally — receives raw X, returns raw
    predictions.

    Parameters
    ----------
    hidden_sizes : tuple of int
        Neurons per hidden layer.  e.g. ``(32,)`` or ``(12, 6)``.
    epochs : int
        Training epochs.
    seed : int
        Random seed for weight initialization.
    lr : float
        Learning rate.
    l2 : float
        L2 regularization strength.
    huber_delta : float
        Huber loss transition point.
    clip_grad : float
        Max gradient norm for clipping.
    momentum : float
        SGD momentum coefficient.
    """

    def __init__(self, hidden_sizes=(4, 4), epochs=5000, seed=42,
                 lr=0.01, l2=1e-4, huber_delta=5.0, clip_grad=5.0,
                 momentum=0.9):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.seed = seed
        self.lr = lr
        self.l2 = l2
        self.huber_delta = huber_delta
        self.clip_grad = clip_grad
        self.momentum = momentum

    def clone(self):
        return NLModel(hidden_sizes=self.hidden_sizes, epochs=self.epochs,
                       seed=self.seed, lr=self.lr, l2=self.l2,
                       huber_delta=self.huber_delta, clip_grad=self.clip_grad,
                       momentum=self.momentum)

    def fit(self, X, y):
        from .nn import compute_norm_stats, normalize, train_nl

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.means_, self.stds_ = compute_norm_stats(X)
        Xn = normalize(X, self.means_, self.stds_)
        self.params_ = train_nl(
            Xn, y, list(self.hidden_sizes), self.epochs, self.seed,
            self.lr, self.l2, self.huber_delta, self.clip_grad,
            self.momentum)
        return self

    def predict(self, X):
        check_is_fitted(self)
        from .nn import normalize, predict_nl

        X = np.asarray(X, dtype=np.float64)
        Xn = normalize(X, self.means_, self.stds_)
        return predict_nl(Xn, self.params_)

    def analyze(self):
        check_is_fitted(self)
        weights = self.params_['W']
        biases = self.params_['b']
        n_params = sum(
            w.size + b.size for w, b in zip(weights, biases))

        # Per-layer weight stats
        layers = []
        for i, (w, b) in enumerate(zip(weights, biases)):
            layers.append({
                'shape': list(w.shape),
                'weight_norm': float(np.linalg.norm(w)),
                'weight_mean_abs': float(np.mean(np.abs(w))),
                'bias_norm': float(np.linalg.norm(b)),
            })

        return {
            'type': 'NLModel',
            'hidden_sizes': list(self.hidden_sizes),
            'n_layers': len(weights),
            'n_params': n_params,
            'layers': layers,
        }


# ---------------------------------------------------------------------------
# EnsembleModel — multi-seed averaging
# ---------------------------------------------------------------------------

class EnsembleModel(BaseMLModel):
    """Multi-seed or heterogeneous ensemble of regression models.

    Two modes (mutually exclusive):

    **Homogeneous** (``base`` + ``n_seeds``): Clones the base model N
    times with different seeds, trains each independently, averages
    predictions.  The base model must have a ``seed`` parameter.

    **Heterogeneous** (``models``): Takes a list of pre-configured
    models, clones and fits each independently, averages predictions.

    Parameters
    ----------
    base : BaseMLModel, optional
        Template model (cloned per seed, never fitted directly).
    n_seeds : int
        Number of ensemble members (homogeneous mode only).
    base_seed : int
        First seed (homogeneous mode only).
    models : list of BaseMLModel, optional
        Pre-configured models for heterogeneous ensemble.
    """

    def __init__(self, base=None, n_seeds=3, base_seed=42, models=None):
        self.base = base
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.models = models

    def clone(self):
        return EnsembleModel(
            base=self.base.clone() if self.base is not None else None,
            n_seeds=self.n_seeds, base_seed=self.base_seed,
            models=[m.clone() for m in self.models]
            if self.models is not None else None)

    def fit(self, X, y):
        if self.models is not None and self.base is not None:
            raise ValueError(
                "Specify either 'base' or 'models', not both")
        if self.models is None and self.base is None:
            raise ValueError(
                "Specify either 'base' or 'models'")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.models_ = []

        if self.models is not None:
            # Heterogeneous: clone and fit each model as-is
            for m in self.models:
                fitted = m.clone()
                fitted.fit(X, y)
                self.models_.append(fitted)
        else:
            # Homogeneous: clone base with different seeds
            has_seed = hasattr(self.base, 'seed')
            for i in range(self.n_seeds):
                m = self.base.clone()
                if has_seed:
                    m.seed = self.base_seed + i
                m.fit(X, y)
                self.models_.append(m)
        return self

    def predict_members(self, X):
        """Per-member predictions.

        Returns (n_members, n_samples) array.
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        return np.array([m.predict(X) for m in self.models_])

    def predict(self, X):
        return self.predict_members(X).mean(axis=0)

    def analyze(self):
        check_is_fitted(self)
        mode = 'heterogeneous' if self.models is not None else 'homogeneous'
        result = {
            'type': 'EnsembleModel',
            'mode': mode,
            'n_members': len(self.models_),
            'members': [m.analyze() for m in self.models_],
        }
        if mode == 'homogeneous':
            result['base_type'] = type(self.base).__name__
        return result


# ---------------------------------------------------------------------------
# StackedModel — sequential cascade (stage 1 augments stage 2)
# ---------------------------------------------------------------------------

class StackedModel(BaseMLModel):
    """Two-stage stacking cascade.

    Stage 1 predictions augment the feature matrix for stage 2.
    Inner splits keep each predicted row out of its first-stage fit.
    Choose grouped or temporal splits when rows are dependent.

    Parameters
    ----------
    first : BaseMLModel
        First stage model.
    second : BaseMLModel
        Second stage model.
    augment : str or callable
        How to build stage 2 input from (X, stage1_pred):
        - ``'append'``: hstack(X, pred) — prediction as extra column
        - ``'replace'``: use pred only
        - callable(X, pred_col) -> X_augmented
    oof_folds : int
        Number of inner splits. 0 uses in-sample predictions and is only
        allowed with the 'rows' strategy.
    oof_strategy : {'rows', 'groups', 'temporal'}
        'rows' uses contiguous k-fold chunks. 'groups' requires aligned
        labels in fit(X, y, groups=...). 'temporal' uses forward splits;
        input rows must already be chronological. Its initial training
        prefix and gap rows without OOF predictions are excluded from the
        second-stage fit. The final first stage still fits all input rows.
    oof_gap : int
        Excluded rows between inner training and validation, temporal only.
    """

    def __init__(self, first, second, augment='append', oof_folds=5,
                 oof_strategy='rows', oof_gap=0):
        self.first = first
        self.second = second
        self.augment = augment
        self.oof_folds = oof_folds
        self.oof_strategy = oof_strategy
        self.oof_gap = oof_gap

    def clone(self):
        return StackedModel(first=self.first.clone(), second=self.second.clone(),
                            augment=self.augment, oof_folds=self.oof_folds,
                            oof_strategy=self.oof_strategy, oof_gap=self.oof_gap)

    def _augment_X(self, X, pred):
        """Build stage 2 input."""
        pred_col = pred.reshape(-1, 1) if pred.ndim == 1 else pred
        if callable(self.augment):
            return self.augment(X, pred_col)
        elif self.augment == 'append':
            return np.hstack([X, pred_col])
        elif self.augment == 'replace':
            return pred_col
        raise ValueError(f"Unknown augment mode: {self.augment!r}")

    def fit(self, X, y, *, groups=None):
        """Fit using inner splits relative to these input rows.

        Group labels must be sliced to the same training rows as X/y.
        Experiment.run and Research do not forward group metadata; use
        the explicit fold loop for grouped stacking.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        from .data import (_validate_integer, _validate_labels, oof_indices,
                           grouped_kfold_indices, expanding_window_indices)

        _validate_integer('oof_folds', self.oof_folds, 0)
        _validate_integer('oof_gap', self.oof_gap, 0)
        if self.oof_strategy not in ('rows', 'groups', 'temporal'):
            raise ValueError("oof_strategy must be 'rows', 'groups', or 'temporal'")
        if self.oof_gap and self.oof_strategy != 'temporal':
            raise ValueError('oof_gap requires the temporal strategy')
        if groups is not None and self.oof_strategy != 'groups':
            raise ValueError("groups requires oof_strategy='groups'")
        if self.oof_strategy != 'rows' and not self.oof_folds:
            raise ValueError('grouped and temporal stacking require inner splits')
        if self.oof_strategy == 'groups':
            if groups is None:
                raise ValueError("oof_strategy='groups' requires groups in fit()")
            groups = _validate_labels('groups', groups, len(y))
            splits = grouped_kfold_indices(groups, k=self.oof_folds)
        elif self.oof_strategy == 'temporal':
            splits = expanding_window_indices(len(y), k=self.oof_folds,
                                              gap=self.oof_gap)
        else:
            splits = (list(oof_indices(len(y), self.oof_folds))
                      if self.oof_folds else [])

        # Fit first stage on all data (kept for predict-time)
        self.first_ = self.first.clone()
        self.first_.fit(X, y)

        # Stage 1 predictions for training stage 2
        if self.oof_folds > 0:
            oof_pred = np.full(len(y), np.nan)
            covered = np.zeros(len(y), dtype=bool)
            for tr_idx, val_idx in splits:
                inner = self.first.clone()
                inner.fit(X[tr_idx], y[tr_idx])
                oof_pred[val_idx] = inner.predict(X[val_idx])
                covered[val_idx] = True
            stage1_pred = oof_pred
        else:
            stage1_pred = self.first_.predict(X)
            covered = np.ones(len(y), dtype=bool)

        # Fit second stage on augmented features
        self.oof_train_idx_ = np.flatnonzero(covered)
        X2 = self._augment_X(X[covered], stage1_pred[covered])
        self.second_ = self.second.clone()
        self.second_.fit(X2, y[covered])
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        stage1_pred = self.first_.predict(X)
        X2 = self._augment_X(X, stage1_pred)
        return self.second_.predict(X2)

    def analyze(self):
        check_is_fitted(self)
        return {
            'type': 'StackedModel',
            'augment': self.augment if isinstance(self.augment, str)
                       else 'custom',
            'oof_folds': self.oof_folds,
            'oof_strategy': self.oof_strategy,
            'oof_gap': self.oof_gap,
            'n_stage2_samples': len(self.oof_train_idx_),
            'first': self.first_.analyze(),
            'second': self.second_.analyze(),
        }


# ---------------------------------------------------------------------------
# RoutedModel — expert selection by routing function
# ---------------------------------------------------------------------------

class RoutedModel(BaseMLModel):
    """Router assigns samples to specialized expert models.

    At **train** time, each expert is fitted on a subset of the data
    (determined by ``train_ranges`` or by ``route_fn``).

    At **predict** time, ``route_fn`` decides which expert handles
    each sample.

    Parameters
    ----------
    router : BaseMLModel or None
        Router model whose predictions feed ``route_fn``.
        None for feature-based routing (``route_fn`` uses X directly).
    experts : dict of {key: BaseMLModel}
        Expert models keyed by routing label.
    route_fn : callable
        ``route_fn(X, router_pred_or_None)`` returns array of keys
        matching the experts dict.
    train_ranges : dict of {key: (lo, hi)}, optional
        Target-value ranges for each expert's training data (may
        overlap).  If None, each expert trains on the samples that
        ``route_fn`` assigns to it.
    min_samples : int
        Minimum samples per expert.  Falls back to all data if below.
    """

    def __init__(self, router, experts, route_fn,
                 train_ranges=None, min_samples=2):
        self.router = router
        self.experts = experts
        self.route_fn = route_fn
        self.train_ranges = train_ranges
        self.min_samples = min_samples

    def clone(self):
        return RoutedModel(
            router=self.router.clone() if self.router is not None else None,
            experts={k: v.clone() for k, v in self.experts.items()},
            route_fn=self.route_fn,
            train_ranges=dict(self.train_ranges)
            if self.train_ranges is not None else None,
            min_samples=self.min_samples)

    def _route_keys(self, X, router_pred):
        keys = np.asarray(self.route_fn(X, router_pred), dtype=object)
        if keys.shape != (len(X),):
            raise ValueError('route_fn must return one key per input row')
        unknown = [key for key in dict.fromkeys(keys) if key not in self.experts]
        if unknown:
            raise ValueError(f'route_fn returned unknown expert keys: {unknown!r}')
        return keys

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Fit router (if model-based)
        if self.router is not None:
            self.router_ = self.router.clone()
            self.router_.fit(X, y)
            router_pred = self.router_.predict(X)
        else:
            self.router_ = None
            router_pred = None

        # Fit each expert on its training subset
        self.experts_ = {}
        if self.train_ranges is not None:
            for key, expert in self.experts.items():
                lo, hi = self.train_ranges[key]
                mask = (y >= lo) & (y < hi)
                if mask.sum() < self.min_samples:
                    mask = np.ones(len(y), dtype=bool)
                self.experts_[key] = expert.clone()
                self.experts_[key].fit(X[mask], y[mask])
        else:
            keys = self._route_keys(X, router_pred)
            for key, expert in self.experts.items():
                mask = (keys == key)
                if mask.sum() < self.min_samples:
                    mask = np.ones(len(y), dtype=bool)
                self.experts_[key] = expert.clone()
                self.experts_[key].fit(X[mask], y[mask])

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)

        if self.router_ is not None:
            router_pred = self.router_.predict(X)
        else:
            router_pred = None

        keys = self._route_keys(X, router_pred)
        pred = np.zeros(len(X))
        for key, expert in self.experts_.items():
            mask = (keys == key)
            if mask.any():
                pred[mask] = expert.predict(X[mask])
        return pred

    def routing_distribution(self, X):
        """Count how many samples route to each expert.

        Returns dict of ``{expert_key: count}``.
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)

        if self.router_ is not None:
            router_pred = self.router_.predict(X)
        else:
            router_pred = None

        keys = self._route_keys(X, router_pred)
        unique, counts = np.unique(keys, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def analyze(self):
        check_is_fitted(self)
        return {
            'type': 'RoutedModel',
            'n_experts': len(self.experts_),
            'train_ranges': self.train_ranges,
            'router': self.router_.analyze()
                     if self.router_ is not None else None,
            'experts': {k: v.analyze()
                        for k, v in self.experts_.items()},
        }


# ---------------------------------------------------------------------------
# Config types for cascade recipes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinConfig:
    """Routing bin: model template + name + training range.

    Parameters
    ----------
    name : str
        Human-readable bin label (used as expert key).
    model : BaseMLModel
        Template model for this expert (cloned at fit time).
    train : tuple of (float, float)
        Target-value range ``(lo, hi)`` for training data.
    """
    name: str
    model: BaseMLModel
    train: tuple


@dataclass(frozen=True)
class CascadeConfig:
    """Threshold-routed cascade recipe.

    Defines the full architecture: router model, routing thresholds,
    and per-bin expert models.  Training budget (``epochs``) is
    propagated to all models at build time.  Seed comes from the
    experiment, not the config.

    Parameters
    ----------
    router : BaseMLModel
        Template router model (e.g. ``EnsembleModel``).
    thresholds : list of float
        Sorted routing thresholds.  N thresholds → N+1 bins.
    bins : list of BinConfig
        Expert bins, ordered low → high.
    epochs : int
        Training budget, propagated to all models.
    min_samples : int
        Minimum samples per expert (falls back to all data).
    """
    router: BaseMLModel
    thresholds: list
    bins: list
    epochs: int = 5000
    min_samples: int = 50


# ---------------------------------------------------------------------------
# build_cascade — generic factory from CascadeConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ThresholdRouter:
    """Pickleable routing rule for cascades, including fitted snapshots."""

    thresholds: tuple
    names: tuple

    def __call__(self, X, router_pred):
        idx = np.digitize(router_pred, self.thresholds)
        return np.asarray(self.names, dtype=object)[idx]


def _set_epochs_and_seed(model, epochs, seed):
    """Propagate settings through composite model templates."""
    if isinstance(model, EnsembleModel):
        if model.models is not None:
            for member in model.models:
                _set_epochs_and_seed(member, epochs, seed)
        elif model.base is not None:
            model.base_seed = seed
            _set_epochs_and_seed(model.base, epochs, seed)
    elif isinstance(model, StackedModel):
        _set_epochs_and_seed(model.first, epochs, seed)
        _set_epochs_and_seed(model.second, epochs, seed + 1)
    elif isinstance(model, RoutedModel):
        if model.router is not None:
            _set_epochs_and_seed(model.router, epochs, seed)
        for i, expert in enumerate(model.experts.values()):
            _set_epochs_and_seed(expert, epochs, seed + i + 1)
    if hasattr(model, 'epochs'):
        model.epochs = epochs
    if hasattr(model, 'seed'):
        model.seed = seed


def build_cascade(config, seed):
    """Build a RoutedModel from a CascadeConfig.

    Propagates ``config.epochs`` to every model.  Assigns seeds
    deterministically: router members all get ``seed`` (diversity
    from architecture), experts get ``seed + 1, seed + 2, ...``

    Parameters
    ----------
    config : CascadeConfig
        Cascade recipe.
    seed : int
        Base random seed (from the experiment).

    Returns
    -------
    RoutedModel
        Unfitted model ready for ``.fit(X, y)``.
    """
    # ---- Router ----
    router = config.router.clone()
    _set_epochs_and_seed(router, config.epochs, seed)

    # ---- Experts ----
    experts = {}
    train_ranges = {}
    for i, b in enumerate(config.bins):
        expert = b.model.clone()
        _set_epochs_and_seed(expert, config.epochs, seed + i + 1)
        experts[b.name] = expert
        train_ranges[b.name] = b.train

    # ---- Routing function from thresholds ----
    route = _ThresholdRouter(tuple(config.thresholds),
                             tuple(b.name for b in config.bins))

    return RoutedModel(router, experts, route, train_ranges,
                       config.min_samples)
