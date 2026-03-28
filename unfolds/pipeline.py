"""Composable pipeline: feature engineering steps + model.

Receives read-only arrays, never sees sanctified data until final
evaluation.  Defines what sequence of feature engineering and
models/routing is used.

Steps are chained: output of step N becomes input of step N+1.
All output arrays are immutable (write-protected views).

Two usage modes:

**Feature engineering only**::

    pipe = Pipeline()
    pipe.add(MyStep(...))
    result = pipe.run(X_train, y_train, feature_names, X_val=X_val)

**Full recipe** (steps + model)::

    pipe = Pipeline()
    pipe.add(MyStep(...))
    pipe.set_model(model)

    for fold in exp.folds(k=5):
        pipe.fit(fold.X_train, fold.y_train,
                 exp.feature_names, X_val=fold.X_val)
        fold.record(pipe.predict())

Domain-specific step types (e.g. SoupStep) can be registered via
``register_step_type(name, cls)`` so that ``Pipeline.add('name', ...)``
works.
"""

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Step registry (populated by domain libraries, e.g. signalfault)
# ---------------------------------------------------------------------------

_STEP_REGISTRY = {}


def register_step_type(name, cls):
    """Register a step type for string-based lookup in Pipeline.add().

    Parameters
    ----------
    name : str
        Short name (e.g. ``'soup'``).
    cls : type
        Step subclass.
    """
    _STEP_REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def _freeze(arr):
    """Read-only view.  Zero-copy."""
    arr = np.asarray(arr)
    out = arr.view()
    out.setflags(write=False)
    return out


# ---------------------------------------------------------------------------
# Step base class
# ---------------------------------------------------------------------------

class Step(ABC):
    """Abstract base class for pipeline steps.

    Every step implements ``execute()`` with a fixed contract:
    receives arrays + feature names, returns a dict with transformed
    arrays.  Pipeline chains steps and enforces this type.

    Subclass this to create new step types (soup, normalize, ridge, ...).
    """

    @abstractmethod
    def execute(self, X_train, y, feature_names, X_val=None):
        """Execute the step.

        Parameters
        ----------
        X_train : ndarray (n_train, d)
        y : ndarray (n_train,)
            Target (or residuals) for steps that need it.
        feature_names : list of str
        X_val : ndarray (n_val, d), optional

        Returns
        -------
        dict with keys:
            'X_train' : ndarray (n_train, d')
            'X_val' : ndarray (n_val, d') or None
            'feature_names' : list of str
            'meta' : dict (optional, step-specific metadata)
        """
        ...


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

class PipelineResult:
    """Result of a pipeline run.  All arrays are read-only.

    Attributes
    ----------
    X_train : ndarray (n_train, d'), read-only
    X_val : ndarray (n_val, d') or None, read-only
    feature_names : list of str
    steps : list of dict
        Per-step metadata (history, library, etc.).
    """

    __slots__ = ('X_train', 'X_val', 'feature_names', 'steps')

    def __init__(self, X_train, X_val, feature_names, steps):
        self.X_train = _freeze(X_train)
        self.X_val = _freeze(X_val) if X_val is not None else None
        self.feature_names = feature_names
        self.steps = steps

    @property
    def n_features(self):
        return self.X_train.shape[1]

    @property
    def history(self):
        """SelectionHistory from the last soup step, or None."""
        for s in reversed(self.steps):
            if 'history' in s:
                return s['history']
        return None

    @property
    def names(self):
        """Alias for feature_names."""
        return self.feature_names


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Composable pipeline: feature engineering steps + model.

    Receives read-only arrays.  Defines the full recipe: what feature
    engineering to apply, what model to train.  Zero knowledge of folds,
    splits, or holdout — that's the experiment's job.
    """

    def __init__(self):
        self._steps = []
        self._model = None

    def add(self, step_or_type, **kwargs):
        """Add a feature engineering step.

        Parameters
        ----------
        step_or_type : str or Step
            Either a step type name (e.g. ``'soup'``) for registry
            lookup, or a :class:`Step` instance directly.
        **kwargs
            Step-specific parameters (only when *step_or_type* is a
            string).

        Returns
        -------
        self (for chaining)
        """
        if isinstance(step_or_type, Step):
            if kwargs:
                raise TypeError(
                    "cannot pass kwargs when adding a Step instance")
            self._steps.append(step_or_type)
        elif isinstance(step_or_type, str):
            if step_or_type not in _STEP_REGISTRY:
                raise ValueError(
                    f"Unknown step type '{step_or_type}', "
                    f"registered: {list(_STEP_REGISTRY)}. "
                    f"Did you import the module that registers it?")
            step = _STEP_REGISTRY[step_or_type](**kwargs)
            self._steps.append(step)
        else:
            raise TypeError(
                f"expected str or Step, got {type(step_or_type).__name__}")
        return self

    def set_model(self, model):
        """Set the model to train after feature engineering.

        Parameters
        ----------
        model : BaseMLModel
            Template model (cloned per fit, never fitted directly).

        Returns
        -------
        self (for chaining)
        """
        self._model = model
        return self

    # -- Full recipe: fit / predict ----------------------------------------

    def _run_steps(self, X_train, y, feature_names, X_val=None):
        """Run feature engineering steps, return processed arrays."""
        step_metas = []
        for step in self._steps:
            out = step.execute(X_train, y, feature_names, X_val=X_val)
            X_train = out['X_train']
            X_val = out.get('X_val')
            feature_names = out['feature_names']
            step_metas.append(out.get('meta', {}))
        return X_train, X_val, feature_names, step_metas

    def fit(self, X_train, y, feature_names=None, X_val=None):
        """Run feature engineering steps, then fit model.

        Parameters
        ----------
        X_train : array-like (n_train, d)
        y : array-like (n_train,)
        feature_names : list of str, optional
            Required when pipeline has feature engineering steps.
        X_val : array-like (n_val, d), optional
            Transformed alongside X_train by steps.  Cached for
            ``predict()``.

        Returns
        -------
        self
        """
        if self._model is None:
            raise ValueError("no model set — use set_model() first")

        X_train = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if feature_names is not None:
            feature_names = list(feature_names)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)

        # Feature engineering
        X_train, X_val, feature_names, step_metas = self._run_steps(
            X_train, y, feature_names, X_val)

        self._feature_names_ = feature_names
        self._step_metas_ = step_metas
        self._X_val_ = _freeze(X_val) if X_val is not None else None

        # Fit model
        self._model_ = self._model.clone()
        self._model_.fit(X_train, y)

        return self

    def predict(self, X=None):
        """Predict on cached X_val (from fit) or on raw X.

        When the pipeline has feature engineering steps, *X* must be
        None — use the X_val passed to ``fit()``, which was already
        transformed.

        When the pipeline has no steps, *X* can be passed directly
        and is forwarded to the model.

        Returns
        -------
        ndarray (n,)
        """
        if not hasattr(self, '_model_'):
            raise ValueError("pipeline not fitted — call fit() first")

        if X is not None:
            if self._steps:
                raise ValueError(
                    "cannot predict on new X when pipeline has feature "
                    "engineering steps — pass X_val during fit() instead")
            return self._model_.predict(
                np.asarray(X, dtype=np.float64))

        if self._X_val_ is None:
            raise ValueError(
                "no X_val cached — pass X_val during fit() "
                "or pass X to predict()")
        return self._model_.predict(self._X_val_)

    def analyze(self):
        """Pipeline diagnostics: per-step metadata then model analysis."""
        if not hasattr(self, '_model_'):
            raise ValueError("pipeline not fitted — call fit() first")

        step_info = []
        for i, (step, meta) in enumerate(
                zip(self._steps, self._step_metas_)):
            info = {'index': i, 'type': type(step).__name__}
            if meta:
                info['meta'] = meta
            step_info.append(info)

        return {
            'steps': step_info,
            'model': self._model_.analyze(),
        }

    @property
    def model_(self):
        """The fitted model (after fit)."""
        if not hasattr(self, '_model_'):
            raise ValueError("pipeline not fitted — call fit() first")
        return self._model_

    # -- Feature-engineering only (backward compat) -------------------------

    def run(self, X_train, y, feature_names, X_val=None):
        """Execute feature engineering steps only.

        Returns
        -------
        PipelineResult
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        feature_names = list(feature_names)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)

        if not self._steps:
            return PipelineResult(X_train, X_val, feature_names, [])

        X_train, X_val, feature_names, step_metas = self._run_steps(
            X_train, y, feature_names, X_val)

        return PipelineResult(X_train, X_val, feature_names, step_metas)
