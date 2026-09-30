"""Hierarchical ridge regression with shrinkage.

HierarchicalRidge: N-level ridge with shrinkage over arbitrary grouping
variables.  Any cascade with discrete routing (element count, source,
anion, density phase, CZ, ...) can use this instead of hand-rolling the
global -> group1 -> (group1 x group2) pattern.
"""

import numpy as np

from .data import fold_normalize


# ---------------------------------------------------------------------------
# Ridge primitives
# ---------------------------------------------------------------------------

def ridge_solve(X, y, alpha=1.0):
    """Ridge regression with unregularized bias term.

    Returns weight vector of length d+1 (last element is bias).
    """
    Xa = np.hstack([X, np.ones((len(X), 1))])
    I = np.eye(Xa.shape[1])
    I[-1, -1] = 0  # don't regularize bias
    return np.linalg.solve(Xa.T @ Xa + alpha * I, Xa.T @ y)


def ridge_predict(X, w):
    """Predict from pre-computed ridge weights (last element is bias)."""
    Xa = np.hstack([X, np.ones((len(X), 1))])
    return Xa @ w


# ---------------------------------------------------------------------------
# HierarchicalRidge
# ---------------------------------------------------------------------------

class HierarchicalRidge:
    """N-level hierarchical ridge with shrinkage.

    Given K grouping variables, builds K+1 levels::

      Level 0: global (all data)
      Level 1: per groups[0], shrunk toward global
      Level 2: per (groups[0], groups[1]), shrunk toward level 1
      ...

    At prediction time, each sample uses the finest available specialist
    and falls back to coarser levels when the group combo was unseen or
    had too few training samples.

    Args:
        alpha: ridge regularization strength.
        shrinkage: blend factor -- ``(1-s)*local + s*parent`` at each level.
        min_samples: minimum training samples to fit a specialist at each
            level.  ``int`` uses the same threshold for all levels.
            ``list[int]`` sets per-level thresholds (length K).
    """

    def __init__(self, alpha=1.0, shrinkage=0.2, min_samples=10):
        self.alpha = alpha
        self.shrinkage = shrinkage
        if isinstance(min_samples, (list, tuple)):
            self._min_list = [int(m) for m in min_samples]
            self._min_scalar = self._min_list[0]
        else:
            self._min_list = None
            self._min_scalar = int(min_samples)

        # Learned state
        self.global_w = None
        self._weights = {}   # {group_key_tuple: weight_vector}
        self._n_levels = 0

    def _min_at(self, level):
        if self._min_list is not None and level < len(self._min_list):
            return self._min_list[level]
        return self._min_scalar

    # ---- fit ----

    def fit(self, X, y, groups):
        """Fit hierarchical ridge.

        Args:
            X: (n, d) features (caller should normalize first).
            y: (n,) targets.
            groups: list of K arrays, each (n,).  ``groups[0]`` is the
                coarsest partition, ``groups[-1]`` the finest.

        Returns:
            self (for chaining).
        """
        groups = [np.asarray(g) for g in groups]
        self._n_levels = len(groups)
        self._weights = {}
        self.global_w = ridge_solve(X, y, self.alpha)
        self._fit_level(X, y, groups, 0, self.global_w,
                        np.ones(len(y), dtype=bool), ())
        return self

    def _fit_level(self, X, y, groups, level, parent_w, mask, key_prefix):
        if level >= self._n_levels:
            return
        g = groups[level]
        for val in sorted(set(g[mask])):
            child = mask & (g == val)
            key = key_prefix + (val,)
            if child.sum() < self._min_at(level):
                continue
            w = ridge_solve(X[child], y[child], self.alpha)
            w_shrunk = (1 - self.shrinkage) * w + self.shrinkage * parent_w
            self._weights[key] = w_shrunk
            self._fit_level(X, y, groups, level + 1, w_shrunk, child, key)

    # ---- predict ----

    def predict(self, X, groups):
        """Predict with hierarchical fallback.

        Each sample uses the finest specialist available; falls back to
        coarser levels when the group combo is unseen.

        Args:
            X: (n, d) features (same normalization as fit).
            groups: list of K arrays, each (n,).

        Returns:
            (n,) float64 predictions.
        """
        groups = [np.asarray(g) for g in groups]
        pred = np.full(len(X), np.nan)
        self._predict_level(X, pred, groups, 0, self.global_w,
                            np.ones(len(X), dtype=bool), ())
        # Safety: fill any leftover NaN with global
        nan = np.isnan(pred)
        if nan.any():
            pred[nan] = ridge_predict(X[nan], self.global_w)
        return pred

    def _predict_level(self, X, pred, groups, level, parent_w, mask, key_prefix):
        if not mask.any():
            return
        if level >= self._n_levels:
            # Leaf -- fill anything still NaN under this mask
            fill = mask & np.isnan(pred)
            if fill.any():
                pred[fill] = ridge_predict(X[fill], parent_w)
            return
        g = groups[level]
        for val in sorted(set(g[mask])):
            child = mask & (g == val)
            key = key_prefix + (val,)
            w = self._weights.get(key, parent_w)
            self._predict_level(X, pred, groups, level + 1, w, child, key)

    # ---- convenience ----

    def fit_predict(self, X, y, groups):
        """Fit and return training predictions (for residual computation).

        Note: these are in-sample predictions (not OOF), so they're
        slightly optimistic.  Fine for residual-soup selection where the
        final model is validated on a held-out set.
        """
        self.fit(X, y, groups)
        return self.predict(X, groups)
