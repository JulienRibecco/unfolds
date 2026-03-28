"""Shared y-hint utilities for signalfault.classify modules.

The "hint" pattern: a first-pass model produces y-hat, which is encoded
as sigmoid regime indicators and appended to the feature space for a
second-pass model.  This stacking approach lets the second-pass model
learn threshold-specific corrections.

Used by battery_electrolyte (GBT first-pass) and perovskite (NN first-pass).

Components:
  build_indicators  — sigmoid regime indicators from y-hat
  augment           — stack [features | indicators | y-hat]
  oof_splits        — inner k-fold index generator for leak-free OOF
"""

import numpy as np

from .nn import sigmoid


# ---------------------------------------------------------------------------
# Sigmoid regime indicators
# ---------------------------------------------------------------------------

def build_indicators(yhat, thresholds, tau=0.3):
    """Construct sigmoid regime indicators from first-pass predictions.

    Each indicator is g_k = sigmoid((yhat - t_k) / tau), encoding how
    far each sample is above threshold t_k.  Sharp tau (small) gives
    near-step-function indicators; smooth tau (large) gives gradual
    transitions.

    Args:
        yhat: (n,) float64 first-pass predictions
        thresholds: array-like of threshold values t_k
        tau: temperature (scalar or array-like matching thresholds).
             Smaller = sharper indicators.

    Returns:
        (n, K) float64 indicator matrix, K = len(thresholds)
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if np.ndim(tau) == 0:
        tau = np.full(len(thresholds), float(tau))
    else:
        tau = np.asarray(tau, dtype=np.float64)
    return np.column_stack([
        sigmoid((yhat - t) / ta) for t, ta in zip(thresholds, tau)
    ])


# ---------------------------------------------------------------------------
# Feature augmentation
# ---------------------------------------------------------------------------

def augment(X, yhat, indicators):
    """Stack [features | indicators | y-hat] into augmented feature matrix.

    Args:
        X: (n, d) raw feature matrix
        yhat: (n,) first-pass predictions
        indicators: (n, K) indicator matrix from build_indicators

    Returns:
        (n, d + K + 1) augmented feature matrix
    """
    return np.column_stack([X, indicators, yhat.reshape(-1, 1)])


# ---------------------------------------------------------------------------
# Inner CV for leak-free OOF
# ---------------------------------------------------------------------------

def oof_splits(n, n_folds=5):
    """Generate inner k-fold index splits for OOF construction.

    Yields (train_idx, test_idx) pairs that partition range(n) into
    n_folds contiguous chunks.  Used to generate leak-free first-pass
    predictions on training data.

    Args:
        n: number of samples
        n_folds: number of inner folds

    Yields:
        (train_idx, test_idx) numpy arrays
    """
    from .data import oof_indices
    yield from oof_indices(n, n_folds=n_folds)
