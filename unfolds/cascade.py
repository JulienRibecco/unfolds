"""Shared cascade utilities for signalfault.classify modules.

Building blocks that are duplicated across cascade modules:
  - Multi-seed NN ensemble training and prediction
  - Per-bracket MAE evaluation and reporting
  - Normalization-aware train/predict wrappers

Used alongside _nn.py which provides the lower-level primitives
(sigmoid, normalize, train_nl, predict_nl).
"""

import numpy as np

from .nn import compute_norm_stats, normalize, train_nl, predict_nl


# ---------------------------------------------------------------------------
# Multi-seed NN ensemble
# ---------------------------------------------------------------------------

def ensemble_train(Xn, y, arch, epochs, seeds, **kwargs):
    """Train a multi-seed NN ensemble.

    Each seed produces an independent model with the same architecture.
    At prediction time, outputs are averaged.

    Args:
        Xn: (n, d) normalized feature matrix
        y: (n,) float64 targets
        arch: list of hidden layer sizes, e.g. [12, 6]
        epochs: number of training epochs
        seeds: list of random seeds
        **kwargs: passed to train_nl (lr, l2, huber_delta, etc.)

    Returns:
        list of param dicts (one per seed)
    """
    return [train_nl(Xn, y, arch, epochs, s, **kwargs) for s in seeds]


def ensemble_predict(Xn, params_list):
    """Predict with a multi-seed NN ensemble (mean of outputs).

    Args:
        Xn: (n, d) normalized feature matrix
        params_list: list of param dicts from ensemble_train

    Returns:
        (n,) float64 averaged predictions
    """
    return np.mean([predict_nl(Xn, p) for p in params_list], axis=0)


# ---------------------------------------------------------------------------
# Per-bracket evaluation
# ---------------------------------------------------------------------------

def bracket_mae(pred, y, brackets):
    """Compute per-bracket MAE.

    Args:
        pred: (n,) float64 predictions
        y: (n,) float64 true targets
        brackets: list of (lo, hi, name) tuples defining ranges.
                  Each sample where lo <= y < hi is assigned to that bracket.

    Returns:
        list of (name, count, mae) tuples for non-empty brackets
    """
    results = []
    for lo, hi, name in brackets:
        mask = (y >= lo) & (y < hi)
        n = int(mask.sum())
        if n > 0:
            mae = float(np.abs(pred[mask] - y[mask]).mean())
            results.append((name, n, mae))
    return results


def print_brackets(pred, y, brackets, label='', indent=2):
    """Print per-bracket MAE table.

    Args:
        pred: (n,) float64 predictions
        y: (n,) float64 true targets
        brackets: list of (lo, hi, name) tuples
        label: optional header label
        indent: number of leading spaces
    """
    pad = ' ' * indent
    results = bracket_mae(pred, y, brackets)
    if label:
        print('%s%s:' % (pad, label), flush=True)
        pad = ' ' * (indent + 2)
    header = '%s%-12s %5s  %6s' % (pad, 'bracket', 'n', 'MAE')
    print(header, flush=True)
    print('%s%s' % (pad, '-' * 27), flush=True)
    for name, n, mae in results:
        print('%s%-12s %5d  %.3f' % (pad, name, n, mae), flush=True)


# ---------------------------------------------------------------------------
# Normalized ensemble convenience (train + predict with auto-normalization)
# ---------------------------------------------------------------------------

def fit_ensemble(X, y, arch, epochs, seeds, **kwargs):
    """Normalize, train ensemble, return (params_list, means, stds).

    Convenience wrapper that handles normalization internally.

    Args:
        X: (n, d) raw feature matrix (NOT normalized)
        y: (n,) float64 targets
        arch, epochs, seeds, **kwargs: passed to ensemble_train

    Returns:
        (params_list, means, stds) tuple
    """
    means, stds = compute_norm_stats(X)
    Xn = normalize(X, means, stds)
    params_list = ensemble_train(Xn, y, arch, epochs, seeds, **kwargs)
    return params_list, means, stds


def predict_ensemble(X, params_list, means, stds):
    """Normalize and predict with a trained ensemble.

    Args:
        X: (n, d) raw feature matrix
        params_list: list of param dicts from fit_ensemble
        means, stds: normalization statistics

    Returns:
        (n,) float64 averaged predictions
    """
    Xn = normalize(X, means, stds)
    return ensemble_predict(Xn, params_list)
