#!/usr/bin/env python3
"""
Example: California Housing with unfolds
=========================================

Complete walkthrough of the unfolds data lifecycle on scikit-learn's
California Housing dataset. Predicts median house value from 8 features
(income, house age, rooms, bedrooms, population, occupancy, latitude,
longitude).

Demonstrates:
  1. Validation and dedup
  2. Sanctified holdout
  3. Experiment with k-fold CV
  4. Model composables (flat → ensemble → stacked → cascade)
  5. One-shot sanctified evaluation
  6. Research bench with CLI and run history

Usage:
    python california_housing.py              # full run (5-fold)
    python california_housing.py --quick      # 3-fold, fewer epochs
    python california_housing.py --exp flat   # single experiment
"""

import numpy as np
from sklearn.datasets import fetch_california_housing

from unfolds import (
    # Data lifecycle
    ValidatedDataset, dedup_average, apply_dedup,
    SanctifiedDataset,
    ExperimentConfig, Experiment,
    # Models
    NLModel, EnsembleModel, StackedModel,
    RoutedModel, BinConfig, CascadeConfig, build_cascade,
    # Research bench
    Research, RunContext,
    # Utilities
    fold_normalize, ridge_solve, ridge_predict, HierarchicalRidge,
)


# ── Data loader ───────────────────────────────────────────────────────
#
# The loader function is the contract between your data and unfolds.
# It receives a data directory and config, returns a SanctifiedDataset.

def load_california(data_dir, config):
    """Load California Housing → SanctifiedDataset."""
    data = fetch_california_housing()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    names = list(data.feature_names)

    # Step 1: Dedup (California Housing has some near-duplicates)
    result = dedup_average(X, y, warn_frac=0.5)
    X, y = result['X'], result['y']

    # Step 2: Validate (confirms no remaining issues)
    ValidatedDataset.validate(X, y)

    # Step 3: Sanctify (15% held out, never touched until the end)
    return SanctifiedDataset(
        X, y,
        feature_names=names,
        seed=config.seed,
        sanctified_fraction=config.sanctified_fraction,
    )


# ── Model factories ──────────────────────────────────────────────────
#
# Each factory takes a RunContext and returns a fresh model.
# unfolds calls it once per fold + once for final retraining.

class FlatRidge:
    """Simplest possible model: normalized ridge regression."""
    def fit(self, X, y):
        Xn, _, self._m, self._s = fold_normalize(X)
        self._w = ridge_solve(Xn, y, alpha=1.0)

    def predict(self, X):
        Xn = (X - self._m) / self._s
        return ridge_predict(Xn, self._w)


def _flat_ridge(ctx):
    return FlatRidge()


def _small_nn(ctx):
    """Single small neural network."""
    epochs = 500 if ctx.quick else 2000
    return NLModel(hidden_sizes=(16, 8), epochs=epochs, seed=ctx.config.seed)


def _ensemble_nn(ctx):
    """5 neural networks averaged (different seeds, same architecture)."""
    epochs = 500 if ctx.quick else 2000
    return EnsembleModel(
        base=NLModel(hidden_sizes=(16, 8), epochs=epochs),
        n_seeds=3 if ctx.quick else 5,
        base_seed=ctx.config.seed,
    )


def _stacked(ctx):
    """Two-stage stacking: NN stage 1 → ridge-augmented stage 2.

    Stage 1 predictions are computed via inner OOF (out-of-fold) to
    prevent stacking leakage: the predictions used as features for
    stage 2 never saw the corresponding training labels.
    """
    epochs = 500 if ctx.quick else 2000
    return StackedModel(
        first=NLModel(hidden_sizes=(16, 8), epochs=epochs,
                       seed=ctx.config.seed),
        second=NLModel(hidden_sizes=(8,), epochs=epochs,
                        seed=ctx.config.seed + 1),
        augment='append',   # stage 2 sees original features + stage 1 preds
        oof_folds=3,        # inner OOF folds
    )


def _cascade(ctx):
    """Bracket-routing cascade: router predicts value range, specialists
    handle each bracket.

    California Housing prices cluster around:
      - Low: < $1.5 (100k)
      - Mid: $1.5 - $3.5
      - High: > $3.5

    Each specialist trains on a slightly wider range than its bracket
    (overlapping training ranges prevent hard-cutoff artifacts).
    """
    epochs = 300 if ctx.quick else 1500
    config = CascadeConfig(
        router=EnsembleModel(
            base=NLModel(hidden_sizes=(8,), epochs=epochs),
            n_seeds=3,
            base_seed=ctx.config.seed,
        ),
        thresholds=[1.5, 3.5],
        bins=[
            BinConfig("low",  NLModel((8,), epochs=epochs),  train=(0, 2.0)),
            BinConfig("mid",  NLModel((12,), epochs=epochs), train=(1.0, 4.0)),
            BinConfig("high", NLModel((8,), epochs=epochs),  train=(3.0, 6.0)),
        ],
        epochs=epochs,
        min_samples=50,
    )
    return build_cascade(config, seed=ctx.config.seed)


# ── Wiring ────────────────────────────────────────────────────────────

def main():
    config = ExperimentConfig(
        seed=42,
        sanctified_fraction=0.15,
        k=5,
        brackets=[(0, 1.5, "low"), (1.5, 3.5, "mid"), (3.5, 6, "high")],
    )

    research = Research(
        "California Housing",
        load_california,
        config,
        save_dir="california_runs",
    )

    # Register experiments
    research.new_experiment("flat_ridge", _flat_ridge)
    research.new_experiment("small_nn", _small_nn)
    research.new_experiment("ensemble", _ensemble_nn)
    research.new_experiment("stacked", _stacked)
    research.new_experiment("cascade", _cascade)

    # CLI: python california_housing.py --quick --exp flat_ridge
    research.main()


if __name__ == '__main__':
    main()
