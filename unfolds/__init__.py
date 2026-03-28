"""unfolds -- Leakage-free experiment framework for tabular ML.

Validation, sanctified holdout, fold-safe preprocessing,
experiment orchestration, cascade models, and composable pipelines.
"""

# Validation
from .validate import (
    ValidatedDataset, dedup_exact, dedup_average, dedup_keep_first,
    apply_dedup, fingerprint_check,
)

# Data splitting
from .data import (
    kfold_indices, grouped_kfold_indices, train_test_indices,
    oof_indices, temporal_split_indices, expanding_window_indices,
    iter_folds, FoldData,
    fold_impute, fold_normalize, fold_safe_preprocess,
    sanctified_indices, SanctifiedDataset, SanctifiedResult,
    derive_seeds,
)

# Experiment
from .experiment import ExperimentConfig, Experiment, Fold, FinalData, RunResult

# Bench
from .bench import Research, RunContext, RunHistory

# Models
from .model import (
    BaseMLModel, NLModel, EnsembleModel, StackedModel,
    RoutedModel, BinConfig, CascadeConfig, build_cascade,
)

# NN primitives
from .nn import sigmoid, compute_norm_stats, normalize, train_nl, predict_nl

# Cascade utilities
from .cascade import (
    ensemble_train, ensemble_predict, bracket_mae, print_brackets,
    fit_ensemble, predict_ensemble,
)

# Hint (y-hint stacking)
from .hint import build_indicators, augment, oof_splits

# Pipeline
from .pipeline import Step, Pipeline, PipelineResult, register_step_type

# Ridge
from .ridge import ridge_solve, ridge_predict, HierarchicalRidge
