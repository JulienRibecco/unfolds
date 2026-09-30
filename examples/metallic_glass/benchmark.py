"""Compare four models for metallic-glass transition temperature (kelvin)."""

from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unfolds import (
    EnsembleModel, ExperimentConfig, NLModel, Research, RoutedModel,
    SanctifiedDataset,
)
from .prepare import DEFAULT_DATA_DIR, FEATURE_NAMES


def load_glass(data_dir, config):
    path = Path(data_dir or DEFAULT_DATA_DIR) / "features.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python -m examples.metallic_glass.prepare")
    with np.load(path, allow_pickle=False) as data:
        if data["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("Unexpected descriptor schema; rerun preparation")
        return SanctifiedDataset(
            data["X"], data["y"], feature_names=FEATURE_NAMES,
            seed=config.seed, sanctified_fraction=config.sanctified_fraction,
            stratify=config.stratify,
        )


def chemistry_route(X, _):
    """A fixed input-only rule: over 30 atomic % Fe + Co + Ni."""
    return np.where(X[:, -1] > 30, "fe_co_ni_rich", "other")


def flat_ensemble(ctx):
    return EnsembleModel(
        base=NLModel(hidden_sizes=(8, 8), epochs=300 if ctx.quick else 3000,
                     l2=0.01),
        n_seeds=3, base_seed=ctx.config.seed,
    )


def chemistry_cascade(ctx):
    return RoutedModel(
        router=None, route_fn=chemistry_route,
        experts={"fe_co_ni_rich": flat_ensemble(ctx), "other": flat_ensemble(ctx)},
    )


def ridge(ctx):
    # The scaler is fitted inside each training fold by sklearn's pipeline.
    return make_pipeline(StandardScaler(), Ridge(alpha=10, solver="svd"))


def boosted_trees(ctx):
    return GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                     random_state=ctx.config.seed)


def make_research(save_dir="artifacts/metallic-glass"):
    research = Research(
        "Metallic-glass Tg (K)", load_glass,
        ExperimentConfig(seed=42, k=5, sanctified_fraction=0.15),
        save_dir=save_dir,
    )
    research.new_experiment("ridge", ridge)
    research.new_experiment("flat_ensemble", flat_ensemble)
    research.new_experiment("chemistry_cascade", chemistry_cascade)
    research.new_experiment("boosted_trees", boosted_trees)
    return research


if __name__ == "__main__":
    # Use importable function identities in pickled snapshots, even with -m.
    from examples.metallic_glass.benchmark import make_research as build
    build().main()
