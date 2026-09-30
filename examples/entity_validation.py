"""Small, offline illustration inspired by the DrugComb investigation.

Run: python -m examples.entity_validation --output /tmp/entity-validation.json
"""

import argparse

import numpy as np

from .validation_audit import ridge_prediction, run_audit, save_report


def make_data(seed=42, n_entities=40, rows_per_entity=50):
    """Transferable signal plus an unpredictable entity-specific target offset.

    Sixty random measurements fingerprint each entity. On previously seen
    entities, ridge can use them to recover the entity's offset. Those offsets
    are independent across entities, so they do not transfer to new ones.
    No explicit entity ID is passed to the model.
    """
    rng = np.random.RandomState(seed)
    groups = np.repeat(np.arange(n_entities), rows_per_entity)
    fingerprints = rng.normal(size=(n_entities, 60))
    signal = rng.normal(size=(len(groups), 1))
    y = (2 * signal[:, 0] + 3 * rng.normal(size=n_entities)[groups]
         + 0.5 * rng.normal(size=len(groups)))
    X = np.column_stack([signal, fingerprints[groups]])
    names = ['transferable_signal'] + [f'entity_measurement_{i:02d}' for i in range(60)]
    return X, y, groups, names


def run(seed=42):
    X, y, groups, names = make_data(seed)
    candidates = ('signal_only', 'signal_and_fingerprints')

    def fit_predict(train, evaluation, candidate):
        cols = np.array([0]) if candidate == 'signal_only' else np.arange(X.shape[1])
        return ridge_prediction(X[train][:, cols], y[train], X[evaluation][:, cols],
                                np.asarray(names)[cols], alpha=1.0)

    return run_audit(y, groups, fit_predict, candidates, seed=seed, metadata={
        'dataset': 'synthetic; not DrugComb measurements',
        'generator': {'entities': 40, 'rows_per_entity': 50, 'fingerprint_features': 60,
                      'signal_coefficient': 2, 'entity_offset_std': 3, 'noise_std': 0.5},
        'model': 'Ridge(alpha=1.0), train-only imputation and standardization',
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='artifacts/entity-validation.json')
    args = parser.parse_args()
    report, artifacts = run(seed=args.seed)
    save_report(report, artifacts, args.output)


if __name__ == '__main__':
    main()
