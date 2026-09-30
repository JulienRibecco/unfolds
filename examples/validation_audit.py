"""Shared evaluation protocol for the synthetic and DrugComb examples."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from unfolds import fold_normalize, grouped_kfold_indices, kfold_indices, sanctified_indices


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ridge_prediction(X_train, y_train, X_test, names, alpha=1.0):
    """Fit filtering, imputation, normalization, and ridge on training rows."""
    train = np.asarray(X_train, dtype=float).copy()
    test = np.asarray(X_test, dtype=float).copy()
    train[~np.isfinite(train)] = np.nan
    test[~np.isfinite(test)] = np.nan
    keep = ~np.isnan(train).all(axis=0)
    train, test = train[:, keep], test[:, keep]
    names = np.asarray(names)[keep]
    medians = np.nanmedian(train, axis=0)
    train = np.where(np.isnan(train), medians, train)
    test = np.where(np.isnan(test), medians, test)
    keep = train.std(axis=0) > 1e-12
    if not keep.any():
        raise ValueError('No varying training features remain')
    train, test = train[:, keep], test[:, keep]
    train, test, _, _ = fold_normalize(train, test)
    model = Ridge(alpha=alpha).fit(train, y_train)
    return model.predict(test), names[keep].tolist()


def metrics(y, prediction):
    return {'r2': float(r2_score(y, prediction)),
            'mae': float(mean_absolute_error(y, prediction))}


def run_audit(y, groups, fit_predict, candidates, *, seed=42, folds=5,
              holdout_fraction=0.2, metadata=None):
    """Choose candidates on dev CV, then score both choices on one fixed holdout.

    fit_predict(train_rows, evaluation_rows, candidate) must learn every
    data-dependent transformation on train_rows only. The candidate list and
    model settings are identical for both protocols; only inner folds change.
    Metrics pool out-of-fold predictions, weighting each row equally.
    """
    started = time.perf_counter()
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape or not np.isfinite(y).all():
        raise ValueError('Expected finite one-dimensional targets and matching groups')
    dev, test = sanctified_indices(len(y), fraction=holdout_fraction,
                                   seed=seed, groups=groups)
    splitters = {
        'random_rows': kfold_indices(len(dev), k=folds, seed=seed),
        'grouped_entities': grouped_kfold_indices(groups[dev], k=folds, seed=seed),
    }
    report = {
        'description': 'Inner validation audit; final test entities never appear in development',
        'config': {'seed': seed, 'folds': folds, 'holdout_fraction': holdout_fraction,
                   'candidates': list(candidates), 'selection_metric': 'pooled OOF R2'},
        'data': {'rows': len(y), 'entities': len(np.unique(groups)),
                 'dev_rows': len(dev), 'test_rows': len(test),
                 'dev_entities': len(np.unique(groups[dev])),
                 'test_entities': len(np.unique(groups[test]))},
        'environment': {'python': platform.python_version(),
                        **{name: version(name) for name in ('numpy', 'scipy', 'scikit-learn')}},
        'metadata': metadata or {}, 'protocols': {},
    }
    artifacts = {'dev_rows': dev, 'test_rows': test, 'groups': groups.astype(str),
                 'y_test': y[test]}
    for mode, splits in splitters.items():
        validation = {}
        assignment = np.empty(len(dev), dtype=int)
        overlap = []
        for fold, (tr, va) in enumerate(splits):
            assignment[va] = fold
            overlap.append(len(np.intersect1d(groups[dev[tr]], groups[dev[va]])))
        artifacts[mode + '_validation_fold'] = assignment
        for candidate in candidates:
            prediction = np.empty(len(dev))
            for tr, va in splits:
                prediction[va], _ = fit_predict(dev[tr], dev[va], candidate)
            validation[candidate] = metrics(y[dev], prediction)
            artifacts[mode + '_' + candidate + '_oof'] = prediction
        selected = max(candidates, key=lambda c: validation[c]['r2'])
        report['protocols'][mode] = {
            'validation': validation, 'selected': selected,
            'shared_entities_per_inner_fold': overlap,
        }

    # Both choices are frozen before any test score is computed. Include the
    # first candidate as a predeclared baseline, independent of the winners.
    final_candidates = list(dict.fromkeys(
        [candidates[0]] + [p['selected'] for p in report['protocols'].values()]))
    final = {}
    for candidate in final_candidates:
        prediction, selected_features = fit_predict(dev, test, candidate)
        final[candidate] = {**metrics(y[test], prediction),
                            'selected_features': selected_features}
        artifacts[candidate + '_test_predictions'] = prediction
    report['final_test'] = final
    report['baseline'] = candidates[0]
    report['elapsed_seconds'] = time.perf_counter() - started
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB. Leave other platforms unset.
        report['peak_process_rss_mib'] = (
            peak / 1024**2 if sys.platform == 'darwin'
            else peak / 1024 if sys.platform.startswith('linux') else None)
    except ImportError:
        report['peak_process_rss_mib'] = None
    return report, artifacts


def save_report(report, artifacts, output):
    output = Path(output)
    if output.suffix != '.json':
        raise ValueError('Output path must end in .json')
    split_path = output.with_suffix('.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_path, **artifacts)
    report['artifacts'] = {'file': split_path.name, 'sha256': file_hash(split_path)}
    root = Path(__file__).resolve().parents[1]
    source_files = sorted((root / 'examples').rglob('*.py')) + sorted((root / 'unfolds').glob('*.py'))
    report['code_sha256'] = {str(path.relative_to(root)): file_hash(path) for path in source_files}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"{'Inner validation':20s} {'Selected candidate':28s} {'CV R2':>8s} {'Test R2':>8s} {'Test MAE':>9s}")
    for mode, result in report['protocols'].items():
        selected = result['selected']
        cv = result['validation'][selected]
        test = report['final_test'][selected]
        print(f"{mode:20s} {selected:28s} {cv['r2']:8.3f} {test['r2']:8.3f} {test['mae']:9.3f}")
    print(f"Elapsed: {report['elapsed_seconds']:.2f}s; results: {output}; splits/predictions: {split_path}")
