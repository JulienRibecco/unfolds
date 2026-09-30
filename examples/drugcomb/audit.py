"""Real-data split audit, using ridge and optional training-selected genes.

python -m examples.drugcomb.audit --data data/drugcomb/prepared.npz

This is a new controlled audit on the original data sources. It does not
reproduce the historical Soup feature-engineering recipe or its scores.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..validation_audit import file_hash, ridge_prediction, run_audit, save_report


def select_genes(expression, training_cells, count):
    """Rank genes using distinct training cell lines, never held-out lines."""
    values = expression[np.unique(training_cells)]
    valid = np.flatnonzero(np.isfinite(values).all(axis=0))
    variance = values[:, valid].var(axis=0)
    valid, variance = valid[variance > 1e-12], variance[variance > 1e-12]
    if len(valid) == 0:
        raise ValueError('No varying finite genes in training cell lines')
    return valid[np.argsort(-variance, kind='stable')[:count]]


def run(data_path, *, seed=42, max_rows=5000, top_genes=200, folds=3):
    if max_rows < 0 or top_genes < 1:
        raise ValueError('max_rows must be nonnegative and top_genes must be positive')
    data_path = Path(data_path)
    manifest = json.loads(data_path.with_suffix('.json').read_text())
    if manifest.get('format_version') != 1 or manifest['prepared_sha256'] != file_hash(data_path):
        raise ValueError('Prepared data does not match its manifest; rerun preparation')
    with np.load(data_path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    # Sample whole row indices deterministically; never tune the subset on y.
    chosen = np.arange(len(data['y']))
    if max_rows and max_rows < len(chosen):
        chosen = np.sort(np.random.RandomState(seed).choice(chosen, max_rows, replace=False))
    y, groups = data['y'][chosen], data['cell_idx'][chosen]
    descriptor_names = ([f'drug1_{n}' for n in data['descriptor_names']]
                        + [f'drug2_{n}' for n in data['descriptor_names']])

    def fit_predict(train, evaluation, candidate):
        def features(rows, genes):
            rows = chosen[rows]
            parts = [data['drug_descriptors'][data['drug1_idx'][rows]],
                     data['drug_descriptors'][data['drug2_idx'][rows]]]
            if genes is not None:
                parts.append(data['gene_expression'][np.ix_(data['cell_idx'][rows], genes)])
            return np.column_stack(parts)

        genes = None
        names = list(descriptor_names)
        if candidate == 'drug_and_genes':
            genes = select_genes(data['gene_expression'], groups[train], top_genes)
            names += [f'gene_{n}' for n in data['gene_names'][genes]]
        return ridge_prediction(features(train, genes), y[train],
                                features(evaluation, genes), names, alpha=10.0)

    report, artifacts = run_audit(
        y, groups, fit_predict, ('drug_only', 'drug_and_genes'), seed=seed, folds=folds,
        metadata={'dataset': 'DrugComb + GDSC2', 'prepared_sha256': manifest['prepared_sha256'],
                  'source_sha256': {n: s['sha256'] for n, s in manifest['sources'].items()},
                  'max_rows': max_rows, 'top_genes': top_genes, 'alpha': 10.0,
                  'preparation_environment': manifest['environment'],
                  'scope': 'New ridge feature-set audit, not historical Soup reproduction'})
    artifacts['prepared_rows'] = chosen
    artifacts['upstream_rows'] = data['source_rows'][chosen]
    report['data']['entity_names'] = data['cell_names'].tolist()
    return report, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/drugcomb/prepared.npz'))
    parser.add_argument('--max-rows', type=int, default=5000, help='0 for all matched rows')
    parser.add_argument('--top-genes', type=int, default=200)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='artifacts/drugcomb-audit.json')
    args = parser.parse_args()
    report, artifacts = run(args.data, seed=args.seed, max_rows=args.max_rows,
                            top_genes=args.top_genes, folds=args.folds)
    save_report(report, artifacts, args.output)


if __name__ == '__main__':
    main()
