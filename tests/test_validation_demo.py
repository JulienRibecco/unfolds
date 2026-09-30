import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from examples.entity_validation import make_data, run
from examples.validation_audit import file_hash, ridge_prediction, run_audit, save_report
from examples.drugcomb.audit import run as run_drugcomb, select_genes
from examples.drugcomb.prepare import verified_sources
from unfolds import sanctified_indices


class ValidationDemoTests(unittest.TestCase):
    def test_demo_exposes_entity_shortcut_and_records_disjoint_test(self):
        report, artifacts = run()
        rows = report['protocols']['random_rows']
        grouped = report['protocols']['grouped_entities']
        self.assertEqual(rows['selected'], 'signal_and_fingerprints')
        self.assertEqual(grouped['selected'], 'signal_only')
        self.assertTrue(all(n > 0 for n in rows['shared_entities_per_inner_fold']))
        self.assertEqual(grouped['shared_entities_per_inner_fold'], [0] * 5)
        groups = artifacts['groups']
        self.assertFalse(set(groups[artifacts['dev_rows']]) & set(groups[artifacts['test_rows']]))
        self.assertGreater(report['final_test']['signal_only']['r2'],
                           report['final_test']['signal_and_fingerprints']['r2'])
        # Both final predictions refer to exactly the same target rows.
        for candidate in report['final_test']:
            self.assertEqual(artifacts[candidate + '_test_predictions'].shape,
                             artifacts['y_test'].shape)

    def test_holdout_labels_do_not_influence_selection(self):
        X, original_y, groups, names = make_data()
        _, test = sanctified_indices(len(groups), groups=groups, fraction=0.2)
        reports = []
        for perturb in (False, True):
            y = original_y.copy()
            if perturb:
                y[test] = np.arange(len(test)) * 100.

            def predict(train, evaluation, candidate):
                cols = [0] if candidate == 'signal' else np.arange(X.shape[1])
                return ridge_prediction(X[train][:, cols], y[train], X[evaluation][:, cols],
                                        np.asarray(names)[cols])

            report, _ = run_audit(y, groups, predict, ('signal', 'both'))
            reports.append(report)
        self.assertEqual(reports[0]['protocols'], reports[1]['protocols'])
        self.assertNotEqual(reports[0]['final_test'], reports[1]['final_test'])

    def test_report_round_trip_preserves_split_and_prediction_artifacts(self):
        report, artifacts = run()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            with patch('sys.stdout', new=io.StringIO()):
                save_report(report, artifacts, path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['artifacts']['sha256'], file_hash(path.with_suffix('.npz')))
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
                for key, value in artifacts.items():
                    assert_array_equal(arrays[key], value)


class DrugCombAuditTests(unittest.TestCase):
    def test_gene_selection_ignores_heldout_cells_and_repeated_training_rows(self):
        expression = np.array([[0., 0.], [3., 1.], [0., 1e9]])
        assert_array_equal(select_genes(expression, [0, 1], 1), [0])
        assert_array_equal(select_genes(expression, [0, 0, 0, 1], 1), [0])
        # Show the fixture detects accidental use of the held-out cell.
        assert_array_equal(select_genes(expression, [0, 1, 2], 1), [1])

    def test_prepared_data_runs_without_optional_preparation_dependencies(self):
        rng = np.random.RandomState(19)
        groups = np.repeat(np.arange(15), 10)
        arrays = {
            'cell_idx': groups, 'cell_names': np.array([f'cell{i}' for i in range(15)]),
            'gene_expression': rng.normal(size=(15, 7)),
            'gene_names': np.array([f'g{i}' for i in range(7)]),
            'drug_descriptors': rng.normal(size=(20, 3)),
            'descriptor_names': np.array(['a', 'b', 'c']),
            'drug1_idx': rng.randint(20, size=len(groups)),
            'drug2_idx': rng.randint(20, size=len(groups)),
            'source_rows': np.arange(len(groups)), 'y': rng.normal(size=len(groups)),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prepared.npz'
            np.savez_compressed(path, **arrays)
            path.with_suffix('.json').write_text(json.dumps({
                'format_version': 1, 'prepared_sha256': file_hash(path),
                'sources': {}, 'environment': {}}))
            report, artifacts = run_drugcomb(path, max_rows=0, top_genes=3)
            self.assertEqual(report['data']['rows'], 150)
            self.assertEqual(report['protocols']['grouped_entities']['shared_entities_per_inner_fold'], [0]*3)
            assert_array_equal(artifacts['upstream_rows'], np.arange(150))
            self.assertTrue(all(np.isfinite(v['r2']) for v in report['final_test'].values()))

    def test_preprocessing_does_not_fit_on_evaluation_rows(self):
        train = np.array([[1., 0.], [3., np.nan], [2., 1.]])
        y = np.array([1., 3., 2.])
        pred, _ = ridge_prediction(train, y, np.array([[2., np.nan]]), ['a', 'b'])
        other, _ = ridge_prediction(train, y, np.array([[2., np.nan], [1e9, 1e9]]), ['a', 'b'])
        assert_allclose(pred, other[:1])

    def test_download_and_cache_require_pinned_checksum(self):
        body = b'gene\nA\nB\n'
        source = {'genes.tab': (123, hashlib.sha256(body).hexdigest())}
        with tempfile.TemporaryDirectory() as directory, \
             patch('examples.drugcomb.prepare.SOURCES', source), \
             patch('examples.drugcomb.prepare.urlopen', return_value=io.BytesIO(body)) as request:
            with patch('sys.stdout', new=io.StringIO()):
                paths = verified_sources(directory, download=True)
            request.assert_called_once()
            verified_sources(directory)
            self.assertEqual(request.call_count, 1)
            paths['genes.tab'].write_bytes(b'wrong file')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                verified_sources(directory)
