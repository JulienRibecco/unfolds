import unittest
import contextlib
import io

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sklearn.linear_model import Ridge

from unfolds import Experiment, ExperimentConfig, Research, SanctifiedDataset


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(17)
        self.X = rng.normal(size=(80, 3))
        self.y = self.X[:, 0] * 2 + self.X[:, 1]
        self.ds = SanctifiedDataset(self.X, self.y, stratify=False)
        self.exp = Experiment(self.ds, ExperimentConfig(k=3))

    def test_folds_only_use_dev_data_and_are_read_only(self):
        validation_indices = []
        for fold in self.exp.folds():
            assert_array_equal(fold.X_train, self.ds.X[fold.tr_idx])
            assert_array_equal(fold.X_val, self.ds.X[fold.te_idx])
            for array in (fold.X_train, fold.X_val, fold.y_train, fold.y_val):
                with self.assertRaises(ValueError):
                    array.flat[0] = 999
            validation_indices.extend(fold.te_idx)
        assert_array_equal(np.sort(validation_indices), np.arange(len(self.ds.X)))
        final = self.exp.final_evaluate()
        self.assertEqual(len(np.intersect1d(final.dev_idx, final.sanct_idx)), 0)
        assert_array_equal(final.X_sanct, self.X[final.sanct_idx])

    def test_holdout_access_guard_is_shared_across_experiments(self):
        other = Experiment(self.ds)
        with self.assertRaises(RuntimeError):
            self.ds.X_sanct
        with self.assertRaises(RuntimeError):
            self.ds.y_sanct
        with self.assertRaises(RuntimeError):
            self.exp.record_final(np.zeros(12))
        final = self.exp.final_evaluate()
        for array in (final.X_dev, final.y_dev, final.X_sanct, final.y_sanct):
            with self.assertRaises(ValueError):
                array.flat[0] = 999
        for access in (self.exp.final_evaluate, other.final_evaluate, self.ds.final_evaluate):
            with self.assertRaises(RuntimeError):
                access()
        with self.assertRaises(RuntimeError):
            list(self.exp.folds())

    def test_full_lifecycle_records_correct_metrics(self):
        for fold in self.exp.folds():
            with self.assertRaises(ValueError):
                fold.record(np.zeros(len(fold.y_val) + 1))
            fold.record(fold.X_val[:, 0] * 2 + fold.X_val[:, 1])
        final = self.exp.final_evaluate()
        with self.assertRaises(ValueError):
            self.exp.record_final(np.zeros(len(final.y_sanct) + 1))
        self.exp.record_final(final.X_sanct[:, 0] * 2 + final.X_sanct[:, 1])
        recap = self.exp.recap(verbose=False)
        assert_allclose(recap['dev']['mae']['mean'], 0.)
        assert_allclose(recap['sanctified']['mae'], 0.)
        assert_allclose(recap['sanctified']['r2'], 1.)

    def test_configured_grouping_applies_to_folds_holdout_and_run(self):
        labels = np.repeat(np.arange(8), 10)
        for grouping in ('groups', 'source'):
            for strategy in ('kfold', 'holdout'):
                with self.subTest(grouping=grouping, strategy=strategy):
                    ds = SanctifiedDataset(self.X, self.y, groups=labels,
                                           source_ids=labels, stratify=False)
                    config = ExperimentConfig(k=3, group_by=grouping, dev_strategy=strategy)
                    exp = Experiment(ds, config)
                    for fold in list(exp.folds()) + list(exp.holdout()):
                        self.assertFalse(set(ds.groups[fold.tr_idx]) & set(ds.groups[fold.te_idx]))
                    exp.run(lambda: Ridge(), verbose=False)
                    self.assertEqual(len(exp._folds), 1 if strategy == 'holdout' else 3)
                    for fold in exp._folds:
                        self.assertFalse(set(ds.groups[fold.tr_idx]) & set(ds.groups[fold.te_idx]))

    def test_explicit_none_overrides_configured_grouping(self):
        labels = np.repeat(np.arange(8), 10)
        ds = SanctifiedDataset(self.X, self.y, groups=labels, stratify=False)
        exp = Experiment(ds, ExperimentConfig(k=3, group_by='groups'))
        for fold in (next(exp.folds(group_by=None)), next(exp.holdout(group_by=None))):
            self.assertTrue(set(ds.groups[fold.tr_idx]) & set(ds.groups[fold.te_idx]))

    def test_temporal_gap_applies_to_every_evaluation_boundary(self):
        ds = SanctifiedDataset(self.X, self.y, temporal=True, temporal_gap=4)
        exp = Experiment(ds, ExperimentConfig(k=3))
        for fold in list(exp.folds()) + list(exp.holdout()):
            self.assertEqual(fold.te_idx.min() - fold.tr_idx.max() - 1, 4)
        final = exp.final_evaluate()
        self.assertEqual(final.sanct_idx.min() - final.dev_idx.max() - 1, 4)
        self.assertEqual(len(final.X_dev) + len(final.X_sanct) + 4, len(self.X))

    def test_research_accepts_unnamed_features_and_preserves_holdout_groups(self):
        labels = np.repeat(np.arange(8), 10)
        ds = SanctifiedDataset(self.X, self.y, groups=labels, stratify=False)
        contexts = []
        def factory(ctx):
            contexts.append(ctx)
            return Ridge()
        research = Research('unnamed', lambda path, cfg: ds,
                            ExperimentConfig(group_by='groups', dev_strategy='holdout'))
        research.new_experiment('ridge', factory)
        # Capture the development split that the real bench invokes.
        original = ds.dev_holdout_indices
        def capture(*args, **kwargs):
            tr, te = original(*args, **kwargs)
            self.assertFalse(set(ds.groups[tr]) & set(ds.groups[te]))
            return tr, te
        ds.dev_holdout_indices = capture
        with contextlib.redirect_stdout(io.StringIO()):
            result = research.run()
        self.assertEqual(len(contexts), 2)
        self.assertIsNone(contexts[0].feature_names)
        self.assertTrue(np.isfinite(result['ridge']['sanctified']['mae']))
