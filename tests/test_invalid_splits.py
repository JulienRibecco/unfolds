import unittest

import numpy as np

from unfolds import (
    Experiment, ExperimentConfig, NLModel, SanctifiedDataset, StackedModel,
    expanding_window_indices, grouped_kfold_indices, kfold_indices,
    oof_indices, oof_splits, sanctified_indices, temporal_split_indices,
    train_test_indices,
)


class InvalidSplitTests(unittest.TestCase):
    def test_sample_counts_must_be_integers_at_least_two(self):
        splitters = (kfold_indices, oof_indices, oof_splits, train_test_indices,
                     sanctified_indices, temporal_split_indices, expanding_window_indices)
        for splitter in splitters:
            for n in (-1, 0, 1, 5.5, True):
                with self.subTest(splitter=splitter.__name__, n=n):
                    with self.assertRaisesRegex(ValueError, 'n must'):
                        list(splitter(n))

    def test_invalid_fold_counts_are_rejected(self):
        for splitter in (kfold_indices, oof_indices, oof_splits):
            for k in (-1, 0, 1, 6, 2.5, True):
                with self.subTest(splitter=splitter.__name__, k=k):
                    with self.assertRaises(ValueError):
                        list(splitter(5, k))

    def test_group_count_limits_number_of_folds(self):
        groups = np.repeat(['a', 'b', 'c'], 4)
        for k in (-1, 0, 1, 4, 2.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                grouped_kfold_indices(groups, k=k)
        for groups in ([], ['a', 'a'], [['a'], ['b']]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                grouped_kfold_indices(groups, k=2)

    def test_invalid_fractions_are_rejected(self):
        for splitter in (train_test_indices, sanctified_indices, temporal_split_indices):
            for fraction in (-0.1, 0., 1., 1.1, np.nan, np.inf, -np.inf, True, '0.2'):
                with self.subTest(splitter=splitter.__name__, fraction=fraction):
                    with self.assertRaisesRegex(ValueError, 'strictly between 0 and 1'):
                        splitter(20, fraction)

    def test_random_holdout_rejects_rounding_to_empty_split(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            train_test_indices(2, test_fraction=0.9)

    def test_grouped_holdout_requires_multiple_groups_and_matching_labels(self):
        for groups in (['a'] * 4, ['a', 'b'], [['a'], ['b'], ['c'], ['d']]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                sanctified_indices(4, groups=groups)

    def test_stratification_requires_matching_labels_and_a_possible_holdout(self):
        for strata in ([0, 1], [[0], [0], [1], [1]]):
            with self.subTest(strata=strata), self.assertRaisesRegex(ValueError, 'length 4'):
                sanctified_indices(4, stratify=strata)
        with self.assertRaisesRegex(ValueError, 'singleton'):
            sanctified_indices(4, stratify=[0, 1, 2, 3])
        train, test = sanctified_indices(4, stratify=[0, 0, 1, 2])
        self.assertTrue({2, 3}.issubset(set(train)))
        self.assertEqual(len(test), 1)

    def test_temporal_configuration_cannot_drop_requested_folds(self):
        for k in (-1, 0, 5, 2.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                expanding_window_indices(5, k=k)
        for gap in (-1, 0.5, True):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                expanding_window_indices(11, k=3, gap=gap)
        # Previously the first two folds were silently skipped, leaving one.
        with self.assertRaisesRegex(ValueError, 'leaves fold 0 empty'):
            expanding_window_indices(11, k=3, gap=2)
        self.assertEqual(len(expanding_window_indices(11, k=3, gap=1)), 3)
        self.assertEqual(len(expanding_window_indices(3, k=1, gap=1)), 1)

    def test_numpy_scalar_configuration_is_accepted(self):
        self.assertEqual(len(kfold_indices(np.int64(6), np.int64(3))), 3)
        train, test = train_test_indices(np.int64(6), np.float64(0.5))
        self.assertEqual((len(train), len(test)), (3, 3))

    def test_experiment_surfaces_invalid_configuration(self):
        rng = np.random.RandomState(12)
        X = rng.normal(size=(40, 3))
        y = rng.normal(size=40)
        ds = SanctifiedDataset(X, y, stratify=False)
        exp = Experiment(ds, ExperimentConfig(k=100))
        with self.assertRaises(ValueError):
            list(exp.folds())
        with self.assertRaises(ValueError):
            list(exp.holdout(fraction=1.))
        with self.assertRaisesRegex(ValueError, 'group_by'):
            list(exp.folds(k=3, group_by='gruops'))
        for gap in (-1, 1.5, True):
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, 'temporal_gap'):
                SanctifiedDataset(X, y, temporal=True, temporal_gap=gap)

    def test_stacking_rejects_invalid_inner_folds_before_fitting(self):
        X = np.arange(12.).reshape(6, 2)
        y = np.arange(6.)
        for count in (-1, 1, 7, 2.5, True):
            model = StackedModel(NLModel(epochs=2), NLModel(epochs=2), oof_folds=count)
            with self.subTest(count=count), self.assertRaises(ValueError):
                model.fit(X, y)
            self.assertFalse(hasattr(model, 'first_'))
        # Keep the documented opt-out for research workflows.
        model = StackedModel(NLModel(epochs=2), NLModel(epochs=2), oof_folds=0)
        model.fit(X, y)
        self.assertTrue(np.isfinite(model.predict(X)).all())
