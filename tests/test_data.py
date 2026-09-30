import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from unfolds import (
    expanding_window_indices, fold_impute, fold_normalize,
    fold_safe_preprocess, grouped_kfold_indices, kfold_indices,
    oof_indices, sanctified_indices, temporal_split_indices,
)


class SplitTests(unittest.TestCase):
    def assert_partition(self, train, test, n):
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertEqual(len(np.intersect1d(train, test)), 0)
        assert_array_equal(np.sort(np.concatenate([train, test])), np.arange(n))

    def test_folds_partition_samples_and_cover_validation_once(self):
        for split_fn in (kfold_indices, oof_indices):
            with self.subTest(split_fn=split_fn.__name__):
                folds = list(split_fn(23, 5))
                self.assertEqual(len(folds), 5)
                for train, test in folds:
                    self.assert_partition(train, test, 23)
                assert_array_equal(
                    np.sort(np.concatenate([test for _, test in folds])),
                    np.arange(23))

    def test_random_folds_are_reproducible(self):
        first = kfold_indices(23, seed=7)
        second = kfold_indices(23, seed=7)
        for (tr1, te1), (tr2, te2) in zip(first, second):
            assert_array_equal(tr1, tr2)
            assert_array_equal(te1, te2)

    def test_groups_stay_together_in_folds_and_holdout(self):
        groups = np.repeat(np.arange(6), [1, 2, 3, 4, 5, 6])
        splits = grouped_kfold_indices(groups, k=3)
        splits.append(sanctified_indices(len(groups), groups=groups))
        for train, test in splits:
            self.assert_partition(train, test, len(groups))
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_temporal_order_and_gap(self):
        train, test = temporal_split_indices(23)
        self.assert_partition(train, test, 23)
        self.assertLess(train.max(), test.min())
        folds = expanding_window_indices(23, k=3, gap=2)
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            self.assertEqual(test.min() - train.max() - 1, 2)

    def test_temporal_holdout_excludes_gap_without_moving_the_test_boundary(self):
        baseline_train, baseline_test = temporal_split_indices(23)
        train, test = temporal_split_indices(23, gap=3)
        assert_array_equal(test, baseline_test)
        assert_array_equal(train, baseline_train[:-3])
        for gap in (-1, True, 0.5, 22):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                temporal_split_indices(23, gap=gap)


class PreprocessingTests(unittest.TestCase):
    def test_normalization_uses_training_statistics_without_mutating_inputs(self):
        train = np.array([[1., 7.], [3., 7.]])
        test = np.array([[101., 9.]])
        original_train, original_test = train.copy(), test.copy()
        normalized, validation, means, stds = fold_normalize(train, test)
        assert_allclose(means, [2., 7.])
        assert_allclose(stds, [1., 1.])
        assert_allclose(normalized, [[-1., 0.], [1., 0.]])
        assert_allclose(validation, [[99., 2.]])
        assert_array_equal(train, original_train)
        assert_array_equal(test, original_test)

    def test_imputation_ignores_validation_values_and_preserves_input(self):
        X = np.array([[1., 2.], [3., np.nan], [1000., 1000.], [np.nan, np.nan]])
        original = X.copy()
        train, test, medians = fold_impute(X, [0, 1], [2, 3])
        assert_allclose(medians, [2., 2.])
        assert_allclose(train, [[1., 2.], [3., 2.]])
        assert_allclose(test, [[1000., 1000.], [2., 2.]])
        assert_array_equal(X, original)

    def test_imputation_rejects_columns_missing_only_in_training(self):
        X = np.array([[1., np.nan], [2., np.nan], [3., 100.], [4., 200.]])
        for evaluate in (
            lambda: fold_impute(X, [0, 1], [2, 3]),
            lambda: fold_safe_preprocess(X[:2], X[2:]),
        ):
            with self.assertRaisesRegex(ValueError, r'all-NaN training columns: \[1\]'):
                evaluate()

    def test_preprocessing_calling_conventions_agree(self):
        X = np.array([[1., 2.], [3., np.nan], [1000., np.nan]])
        sliced = fold_safe_preprocess(X[:2], X[2:])
        indexed = fold_safe_preprocess(
            None, full_X=X, tr_idx=np.array([0, 1]), te_idx=np.array([2]))
        for left, right in zip(sliced[:2], indexed[:2]):
            assert_allclose(left, right)
        for key in ('medians', 'means', 'stds'):
            assert_allclose(sliced[2][key], indexed[2][key])
