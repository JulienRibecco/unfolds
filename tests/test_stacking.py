import unittest

import numpy as np
from numpy.testing import assert_array_equal

from unfolds import StackedModel, expanding_window_indices


class MemorizingModel:
    """Expose in-sample predictions so a stacking leak is observable."""

    def clone(self):
        return MemorizingModel()

    def fit(self, X, y):
        self.X_ = np.array(X, copy=True)
        self.y_ = np.array(y, copy=True)
        self.targets_ = dict(zip(map(tuple, X), y))
        return self

    def predict(self, X):
        return np.array([self.targets_.get(tuple(row), 0.) for row in X])


class StackingTests(unittest.TestCase):
    def test_second_stage_receives_only_out_of_fold_predictions(self):
        X = np.arange(22.).reshape(11, 2)
        y = np.arange(1., 12.)
        model = StackedModel(MemorizingModel(), MemorizingModel(), oof_folds=3)
        model.fit(X, y)
        assert_array_equal(model.second_.X_[:, :-1], X)
        assert_array_equal(model.second_.X_[:, -1], np.zeros(len(y)))
        # The full-data first stage remains available for inference.
        assert_array_equal(model.first_.predict(X), y)

    def test_grouped_stacking_cannot_memorize_entities(self):
        groups = np.tile(np.arange(6), 4)
        X = groups.astype(float).reshape(-1, 1)
        y = groups + 1.
        model = StackedModel(MemorizingModel(), MemorizingModel(), oof_folds=3,
                             oof_strategy='groups')
        model.fit(X, y, groups=groups)
        assert_array_equal(model.second_.X_[:, -1], np.zeros(len(y)))
        assert_array_equal(model.first_.predict(X), y)
        self.assertEqual(model.clone().oof_strategy, 'groups')

    def test_temporal_stacking_uses_past_rows_and_omits_uncovered_targets(self):
        class PastOnly(MemorizingModel):
            def clone(self):
                return PastOnly()

            def predict(self, X):
                if self.X_[:, 0].max() + 2 >= X[:, 0].min():
                    raise AssertionError('inner fit includes future rows or misses the gap')
                return np.full(len(X), self.X_[:, 0].max())

        X = np.arange(24.).reshape(-1, 1)
        y = np.arange(24.) * 2
        model = StackedModel(PastOnly(), MemorizingModel(), oof_folds=3,
                             oof_strategy='temporal', oof_gap=2)
        model.fit(X, y)
        splits = expanding_window_indices(24, k=3, gap=2)
        expected_idx = np.concatenate([te for _, te in splits])
        expected_pred = np.concatenate([np.full(len(te), tr.max()) for tr, te in splits])
        assert_array_equal(model.oof_train_idx_, expected_idx)
        assert_array_equal(model.second_.X_[:, 0], X[expected_idx, 0])
        assert_array_equal(model.second_.X_[:, -1], expected_pred)
        assert_array_equal(model.second_.y_, y[expected_idx])
        assert_array_equal(model.first_.X_, X)
        self.assertEqual(model.clone().oof_gap, 2)

    def test_stacking_rejects_missing_or_inconsistent_split_metadata(self):
        X = np.arange(24.).reshape(12, 2)
        y = np.arange(12.)
        for options, fit_kwargs in (
            ({'oof_strategy': 'groups'}, {}),
            ({'oof_strategy': 'groups'}, {'groups': [0, 1]}),
            ({'oof_strategy': 'rows'}, {'groups': y}),
            ({'oof_strategy': 'other'}, {}),
            ({'oof_gap': 1}, {}),
            ({'oof_strategy': 'temporal', 'oof_folds': 0}, {}),
        ):
            model = StackedModel(MemorizingModel(), MemorizingModel(), **options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                model.fit(X, y, **fit_kwargs)
            self.assertFalse(hasattr(model, 'first_'))
