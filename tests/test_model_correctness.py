import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone, is_regressor
from sklearn.ensemble import VotingRegressor

from unfolds import (
    BinConfig, CascadeConfig, EnsembleModel, NLModel, RoutedModel,
    StackedModel, build_cascade, predict_nl, train_nl,
)


class ModelCorrectnessTests(unittest.TestCase):
    def test_nn_gradients_match_finite_differences_in_every_layer(self):
        rng = np.random.RandomState(8)
        X, y = rng.normal(size=(5, 2)), np.array([-8., -1., 0., 2., 9.])
        initial = train_nl(X, y, [3, 2], 0, seed=7)
        lr, epsilon = 1e-3, 1e-6
        trained = train_nl(X, y, [3, 2], 1, seed=7, lr=lr, l2=0,
                           huber_delta=2., clip_grad=1e6, momentum=0)

        def loss():
            error = np.abs(predict_nl(X, initial) - y)
            return np.where(error <= 2., 0.5 * error**2, 2. * (error - 1.)).mean()

        for kind in ('W', 'b'):
            for layer, parameter in enumerate(initial[kind]):
                numerical = np.empty_like(parameter)
                for index in np.ndindex(parameter.shape):
                    original = parameter[index]
                    parameter[index] = original + epsilon
                    plus = loss()
                    parameter[index] = original - epsilon
                    minus = loss()
                    parameter[index] = original
                    numerical[index] = (plus - minus) / (2 * epsilon)
                actual = (parameter - trained[kind][layer]) / lr
                assert_allclose(actual, numerical, atol=1e-8, rtol=1e-5)

    def test_nn_update_is_invariant_to_duplicating_the_batch(self):
        X = np.array([[0., 1.], [2., 3.], [-1., 0.]])
        y = np.array([2., 3., -1.])
        params = [train_nl(np.tile(X, (copies, 1)), np.tile(y, copies),
                           [3], 1, seed=7, l2=0, momentum=0)
                  for copies in (1, 5)]
        for kind in ('W', 'b'):
            for left, right in zip(params[0][kind], params[1][kind]):
                assert_allclose(left, right, atol=1e-15)

    def test_sklearn_recognizes_and_composes_regressors(self):
        model = NLModel(hidden_sizes=(2,), epochs=2)
        self.assertTrue(is_regressor(model))
        assert clone(model).get_params() == model.get_params()
        X = np.arange(20.).reshape(10, 2)
        ensemble = VotingRegressor([('nn', model)]).fit(X, np.arange(10.))
        self.assertEqual(ensemble.predict(X).shape, (10,))

    def test_cascade_settings_reach_nested_templates_without_mutating_them(self):
        router = EnsembleModel(base=NLModel(epochs=99), n_seeds=2, base_seed=42)
        expert = StackedModel(NLModel(epochs=88), NLModel(epochs=77), oof_folds=2)
        config = CascadeConfig(router, [], [BinConfig('all', expert, (-np.inf, np.inf))],
                               epochs=2)
        model = build_cascade(config, seed=7)
        self.assertEqual((model.router.base.epochs, model.router.base_seed), (2, 7))
        self.assertEqual((model.experts['all'].first.epochs,
                          model.experts['all'].second.epochs), (2, 2))
        self.assertEqual((model.experts['all'].first.seed,
                          model.experts['all'].second.seed), (8, 9))
        self.assertEqual((router.base.epochs, router.base_seed), (99, 42))
        self.assertEqual(expert.first.epochs, 88)
        X = np.arange(24.).reshape(12, 2)
        model.fit(X, np.arange(12.))
        self.assertEqual([m.seed for m in model.router_.models_], [7, 8])
        heterogeneous = EnsembleModel(models=[NLModel(epochs=99), NLModel(epochs=88)])
        config = CascadeConfig(heterogeneous, [], [BinConfig('all', NLModel(), (-1, 20))],
                               epochs=3)
        model = build_cascade(config, seed=17)
        self.assertEqual([(m.epochs, m.seed) for m in model.router.models], [(3, 17)] * 2)

    def test_routing_rejects_unknown_keys_and_wrong_shapes(self):
        model = RoutedModel(None, {'known': NLModel(epochs=1)},
                            lambda X, _: ['known' if x < 5 else 'missing' for x in X[:, 0]])
        X = np.arange(4.).reshape(-1, 1)
        model.fit(X, np.arange(4.))
        self.assertEqual(model.predict(X).shape, (4,))
        for action in (model.predict, model.routing_distribution):
            with self.assertRaisesRegex(ValueError, 'unknown expert keys'):
                action(np.array([[9.]]))
        with self.assertRaisesRegex(ValueError, 'unknown expert keys'):
            model.clone().fit(np.array([[9.], [10.]]), np.array([1., 2.]))
        model.route_fn = lambda X, _: 'known'
        with self.assertRaisesRegex(ValueError, 'one key per input row'):
            model.predict(X)
