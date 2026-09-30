import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from examples import california_housing
from unfolds import (BinConfig, CascadeConfig, Experiment, ExperimentConfig,
                     NLModel, SanctifiedDataset, StackedModel, build_cascade)


ROOT = Path(__file__).resolve().parents[1]


class ExampleTests(unittest.TestCase):
    def test_readme_grouped_stacking_completes_the_documented_lifecycle(self):
        readme = (ROOT / 'README.md').read_text()
        section = readme.split('training fold:\n', 1)[1]
        code = re.search(r'```python\n(.*?)```', section, re.S).group(1)
        rng = np.random.RandomState(27)
        namespace = dict(
            X=rng.normal(size=(80, 3)), y=rng.normal(size=80),
            groups=np.repeat(np.arange(8), 10),
            SanctifiedDataset=SanctifiedDataset, Experiment=Experiment,
            ExperimentConfig=ExperimentConfig, StackedModel=StackedModel,
            NLModel=lambda *args: NLModel(*args, epochs=2))
        exec(code, namespace)
        recap = namespace['exp'].recap(verbose=False)
        self.assertTrue(np.isfinite(recap['sanctified']['mae']))
        for fold in namespace['exp']._folds:
            labels = namespace['sd'].groups
            self.assertFalse(set(labels[fold.tr_idx]) & set(labels[fold.te_idx]))

    def test_readme_model_composables_fit_and_predict(self):
        readme = (ROOT / 'README.md').read_text()
        section = readme.split('## Model composables', 1)[1]
        code = re.search(r'```python\n(.*?)```', section, re.S).group(1)
        namespace = {}

        def small_model(*args, **kwargs):
            kwargs['epochs'] = 3
            return NLModel(*args, **kwargs)

        # Run the actual documented constructors, with a smaller training budget.
        with patch('unfolds.NLModel', side_effect=small_model):
            exec(code, namespace)
        rng = np.random.RandomState(23)
        X = rng.normal(size=(30, 3))
        y = X[:, 0] + 2
        for name in ('ensemble', 'stacked', 'routed'):
            with self.subTest(model=name):
                model = namespace[name]
                model.fit(X, y)
                prediction = model.predict(X[:7])
                self.assertEqual(prediction.shape, (7,))
                self.assertTrue(np.isfinite(prediction).all())
        assert_array_equal(
            namespace['routed'].route_fn(X[:3], np.array([49., 50., 51.])),
            ['low', 'high', 'high'])

    def test_cascade_snapshot_round_trip_in_fresh_process(self):
        config = CascadeConfig(
            router=NLModel((2,)), thresholds=[1., 3.], epochs=3,
            bins=[BinConfig(name, NLModel((2,)), train=(0., 5.))
                  for name in ('low', 'mid', 'high')], min_samples=2)
        model = build_cascade(config, seed=7)
        boundary = np.array([0., 1., 2., 3., 4.])
        assert_array_equal(model.route_fn(None, boundary),
                           ['low', 'mid', 'mid', 'high', 'high'])
        # Both unfitted templates and trained models should be serializable.
        model = pickle.loads(pickle.dumps(model))
        rng = np.random.RandomState(4)
        X = rng.normal(size=(30, 3))
        model.fit(X, rng.uniform(0., 5., size=30))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'model.pkl').write_bytes(pickle.dumps(model))
            np.save(directory / 'X.npy', X)
            subprocess.run([
                sys.executable, '-c',
                "import pickle, sys, numpy as np; from pathlib import Path; "
                "p = Path(sys.argv[1]); "
                "m = pickle.loads((p / 'model.pkl').read_bytes()); "
                "np.save(p / 'prediction.npy', m.predict(np.load(p / 'X.npy')))",
                str(directory),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            assert_allclose(np.load(directory / 'prediction.npy'), model.predict(X))

    def test_california_cli_runs_all_models_and_saves_snapshots_offline(self):
        rng = np.random.RandomState(42)
        data = SimpleNamespace(
            data=rng.normal(size=(80, 8)),
            target=rng.uniform(0.2, 5., size=80),
            feature_names=[f'x{i}' for i in range(8)])
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.object(california_housing, 'fetch_california_housing', return_value=data), \
                     patch.object(sys, 'argv', ['california_housing.py', '--quick', '--folds', '2']), \
                     contextlib.redirect_stdout(io.StringIO()):
                    california_housing.main()
                saved = Path('california_runs')
                self.assertEqual(len(json.loads((saved / 'runs.json').read_text())), 1)
                for name in ('flat_ridge', 'small_nn', 'ensemble', 'stacked', 'cascade'):
                    with self.subTest(model=name):
                        model = pickle.loads((saved / 'snapshots' / f'{name}_model.pkl').read_bytes())
                        prediction = model.predict(data.data[:5])
                        self.assertEqual(prediction.shape, (5,))
                        self.assertTrue(np.isfinite(prediction).all())
                        with np.load(saved / 'snapshots' / f'{name}.npz') as arrays:
                            self.assertTrue(np.isfinite(arrays['oof_pred']).all())
                            self.assertTrue(np.isfinite(arrays['sanct_pred']).all())
            finally:
                os.chdir(previous_cwd)
