import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from examples.metallic_glass.benchmark import chemistry_route, make_research
from examples.metallic_glass.prepare import (
    FEATURE_NAMES, PROPERTIES, canonical_composition, checked_bytes, descriptors,
    prepare_frames,
)

ROOT = Path(__file__).resolve().parents[1]


class MetallicGlassTests(unittest.TestCase):
    def test_composition_order_scaling_and_repeated_symbols_are_equivalent(self):
        first = canonical_composition(["Fe", "B", "Ni"], [60, 20, 20])
        second = canonical_composition(["Ni", "Fe", "B", "Fe"], [.2, .3, .2, .3])
        self.assertEqual(first, second)
        props = {e: {p: (i + 1) * (j + 1.) for j, p in enumerate(PROPERTIES)}
                 for i, e in enumerate(["Fe", "B", "Ni"])}
        assert_array_equal(descriptors(first, props), descriptors(second, props))
        self.assertEqual(descriptors(first, props)[-1], 80.)
        for elements, values in [(["Fe"], [0]), (["Fe"], [-1]), (["Fe"], [])]:
            with self.assertRaises(ValueError):
                canonical_composition(elements, values)

    def test_routing_boundary_uses_only_composition(self):
        X = np.zeros((3, 82))
        X[:, -1] = [29.9, 30, 30.1]
        assert_array_equal(chemistry_route(X, None), ["other", "other", "fe_co_ni_rich"])

    def test_modified_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.zip"
            path.write_bytes(b"unexpected content")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                checked_bytes(path, "0" * 64)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "optional preparation dependency")
    def test_preparation_merges_compositions_and_ignores_noncomposition_columns(self):
        import pandas as pd
        props = pd.DataFrame([
            {"Unnamed: 0": e, **{c: float(i + j + 1)
                                 for j, c in enumerate(PROPERTIES.values())}}
            for i, e in enumerate(["Fe", "B", "Ni"])
        ])
        raw = pd.DataFrame({"Composition": ["Fe B", "B Fe", "Ni B", "Fe B"],
                            "Fraction": ["80 20", "2 8", "80 20", "80 20"],
                            "Tg": [600., 620., 700., np.nan],
                            "Tx": [1000., 1., 999., 2.]})
        X, y, metadata = prepare_frames(raw, props)
        self.assertEqual(X.shape, (2, 82))
        assert_array_equal(y, [610., 700.])
        self.assertEqual(metadata["merged_rows"], 1)
        self.assertEqual(metadata["composition_groups_with_different_targets"], 1)
        raw["Tx"] = -123
        assert_array_equal(prepare_frames(raw, props)[0], X)

    def test_cli_all_models_and_snapshot_reload_in_another_process(self):
        rng = np.random.RandomState(8)
        X = rng.normal(size=(80, 82))
        X[:, -1] = rng.uniform(0, 100, size=80)
        y = 600 + 20 * X[:, 0] + rng.normal(size=80)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path / "features.npz", X=X, y=y, feature_names=FEATURE_NAMES)
            env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1")
            result = subprocess.run(
                [sys.executable, "-m", "examples.metallic_glass.benchmark", "--quick",
                 "--folds", "2", "--data-dir", str(path)],
                cwd=path, env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = path / "artifacts" / "metallic-glass"
            self.assertTrue((saved / "runs.json").exists())
            research = make_research(save_dir=str(saved))
            for name in ["ridge", "flat_ensemble", "chemistry_cascade", "boosted_trees"]:
                with self.subTest(name=name):
                    snapshot = research.load_snapshot(name)
                    self.assertIsNotNone(snapshot["model"])
                    self.assertTrue(np.isfinite(snapshot["model"].predict(X[:5])).all())
                    self.assertTrue(np.isfinite(snapshot["oof_pred"]).all())
                    self.assertEqual(len(snapshot["sanct_pred"]) + len(snapshot["y_dev"]), 80)


if __name__ == "__main__":
    unittest.main()
