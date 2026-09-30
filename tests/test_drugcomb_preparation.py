import importlib.util
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from examples.drugcomb.prepare import build_arrays, rdkit_descriptors


@unittest.skipUnless(importlib.util.find_spec('pandas'), 'optional DrugComb preparation dependencies')
class PreparationTests(unittest.TestCase):
    def test_preparation_preserves_all_genes_and_row_alignment(self):
        import pandas as pd

        combinations = pd.DataFrame({
            'Drug1_ID': ['b', 'a', 'a'], 'Drug2_ID': ['a', 'b', 'b'],
            'Drug1': ['CC', 'C', 'C'], 'Drug2': ['C', 'CC', 'CC'],
            'Cell_Line_ID': ['OVCAR3', 'missing', 'A'],
            'Synergy_Bliss': [1., 2., 3.],
        })
        expression = pd.DataFrame({
            'ID2': ['OVCAR-3', 'A'],
            'X2': [np.array([0., 1e9, 7.]), np.array([1., 0., 7.])],
        })

        def descriptors(smiles):
            return ['length', 'constant'], [[len(s), 1.] for s in smiles]

        arrays, summary = build_arrays(combinations, expression, ['g0', 'g1', 'constant'], descriptors)
        self.assertEqual(summary['retained_rows'], 2)
        self.assertEqual(summary['unmatched_cells'], ['missing'])
        assert_array_equal(arrays['source_rows'], [0, 2])
        assert_array_equal(arrays['y'], [1., 3.])
        # No global variance selection or constant-column removal at preparation.
        assert_array_equal(arrays['gene_expression'][arrays['cell_idx']],
                           [[0., 1e9, 7.], [1., 0., 7.]])
        assert_array_equal(arrays['drug_descriptors'][arrays['drug1_idx']], [[2., 1.], [1., 1.]])
        assert_array_equal(arrays['drug_descriptors'][arrays['drug2_idx']], [[1., 1.], [2., 1.]])

    @unittest.skipUnless(importlib.util.find_spec('rdkit'), 'optional RDKit dependency')
    def test_rdkit_descriptors_are_per_molecule_and_order_independent(self):
        names, together = rdkit_descriptors(['C', 'CC'])
        other_names, separate = rdkit_descriptors(['CC'])
        self.assertEqual(names, other_names)
        assert_array_equal(together[1], separate[0])
        self.assertGreater(together[1, names.index('MolWt')], together[0, names.index('MolWt')])
