"""Prepare compact DrugComb/GDSC2 inputs without fitting preprocessing.

python -m examples.drugcomb.prepare --download

Uses the three Harvard Dataverse files referenced by TDC's dataset registry.
Checksums pin the inputs used by the original research. Raw upstream pickles
are deserialized only after verification. Outputs contain no pickle objects.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

import numpy as np

from ..validation_audit import file_hash


SOURCES = {
    'drugcomb.pkl': (4215720, 'c0e195c943d5a238e27e0a3e389c4637f9ac0212309c250f9d823b7196ad00bf'),
    'gdsc2.pkl': (4165727, '0d09b9d69cc714fd2b4a31e2b1f602c09ed82c282533683c5fc734ceb226f9e5'),
    'gdsc_gene_symbols.tab': (5255026, '5712acc83f8be3b03345cbd56f5ae3289582ef08caf6991b92eca1728eb73296'),
}

# Original research's explicit DrugComb -> GDSC2 cell-line name corrections.
CELL_ALIASES = {
    'OVCAR3': 'OVCAR-3', 'RXF 393': 'RXF393', 'SF-268': 'SF268',
    'SF-539': 'SF539', 'SW-620': 'SW620', 'T-47D': 'T47D', 'TK-10': 'TK10',
    'UACC62': 'UACC-62', 'COLO 205': 'COLO-205', 'HCC-2998': 'HCC2998',
    'HCT116': 'HCT-116', 'HL-60(TB)': 'HL-60', 'HS 578T': 'Hs-578-T',
    'HT29': 'HT-29', 'IGROV1': 'IGROV-1', 'LOX IMVI': 'LOXIMVI', 'NCIH23': 'NCI-H23',
}


def verified_sources(raw_dir, download=False):
    raw_dir = Path(raw_dir)
    for name, (file_id, expected) in SOURCES.items():
        path = raw_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path}: pass --download or point --raw-dir at a TDC cache')
            raw_dir.mkdir(parents=True, exist_ok=True)
            url = f'https://dataverse.harvard.edu/api/access/datafile/{file_id}'
            print(f'Downloading {name} from Harvard Dataverse...', flush=True)
            temporary = path.with_suffix(path.suffix + '.part')
            try:
                request = Request(url, headers={
                    'User-Agent': 'unfolds/0.1 DrugComb reproducibility example'})
                with urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
                if file_hash(temporary) != expected:
                    raise ValueError(f'Checksum mismatch for {name}; upstream input differs from the pinned dataset')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if file_hash(path) != expected:
            raise ValueError(f'Checksum mismatch for {name}; refusing to load an unverified pickle')
    return {name: raw_dir / name for name in SOURCES}


def build_arrays(combinations, expression, gene_names, descriptor_fn):
    """Keep shared molecular/expression tables; expand rows only during training."""
    required = {'Drug1_ID', 'Drug2_ID', 'Cell_Line_ID', 'Drug1', 'Drug2', 'Synergy_Bliss'}
    if not required.issubset(combinations.columns):
        raise ValueError(f'DrugComb schema missing: {sorted(required - set(combinations.columns))}')
    unique_cells = expression.drop_duplicates('ID2').set_index('ID2')['X2'].to_dict()
    matched = {}
    for name in sorted(combinations['Cell_Line_ID'].unique()):
        gdsc_name = name if name in unique_cells else CELL_ALIASES.get(name)
        if gdsc_name in unique_cells:
            matched[name] = gdsc_name
    if not matched:
        raise ValueError('No cell lines match between DrugComb and GDSC2')
    mask = combinations['Cell_Line_ID'].isin(matched) & np.isfinite(combinations['Synergy_Bliss'])
    frame = combinations.loc[mask]
    cells = sorted(matched)
    cell_index = {name: i for i, name in enumerate(cells)}
    genes = np.stack([unique_cells[matched[name]] for name in cells]).astype(float)
    if genes.shape[1] != len(gene_names):
        raise ValueError('Gene symbol count does not match GDSC2 expression width')

    smiles = {}
    for position in ('Drug1', 'Drug2'):
        for identifier, molecule in frame[[position + '_ID', position]].itertuples(index=False, name=None):
            if identifier in smiles and smiles[identifier] != molecule:
                raise ValueError(f'Conflicting molecular structures for drug {identifier}')
            smiles[identifier] = molecule
    drugs = sorted(smiles)
    descriptor_names, descriptors = descriptor_fn([smiles[name] for name in drugs])
    drug_index = {name: i for i, name in enumerate(drugs)}
    arrays = {
        'drug_descriptors': np.asarray(descriptors, dtype=float),
        'descriptor_names': np.asarray(descriptor_names, dtype=str),
        'gene_expression': genes, 'gene_names': np.asarray(gene_names, dtype=str),
        'drug1_idx': frame['Drug1_ID'].map(drug_index).to_numpy(dtype=int),
        'drug2_idx': frame['Drug2_ID'].map(drug_index).to_numpy(dtype=int),
        'cell_idx': frame['Cell_Line_ID'].map(cell_index).to_numpy(dtype=int),
        'y': frame['Synergy_Bliss'].to_numpy(dtype=float),
        'source_rows': np.flatnonzero(mask.to_numpy()),
        'cell_names': np.asarray(cells), 'gdsc_cell_names': np.asarray([matched[c] for c in cells]),
        'drug_ids': np.asarray(drugs),
    }
    return arrays, {'upstream_rows': len(combinations), 'retained_rows': len(frame),
                    'matched_cells': matched,
                    'unmatched_cells': sorted(set(combinations['Cell_Line_ID']) - set(matched)),
                    'genes_retained': genes.shape[1], 'drugs': len(drugs)}


def rdkit_descriptors(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    names = [name for name, _ in Descriptors.descList]
    matrix = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            raise ValueError(f'RDKit cannot parse a source molecule: {text}')
        values = Descriptors.CalcMolDescriptors(molecule)
        matrix.append([values[name] for name in names])
    return names, np.asarray(matrix, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/drugcomb/raw'))
    parser.add_argument('--output', type=Path, default=Path('data/drugcomb/prepared.npz'))
    parser.add_argument('--download', action='store_true', help='Fetch missing pinned source files (~146 MiB)')
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    started = time.perf_counter()
    paths = verified_sources(args.raw_dir, args.download)
    import pandas as pd

    combos = pd.read_pickle(paths['drugcomb.pkl'])
    expression = pd.read_pickle(paths['gdsc2.pkl'])
    gene_names = pd.read_csv(paths['gdsc_gene_symbols.tab'], sep='\t').to_numpy().ravel()
    arrays, summary = build_arrays(combos, expression, gene_names, rdkit_descriptors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        'format_version': 1, 'target': 'Synergy_Bliss',
        'source_registry': 'https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py',
        'sources': {name: {'url': f'https://dataverse.harvard.edu/api/access/datafile/{fid}',
                           'sha256': sha, 'bytes': paths[name].stat().st_size}
                    for name, (fid, sha) in SOURCES.items()},
        'preparation': 'No gene selection, imputation, scaling, or descriptor filtering fitted here',
        'summary': summary, 'environment': {n: version(n) for n in ('numpy', 'pandas', 'rdkit')},
        'prepared_sha256': file_hash(args.output), 'prepare_code_sha256': file_hash(__file__),
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Prepared {len(arrays['y']):,} rows, {len(arrays['cell_names'])} cell lines, "
          f"{len(gene_names):,} genes in {manifest['elapsed_seconds']:.1f}s")
    print(f'{args.output}: {args.output.stat().st_size / 1024**2:.1f} MiB')


if __name__ == '__main__':
    main()
