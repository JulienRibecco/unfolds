"""Download pinned source workbooks and prepare composition-only descriptors."""

import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

REVISION = "54bceddee4980217ea55998da0ee56272a81b1bf"
SOURCE = "https://github.com/ZHOU-Ziqing/ML_Metallicglass_GFA"
URL = ("https://raw.githubusercontent.com/ZHOU-Ziqing/ML_Metallicglass_GFA/"
       + REVISION + "/descriptors.zip")
ARCHIVE_SHA256 = "d40e8f8d490af41b4daa7d81f87ad7085ef3e96b1e363100774365a743bfc129"
WORKBOOKS = {
    "descriptors/Final_dataset_all_2.xlsx":
        "0051284814a41f281f2635d1421551802624e346e0af9a35548b5c7b3b6a4f55",
    "descriptors/properties.xlsx":
        "41c9286e37236ec0af6f8996a95b5b019235e30d3d16d683ef96e04c20cb289a",
}
DEFAULT_DATA_DIR = Path("data/metallic-glass")
PROPERTIES = {
    "atomic_size": "atomic size (Ȧ)",
    "Tm": "Tm (K)",
    "electronegativity": "Pauling electronegativity",
    "VEC": "VEC",
    "Youngs_modulus": "Young's modulus (GPa)",
    "Bulk_modulus": "Bulk modulus (GPa)",
    "Density": "Density(kg/m^3)",
    "atomic_mass": "atomic mass",
}
STATS = ("mean", "wtd_mean", "gmean", "wtd_gmean", "entropy",
         "wtd_entropy", "range", "wtd_range", "std", "wtd_std")
FEATURE_NAMES = (["number_of_elements"]
                 + [f"{p}_{s}" for p in PROPERTIES for s in STATS]
                 + ["mag_fraction"])


def canonical_composition(elements, amounts):
    """Exact rational fractions make order and percentage scaling irrelevant."""
    if not elements or len(elements) != len(amounts):
        raise ValueError("Composition and fraction lengths must match and be nonempty")
    totals = {}
    for element, amount in zip(elements, amounts):
        value = Fraction(str(amount))
        if value < 0:
            raise ValueError("Composition fractions must be nonnegative")
        totals[element] = totals.get(element, Fraction(0)) + value
    total = sum(totals.values())
    if total <= 0:
        raise ValueError("Composition must have a positive total")
    return tuple((e, v / total) for e, v in sorted(totals.items()) if v)


def descriptors(composition, properties):
    """81 statistical descriptors plus the Fe/Co/Ni atomic percentage."""
    elements, weights = zip(*composition)
    weights = np.asarray(weights, dtype=float)
    result = [len(elements)]
    for prop in PROPERTIES:
        values = np.array([properties[e][prop] for e in elements])
        weighted_mean = weights @ values
        safe = np.abs(values) + 1e-10
        probability = safe / safe.sum()
        result.extend([
            values.mean(), weighted_mean,
            np.exp(np.log(safe).mean()), np.exp(weights @ np.log(safe)),
            -np.sum(probability * np.log(probability)),
            -np.sum(weights * probability * np.log(probability)),
            np.ptp(values), np.ptp(weights * values), values.std(),
            np.sqrt(weights @ ((values - weighted_mean) ** 2)),
        ])
    result.append(100 * sum(w for e, w in composition if e in {"Fe", "Co", "Ni"}))
    return np.asarray(result, dtype=float)


def prepare_frames(data, property_table):
    """Merge repeated compositions before splitting; average repeated Tg values."""
    properties = {}
    for _, row in property_table.dropna(subset=["Unnamed: 0"]).iterrows():
        properties[str(row["Unnamed: 0"]).strip()] = {
            p: float(row[column]) if not np.isnan(row[column]) else 0.0
            for p, column in PROPERTIES.items()
        }
    # Correction retained from the original investigation (source cell is 811).
    if "C" in properties:
        properties["C"]["atomic_mass"] = 12.011

    labelled = data.dropna(subset=["Tg"])
    measurements = {}
    for _, row in labelled.iterrows():
        key = canonical_composition(str(row["Composition"]).split(),
                                    str(row["Fraction"]).split())
        measurements.setdefault(key, []).append(float(row["Tg"]))
    X = np.array([descriptors(key, properties) for key in measurements])
    y = np.array([np.mean(values) for values in measurements.values()])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite descriptors or targets in prepared data")
    metadata = {
        "raw_rows": len(data), "labelled_rows": len(labelled),
        "unique_compositions": len(y), "merged_rows": len(labelled) - len(y),
        "composition_groups_with_different_targets": sum(
            len(set(v)) > 1 for v in measurements.values()),
        "feature_count": X.shape[1],
    }
    return X, y, metadata


def checked_bytes(path, expected):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}; remove the file and download again")
    return content


def prepare(data_dir=DEFAULT_DATA_DIR, archive=None):
    try:
        import pandas as pd
        import openpyxl
    except ImportError as exc:
        raise SystemExit("Install preparation dependencies: pip install -r "
                         "examples/metallic_glass/requirements.txt") from exc
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = data_dir / "descriptors.zip"
        if not archive.exists():
            print(f"Downloading 2.8 MB from {SOURCE}")
            with urllib.request.urlopen(URL, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
                raise ValueError("Downloaded archive failed SHA-256 verification")
            archive.write_bytes(content)
    content = checked_bytes(archive, ARCHIVE_SHA256)
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        # Read only the two named members; never extract archive paths.
        for member, expected in WORKBOOKS.items():
            workbook = z.read(member)
            if hashlib.sha256(workbook).hexdigest() != expected:
                raise ValueError(f"Workbook checksum mismatch: {member}")
            frames.append(pd.read_excel(io.BytesIO(workbook), engine="openpyxl"))
    X, y, metadata = prepare_frames(*frames)
    output = data_dir / "features.npz"
    np.savez_compressed(output, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    metadata.update({
        "source": SOURCE, "revision": REVISION, "url": URL,
        "archive_sha256": ARCHIVE_SHA256, "workbook_sha256": WORKBOOKS,
        "features_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "preparation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "openpyxl": openpyxl.__version__},
    })
    (data_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(y)} compositions × {X.shape[1]} features at {output}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--archive", type=Path, help="Use a local, checksum-verified archive")
    args = parser.parse_args()
    prepare(args.data_dir, args.archive)


if __name__ == "__main__":
    main()
