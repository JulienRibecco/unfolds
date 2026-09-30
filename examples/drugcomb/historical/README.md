# Original research material

These files preserve the investigation behind the [case study](../../../docs/drugcomb-case-study.md):

- [research-notes.md](research-notes.md): the original notes and recorded results.
- [build_features.py.txt](build_features.py.txt): the original data preparation.
- [explore_soup_v3.py.txt](explore_soup_v3.py.txt): the revised, cell-line-grouped
  Soup feature-selection experiment.

Copied from `signal_fault/research/meta-drugs/drug-combination/` on
2026-09-30. The `.txt` suffix marks the scripts as archival source for
inspection. They assume the original workspace layout, internal
`signalfault.classify` modules, and the separate `potage` library. They are
not executable examples supported by this repository's installation.

The original preparation selected genes globally before splitting. The
notes also contain exploratory biological interpretations and results from
different configurations. They are retained as historical records, not
endorsed as validated biological conclusions or a controlled split ablation.

For a runnable, independently packaged experiment using the original data
sources and training-only gene selection, use the [real-data audit](../README.md).
