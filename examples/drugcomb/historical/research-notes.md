# Drug Combination Synergy — Research Notes

## Dataset

- **Source**: DrugComb via TDC (Therapeutics Data Commons)
- **Size**: 246,776 drug-pair × cell-line triples (after cell line matching)
- **Features**: 567 = 184 drug1 RDKit + 183 drug2 RDKit + 200 genes
- **Target**: Synergy_Bliss (centered ~0, positive = synergistic)
  - Range: [-77, +79], mean=0.13, std=6.23
- **Cell lines**: 49 (matched from 59 NCI-60 lines to GDSC2 expression)
  - 32 exact name matches + 17 manual name corrections
  - 10 NCI-60 lines not in GDSC2 (dropped: CAKI-1, MALME-3M, MDA-MB-435, NCI-H460, NCI/ADR-RES, SF-295, SK-MEL-28, SNB-19, SNB-75, UO-31)
- **Drug pairs**: 5,628 unique pairs from ~129 drugs
- **Gene expression**: top 200 by variance from GDSC2 (17,737 total)
- **Grouping**: cell-line-grouped k-fold CV (no leakage across cell lines)

## Phase 1: Baseline Architectures

### Flat Ridge (R²=0.12, MAE=4.16)

| Metric | Dev (5-fold CV) | Sanctified |
|--------|-----------------|------------|
| MAE    | 4.156 ± 0.511   | 3.694      |
| R²     | 0.119 ± 0.010   | 0.086      |

High fold variance (3.56–4.91) — cell line grouping creates very different
difficulty levels. Sanctified is easier (lower MAE) — the held-out cell lines
happen to have less extreme synergy.

### Cross-Product Ridge (R²=0.15, MAE=4.13) — quick mode

Drug1 × drug2 descriptor cross-products add modest signal (+0.03 R²).

### Hybrid Ridge+NN (R²=0.125, MAE=4.14)

| Metric | Dev (5-fold CV) | Sanctified |
|--------|-----------------|------------|
| MAE    | 4.136 ± 0.501   | 3.686      |
| R²     | 0.125 ± 0.014   | 0.089      |

Gene split: **0 stable / 200 flipping**. Every gene flips sign across drug
pairs (vs 72% on single drugs in GDSC2). Makes sense: with 5,628 pairs instead
of 5 drugs, every gene will be positive for some pairs and negative for others.

The hybrid architecture degenerates: ridge sees only drug descriptors (368 features),
NN sees all 200 genes. Functionally a "drug ridge + gene NN" split rather
than a "stable ridge + flipping NN" split. Marginal gain over flat ridge.

## Phase 2: Soup Feature Engineering

### Critical methodological finding: val scoring must be cell-line-grouped

With only 49 cell lines, random train/val splits leak cell-line identity
into feature selection. Soup with internal kfold (v2) found features that
discriminated cell lines rather than synergy — **hurt R² by -0.01** on
held-out data.

Fix: pass `X_val`/`y_val` with cell-line-grouped split to SoupPipe, disable
internal fold_eval. This forces selection to find features that generalize
**across** cell lines.

### Domain-level soup: mostly noise

Individual drug/gene soup features add nothing in isolation. Makes sense:
synergy is a cross-term phenomenon — no single drug's descriptor space
or gene expression alone encodes it.

### Cross-domain soup: real signal (+0.020 R²)

Full 246k dataset, cell-line-grouped val, held-out test (36k samples, unseen CLs):

| Config | R² | ΔR² | Features |
|--------|-----|------|----------|
| baseline (raw only) | 0.113 | — | 567 |
| + drug1 soup | 0.113 | +0.000 | 582 |
| + drug2 soup | 0.114 | +0.001 | 582 |
| + drug1+drug2 soup | 0.114 | +0.001 | 597 |
| + gene soup | 0.120 | +0.007 | 582 |
| + cross-drug | 0.122 | +0.009 | 597 |
| + drug×gene | 0.124 | +0.011 | 597 |
| + all domain soups | 0.121 | +0.008 | 612 |
| **+ all soups + cross** | **0.133** | **+0.020** | 672 |
| soup only (no raw) | 0.101 | -0.012 | 105 |

### What soup discovered

**Cross-drug features** (survived held-out CL validation):
- `drug2_MinEStateIndex → cos(rank(drug1_MolWt) × drug2_MolWt)` — electrotopological
  state of drug2 modulates a periodic function of the molecular weight product.
  Heavier drug pairs have periodic synergy/antagonism structure.
- `drug1_MinEStateIndex → cos(drug2_MolWt × rank(drug2_MinAbsEStateIndex))` — drug1's
  electronic character interacts with drug2's size×charge axis.
- `min(log|drug1_MolWt|, log|drug2_MolWt|)` — the **smaller** drug's molecular weight
  matters for synergy. Consistent with delivery/uptake bottleneck.
- `rank(drug1_qed) × drug1_MolWt^3` — drug-likeness × mass interaction as a gating
  signal (step function — binary on/off).

**Gene features** (emerged at 150k+ training samples, invisible at 50k):
- LOX→MX1: lysyl oxidase modulates interferon response (MX1) — extracellular matrix
  remodeling affects immune signaling, both relevant to drug combo response.
- SRGN→CCL2: serglycin proteoglycan modulates CCL2 (monocyte chemotactic protein) —
  immune microenvironment fingerprint.
- MLANA→CCL2: melanocyte marker modulating immune chemokine — tissue lineage
  interacts with immune context.
- CYBA (cytochrome b-245 alpha): reactive oxygen species gene. Survived val scoring
  in drug×gene cross-domain pipe. ROS modulates drug oxidative stress differently
  depending on the drug pair.

**Negative results** (didn't survive val):
- MLANA as a standalone gene soup feature — cell-line fingerprint, not synergy predictor.
  Only useful when crossed with drug descriptors.
- Any single-drug soup feature — pharmacokinetic periodicity within one drug's
  space doesn't predict synergy.

### Scale effects

| Subsample | kfold val (no CL grouping) | CL-grouped val |
|-----------|---------------------------|----------------|
| 10k | drug soup +0.004 | — |
| 20k | drug soup -0.008 (overfit) | — |
| 50k | — | cross +0.011, drug×gene +0.012 |
| 246k (full) | — | **all +0.020** |

Gene soup signal only appears at 150k+ samples. Cross-domain features
scale well — +0.020 at full scale vs +0.014 at 50k.

## Key Observations

1. **R²=0.13 with soup is near the ceiling for these features.** Synergy is a
   second-order signal — most of Bliss score variance is noise from dose-response
   fitting, not learnable biology.

2. **100% gene flipping** across drug pairs. The stable/flipping split from
   GDSC2 doesn't apply — all genes are regime-dependent when there are 5,628
   drug pair contexts instead of 5 single drugs.

3. **Cross-domain features are the only useful soup output.** Drug1×drug2 and
   drug×gene interactions contain synergy signal; individual domain features don't.
   This confirms the structural hypothesis: synergy is fundamentally a cross-term.

4. **Val scoring methodology matters more than data size.** Without CL-grouped val,
   soup actively hurts. With it, soup consistently helps. The difference is whether
   selection finds cell-line fingerprints (bad) or drug-pair synergy patterns (good).

5. **High fold variance** (MAE 3.56–4.91) suggests cell-line-specific effects
   dominate. Some cell lines are inherently harder to predict.

## Phase 3: Residual Soup

Fit ridge on raw+soup → soup on augmented-ridge residuals → re-fit.

**Result: no gain.** Every stacked config within noise of direct soup (+0.016 vs +0.017).
Direct soup already found the cross-domain interactions. The residual after
augmented ridge is noise — no second layer of structure to find.

Important fix: must exclude `rank` and `am_ratio` families for residual soup.
Rank transforms destroy residual structure (ordinal position is cell-line-dominated).
Ratio features with rank denominators blow up on held-out cell lines (division by
unexpected near-zero values → R² = -47M).

## Phase 4: NN Experiments

Tested standalone NN, hybrid (ridge + NN on residuals), various architectures.

| Method | R² (50k) | Notes |
|---|---|---|
| Flat ridge | 0.103 | baseline |
| Ridge + direct soup | 0.120 | cross-domain features |
| Ridge + soup + NN hybrid [16,8] | 0.122 | NN adds +0.002 |
| Standalone NN [64,32] on raw | 0.008 | can't generalize across CLs |
| Standalone NN [32,16] on raw+soup | 0.010 | same problem |
| NN on soup only | -0.004 | worse than random |

**Standalone NN is terrible** — R²=0.01 at best. With 49 cell lines, the NN
memorizes training CLs and can't generalize. The hybrid adds +0.002 over
ridge+soup — noise-level.

**Conclusion: the signal is genuinely linear at this feature level.** Ridge
already captures it. The NN brings nothing because:
1. 49 cell lines = 49 unique expression vectors — too few for nonlinear learning
2. Cross-domain interactions are already encoded in soup features
3. The bottleneck is cell line diversity, not model capacity

## Final Results

| Method | R² (50k) | R² (246k) |
|---|---|---|
| Flat ridge | 0.103 | 0.113 |
| Ridge + direct soup | 0.120 | **0.133** |
| + residual soup | 0.120 | — |
| + NN hybrid | 0.122 | — |

Best: **R²=0.133** on Bliss synergy, 246k samples, held-out cell lines.
Competitive with published deep learning baselines (R²=0.12-0.25 on Bliss)
without molecular graphs, learned embeddings, or deep learning.

## Key Findings (cross-project)

1. **Soup val scoring must respect grouping structure.** Without CL-grouped val,
   soup learns entity fingerprints → hurts on held-out data. With it, soup finds
   genuine cross-domain interactions → +0.020 R². Applies to all grouped datasets
   (GDSC2, PRISM, perovskite with composition groups).

2. **Synergy signal lives exclusively in cross-terms.** Drug1-only, drug2-only,
   and gene-only soup features add zero. Only drug1×drug2 and drug×gene
   interactions predict synergy. Fundamental structural property of Bliss scores.

3. **100% gene flipping** across 5,628 drug pairs. The GDSC2 stable/flipping
   split (72% flip) doesn't apply — every gene is regime-dependent when there
   are thousands of drug-pair contexts. The hybrid architecture degenerates.

4. **49 cell lines is the hard bottleneck.** NN can't generalize, gene features
   are cell-line fingerprints in disguise, and val scoring is fragile. More cell
   lines (DepMap has 1,400+) would unlock nonlinear models and richer gene soup.

5. **Loewe vs Bliss**: Loewe R² = 0.55-0.65 in published work, but the null model
   is pharmacologically wrong (treats drugs as dilutions of each other). Loewe
   "synergy" is mostly artifact of the bad null. Bliss R²=0.13 is predicting
   actual pathway crosstalk — harder but more meaningful.

## Next Steps (M5)

- [ ] Full scale (246k) with Research framework + proper 5-fold sanctified
- [ ] More cell lines via DepMap (1,400+ CLs with expression)
- [ ] CSS target (raw combo sensitivity, not synergy residual)
- [ ] Per-MoA specialist cascade for drug pairs
- [ ] Molecular graph features (if RDKit fingerprints become available)
