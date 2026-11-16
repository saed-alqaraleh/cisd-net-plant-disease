# Reproducing CISD-Net numerical results and regeneration

**Current source version: v1.0.1.** This release adds numerical table and figure
regeneration, comparison against frozen results, and expanded implementation
documentation. The version-specific Zenodo DOI is pending; the
previous v1.0.0 archive remains immutable.

**Citable frozen release:** [CISD-Net v1.0.0](https://github.com/saed-alqaraleh/cisd-net-plant-disease/releases/tag/v1.0.0)
(software DOI: [10.5281/zenodo.24263471](https://doi.org/10.5281/zenodo.24263471)).
For a precise replication of the archived software, check out the `v1.0.0`
Git tag before executing the Tier 1 verification commands. Documentation
updates made on `main` after publication do not change the tagged archive.

## Regenerating research tables and data figures

From the repository root, using Python 3.11:

```bash
python -m pip install -r requirements-analysis.txt
python analysis/recompute_v3.py
python analysis/verify_reference.py
```

The original `analysis/recompute_v3.py` statistical computations are unchanged;
only input/output paths have been adapted to the public repository. It reads the
99 archived prediction NPZ files, the frozen run ledger under `results/`, and
the two final split manifests under `manifests/`. No training or GPU is needed.
It creates these output files under the git-ignored `outputs/` directory:

- `outputs/manuscript/tables/`: `main_results.tex`, `geometry.tex`,
  `paired_overall.tex`, `paired_quartile.tex`, `gates.tex`, `uncertainty.tex`.
- `outputs/manuscript/figures/`: `fig2_accuracy_by_clutter.pdf/.png` and
  `fig3_paired_differences.pdf/.png`.
- `outputs/analysis/numbers_v3.json`: regenerated per-run summaries,
  three-seed sample SDs, paired bootstrap intervals and p-values.

`analysis/numbers_v3.json` is the original supplied numerical reference,
not an input to the recomputation. The comparison checks all common numeric
entries. With repository files alone, all 33 run summaries, 165 paired
contrasts and 99 conformal-set replays match. The original reference also
contains 11 **FPclean** paired contrasts and a FieldPlant perceptual-hash
sensitivity analysis. To reproduce those additional findings, pass the earlier
FieldPlant manifest containing `sha256` and `phash` as the optional argument:

```bash
python analysis/recompute_v3.py /path/to/earlier_fieldplant_manifest_with_phash.csv
python analysis/verify_reference.py
```

That earlier pHash input is **not included** in the current repository.
The numerical data underlying Table 1 (protocol counts) appear in the
regenerated JSON; the supplied script generates LaTeX bodies for Tables 2–7
and the data figures (Figures 2–3), rather than retypesetting Table 1.

## Tier 1 — CPU evidence replay (no images, GPU or network required)

Python 3.11 is recommended.

```bash
python -m pip install -r requirements-verify.txt
python scripts/verify_reproduction.py
```

This verifies the checksum list, archived per-run NPZ SHA-256s and 99-run
ledger, recomputes Top-1, Macro-F1, clutter Q1–Q4, AURCC, nominal-0.05
split-conformal coverage and mean set sizes, replay thresholds, selected
gamma and the exact LAC/modulated set matrices. All 11 × 3 × 3 runs are
included, and the full 33-row aggregate table is printed. FieldPlant
partitions are disjoint by export file stem (`stem_group`);
this does not establish perceptual-duplicate separation.

**Scope distinction:** HSIC, Stiefel overlap and learned view gates are
stored as scalars in the SHA-authenticated ledger, but the small prediction
archives do not contain the learned subspace matrices or per-example
projected embeddings. Those scalars cannot be independently recomputed from
this Tier 1 bundle. Likewise, original checkpoint files are not included.
Do not describe Tier 1 as verifying those quantities from first principles.

The checksum manifest contains all files except itself, since a file cannot
meaningfully contain its own SHA-256 digest. Excluded local directories are
listed in `.gitignore`.

### Paired exploratory confidence intervals

```bash
python scripts/compute_bootstrap_ci.py --protocol P1 --baseline gem_linear_baseline --method cisd_net_single --seed 42
```

This optional 10,000-resample percentile bootstrap is an exploratory
alternative. For the original reported confidence intervals and permutation
tests, run `python analysis/recompute_v3.py`, which reproduces the paper's
specified 20,000-resample paired procedure from the prediction archives.

The perceptual-hash audit identified 35 near-duplicate pairs, including
19 cross-partition pairs affecting 10 FieldPlant test images. The
reported conclusions persisted after excluding those images, as recorded in the
supplied `analysis/numbers_v3.json`; reproducing the sensitivity analysis
requires the optional earlier pHash manifest (see above).

## Mathematical attribution and implementation details

The joint frame uses the **Stiefel Cayley transform of Jauch, Hoff and
Dunson (2020, *Bernoulli* 26(2):1560–1586,
https://doi.org/10.3150/19-BEJ1176)**, rather than a newly invented
Cayley transform. The largest measured cross-block overlap over all 36
joint-frame runs is `2.0220069661553594e-15` (below `2.1e-15`).

The classifier head is trained with **Adam (coupled L2 on W only)**;
its `W` penalty is added to the loss and its gradient, not applied as
separate decoupled decay. The background swap updates
`h_swap = h + 0.5 * (delta_z_B @ U_B.T)`; background coordinates are moved
toward a paired sample, not replaced. Each clutter probe minimises its own
squared error plus a `1e-3` L2 term, whereas the frame receives the
adversarial `L_B - L_P` gradient. The four-view GeM baseline uses exponent
`2.25` after shifting each image's features by its minimum minus `1e-3`
and adds the shift back afterward.

## Tier 2A — Regenerate experiment heads using verified frozen embeddings

Install the pinned CPU/GPU research dependencies (PyTorch CUDA build selected
for the user's GPU); the frozen head-learning implementation is NumPy-based.

```bash
python -m pip install -r requirements.txt
python scripts/stage_local_inputs.py --data-root ./data
```

Acquire the original `real_4view_backbone_embeddings_v3.npz` from the
study's shared frozen-data distribution (75,604,736 bytes). The original
SHA-256 is:

```
736424459e298317ffb1f94838d45e7daad0ecb71f99d6f44ca54ad96c036205
```

Stage it at `data/real_4view_backbone_embeddings_v3.npz` and run:

```bash
python src/train_and_evaluate.py --data-root ./data \
  --embeddings-npz ./data/real_4view_backbone_embeddings_v3.npz \
  --out-dir ./outputs
```

The full 99-run program writes new checkpoints, prediction NPZ files,
manifests and ledgers under `outputs/`. These can be compared with the
frozen evidence. Results can vary if dependencies, BLAS threading, or
hardware differ; numerical agreement of reconstructed results should be
checked rather than assuming bitwise identical reruns.

## Tier 2B — Complete raw-image regeneration

Source data are not redistributed. Place the upstream FieldPlant images in
an extracted directory and follow the pinned PlantDoc acquisition path. To
re-render all source PNG variants, run:

```bash
python scripts/stage_local_inputs.py --data-root ./data
python src/prepare_views.py --data-root ./data/raw \
  --fieldplant-root ./external/fieldplant --inputs ./data/code/field_eval/inputs \
  --out ./data/field_eval --stages fetch,cohorts,views
```

The original source script assumes external data inventories and precise
original filename encodings. Check all source-file hashes before attempting
feature extraction. To force new frozen features, remove the cached NPZ
and call `src/train_and_evaluate.py` with `--embeddings-npz` pointing to
a new cache path; a compatible CUDA GPU and network access to the official
pretrained weights are required. The program then extracts features from
the four canonical 224×224 view PNGs and executes the same head-training
logic.

The released `views_manifest.csv` is authoritative metadata for checking
regenerated PNGs. A complete raw-image rerun must be independently validated
against the original source data, code and recorded hashes.

## Why the 75.6 MB embedding NPZ is separate

GitHub warns when tracked files exceed 50 MB; this research cache is kept
outside the Git tree and is **not required for the CPU evidence replay**.
The cache is not included in the initial source release. Its redistribution
is subject to clearance of the applicable third-party image/model terms.
If an independently authorized copy is distributed later, verify its exact
SHA-256 (above) before use. Without the cache, regenerate features using
the upstream datasets and model distributions as explained in Tier 2B.
