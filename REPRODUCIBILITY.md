# Reproducing CISD-Net numerical results and regeneration

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
`stem_group` disjointness is also tested.

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

This optional 10,000-resample percentile bootstrap is specified explicitly
in the script. It is **not** claimed as the original manuscript's archived
confidence-interval procedure, because the frozen 99-run ledger does not
contain a matching bootstrap specification or stored CI endpoints.

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
outside the Git tree. Once the private repository is approved for release,
attach it as a GitHub release asset or configure Git LFS if licensing
permits. Do not create a fictitious download URL before the release exists.
