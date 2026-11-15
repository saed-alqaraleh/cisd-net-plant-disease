# CISD-Net — Counterfactual Invariant Subspace Decomposition Network

**Reproducible frozen-backbone experiments for plant disease recognition under background clutter**

CISD-Net is a research head trained on four 224×224 visual views extracted
from PlantDoc and FieldPlant images. It combines a joint block-Cayley Stiefel
parameterization for orthogonal pathology/background subspaces, a
differentiable counterfactual pairing invariance objective, and
validation-selected clutter-modulated split-conformal prediction sets.

This research software includes 99 frozen evaluations (3 protocols ×
11 configurations × 3 seeds), full-precision prediction records, split
manifests and a self-contained offline verification workflow. The numerical
replay uses only repository files. Raw plant images, pretrained backbone
weights, trained model checkpoints and the optional embedding cache are
not distributed in the source repository.

## Quick start: reproduce the archived numerical evidence

```bash
python -m pip install -r requirements-verify.txt
python scripts/verify_reproduction.py
```

This verifies SHA-256 of the tracked release, all 99 full-precision
prediction records and 99/99 conformal set agreement, reproduces all
computable benchmark metrics and 3-seed aggregates, and checks FieldPlant
split-group separation. Requires no GPU, raw images or external data paths.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full details and limitations.

## Experimental design

| Component | Fixed design |
| --- | --- |
| Frozen backbone 1 | `vit_base_patch14_dinov2.lvd142m` (DINOv2-B/14, 768-d) |
| Frozen backbone 2 | `convnext_tiny.fb_in1k` (ConvNeXt-Tiny, 768-d) |
| Used spatial views | F/resize, C/resize, M/resize, F/letterbox |
| View generation | 224×224, 10% enlarged leaf crop, RGB mask (124,116,104) |
| Protocols | P1 PlantDoc official-test preserved, P2 PlantDoc stratified, FP FieldPlant group-disjoint |
| Group-separation checks | FieldPlant `stem_group` does not span train/val/cal/test |
| Seeds and initialization | 42, 108, 2026; random basis initialization; uniform `[0.25]×4` view gates |
| Orthogonal subspace dimensions | `(dP,dB)=(64,16)` for 768-d; `(128,32)` for 1536-d |
| Training | 40 epochs, AdamW, three LRs (0.0015/0.0040/0.0250), cosine decay (15% floor) |
| Fixed conformal alpha | 0.05; calibration-only threshold, validation-only gamma selection |
| Source view count | 3,721 target images; 22,326 generated view records; 4 used for features |

The score `clutter_raw` is measured from L2-normalized frozen DINOv2
full-resize versus masked-resize embeddings. It is not a manually annotated
background score.

## Primary results

The best overall P1 model by Top-1 and Macro-F1 is **single-stream CISD-Net**,
with 75.21 ± 0.23% Top-1 and 64.05% high-clutter Q4 Top-1. It does **not**
have the best overall Top-1 on P2 (cosine-prototype does) or FieldPlant
(dual-stream concat does). This distinction is preserved in the full table.

The following numbers are mean ± population SD across seeds 42, 108, 2026,
from the frozen experiment ledger. Q1 and Q4 are mean accuracies by recorded
clutter quartiles. HSIC/orthogonality values are archived scalar
measurements (not inferable from the prediction NPZs alone).

| Protocol | Configuration | Top-1 % | Macro-F1 % | Q1 % | Q4 % | LAC size (coverage) | Mod size (coverage) | AURCC % | Cross overlap | Raw HSIC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | `erm_cls` | 73.89 ± 0.00 | 72.75 ± 0.00 | 78.43 | 60.78 | 3.48 (97.5%) | 3.48 (97.5%) | 9.32 | 0 | 0.0000 |
| P1 | `gem_linear_baseline` | 74.38 ± 0.00 | 73.07 ± 0.00 | 80.39 | 56.86 | 2.59 (96.1%) | 2.59 (96.1%) | 8.37 | 0 | 0.0000 |
| P1 | `cosine_proto_baseline` | 72.41 ± 0.00 | 70.08 ± 0.00 | 80.39 | 56.86 | 2.23 (94.6%) | 2.23 (94.6%) | 9.40 | 0 | 0.0000 |
| P1 | `dualstream_concat_ctrl` | 70.94 ± 0.00 | 69.27 ± 0.00 | 70.59 | 56.86 | 3.19 (95.6%) | 3.19 (95.6%) | 10.37 | 0 | 0.0000 |
| P1 | `ablation_unconstrained` | 70.94 ± 0.70 | 69.18 ± 0.47 | 75.82 | 56.86 | 4.07 (95.2%) | 4.07 (95.2%) | 10.90 | 1.41 | 0.0053 |
| P1 | `ablation_soft_ortho` | 74.71 ± 0.23 | 73.42 ± 0.56 | 79.08 | 59.48 | 2.78 (95.9%) | 2.78 (95.9%) | 9.37 | 0.15 | 0.0025 |
| P1 | `ablation_qr_decoupled_v1` | 73.23 ± 0.46 | 71.83 ± 0.63 | 75.16 | 60.78 | 3.88 (96.9%) | 3.88 (96.9%) | 9.89 | 1.58 | 0.0194 |
| P1 | `ablation_stiefel_no_cpi` | 74.06 ± 0.93 | 73.03 ± 1.26 | 78.43 | 62.75 | 3.05 (95.9%) | 3.08 (96.2%) | 9.80 | 1.44e-15 | 0.0064 |
| P1 | `ablation_stiefel_no_adv` | 73.40 ± 1.84 | 71.88 ± 1.81 | 75.82 | 62.75 | 3.44 (96.4%) | 3.44 (96.4%) | 9.09 | 1.2e-15 | 0.0038 |
| P1 | `cisd_net_single` | 75.21 ± 0.23 | 74.19 ± 0.18 | 79.08 | 64.05 | 2.87 (95.9%) | 2.87 (95.9%) | 9.27 | 1.55e-15 | 0.0059 |
| P1 | `cisd_net_dualstream` | 70.77 ± 1.67 | 69.75 ± 1.87 | 71.90 | 56.86 | 5.09 (97.2%) | 5.09 (97.2%) | 12.73 | 1.76e-15 | 0.0035 |
| P2 | `erm_cls` | 74.89 ± 0.00 | 75.31 ± 0.00 | 83.64 | 78.18 | 2.14 (90.9%) | 2.14 (90.9%) | 9.37 | 0 | 0.0000 |
| P2 | `gem_linear_baseline` | 76.71 ± 0.00 | 76.51 ± 0.00 | 81.82 | 80.00 | 2.37 (93.6%) | 2.37 (93.6%) | 9.02 | 0 | 0.0000 |
| P2 | `cosine_proto_baseline` | 77.63 ± 0.00 | 77.17 ± 0.00 | 87.27 | 78.18 | 1.96 (94.1%) | 1.96 (94.1%) | 7.89 | 0 | 0.0000 |
| P2 | `dualstream_concat_ctrl` | 75.34 ± 0.00 | 75.77 ± 0.00 | 81.82 | 78.18 | 2.52 (91.3%) | 2.52 (91.3%) | 10.66 | 0 | 0.0000 |
| P2 | `ablation_unconstrained` | 75.65 ± 0.43 | 74.85 ± 0.90 | 83.64 | 80.61 | 3.55 (95.1%) | 3.55 (95.1%) | 9.41 | 1.87 | 0.0073 |
| P2 | `ablation_soft_ortho` | 76.26 ± 1.12 | 76.00 ± 1.16 | 84.24 | 80.61 | 2.58 (93.3%) | 2.58 (93.3%) | 8.86 | 0.314 | 0.0040 |
| P2 | `ablation_qr_decoupled_v1` | 74.28 ± 2.48 | 73.97 ± 2.65 | 84.85 | 76.36 | 3.43 (94.4%) | 3.43 (94.4%) | 9.41 | 2.04 | 0.0195 |
| P2 | `ablation_stiefel_no_cpi` | 75.34 ± 0.65 | 74.80 ± 0.80 | 84.24 | 80.61 | 2.61 (93.3%) | 2.61 (93.3%) | 9.24 | 1.55e-15 | 0.0047 |
| P2 | `ablation_stiefel_no_adv` | 75.49 ± 0.22 | 75.19 ± 0.07 | 81.21 | 81.21 | 2.75 (93.0%) | 2.75 (93.0%) | 9.32 | 1.2e-15 | 0.0033 |
| P2 | `cisd_net_single` | 75.80 ± 0.65 | 75.25 ± 0.68 | 84.24 | 81.82 | 3.04 (93.9%) | 3.04 (93.9%) | 9.10 | 1.37e-15 | 0.0061 |
| P2 | `cisd_net_dualstream` | 70.32 ± 0.65 | 71.04 ± 0.55 | 78.79 | 72.73 | 2.43 (91.2%) | 2.43 (91.2%) | 12.04 | 1.77e-15 | 0.0027 |
| FP | `erm_cls` | 93.77 ± 0.00 | 82.04 ± 0.00 | 92.19 | 93.85 | 1.00 (93.8%) | 1.00 (93.8%) | 0.88 | 0 | 0.0000 |
| FP | `gem_linear_baseline` | 95.33 ± 0.00 | 85.68 ± 0.00 | 92.19 | 96.92 | 1.01 (95.3%) | 1.01 (95.3%) | 0.60 | 0 | 0.0000 |
| FP | `cosine_proto_baseline` | 85.60 ± 0.00 | 72.50 ± 0.00 | 84.38 | 78.46 | 1.54 (94.6%) | 1.40 (92.6%) | 5.27 | 0 | 0.0000 |
| FP | `dualstream_concat_ctrl` | 95.72 ± 0.00 | 88.17 ± 0.00 | 93.75 | 98.46 | 1.00 (95.7%) | 1.00 (95.7%) | 1.34 | 0 | 0.0000 |
| FP | `ablation_unconstrained` | 93.39 ± 0.32 | 78.66 ± 1.78 | 92.71 | 94.87 | 1.00 (93.4%) | 1.00 (93.4%) | 3.33 | 2.57 | 0.0224 |
| FP | `ablation_soft_ortho` | 92.74 ± 0.37 | 77.67 ± 1.07 | 90.10 | 93.85 | 1.01 (93.3%) | 1.01 (93.3%) | 2.35 | 0.29 | 0.0059 |
| FP | `ablation_qr_decoupled_v1` | 91.83 ± 1.15 | 77.45 ± 4.02 | 90.10 | 93.33 | 1.13 (94.2%) | 1.13 (94.2%) | 3.87 | 1.73 | 0.0248 |
| FP | `ablation_stiefel_no_cpi` | 93.77 ± 0.84 | 84.95 ± 2.61 | 91.15 | 93.85 | 1.04 (94.7%) | 1.04 (94.7%) | 1.45 | 1.4e-15 | 0.0088 |
| FP | `ablation_stiefel_no_adv` | 93.13 ± 0.92 | 83.17 ± 2.22 | 89.58 | 93.33 | 1.03 (93.6%) | 1.03 (93.6%) | 1.60 | 1.3e-15 | 0.0076 |
| FP | `cisd_net_single` | 93.39 ± 0.55 | 82.69 ± 2.13 | 89.58 | 93.33 | 1.02 (93.9%) | 1.02 (93.9%) | 1.49 | 1.4e-15 | 0.0089 |
| FP | `cisd_net_dualstream` | 93.39 ± 0.00 | 82.35 ± 0.92 | 90.10 | 96.92 | 1.03 (94.4%) | 1.03 (94.4%) | 2.41 | 1.92e-15 | 0.0071 |

## Repository contents

| Path | Contents |
| --- | --- |
| `src/cisd_net.py` | Model, Cayley parameterization, CPI objective, conformal computations |
| `src/train_and_evaluate.py` | Original verified v3 run functions with sanitized local CLI |
| `src/prepare_views.py` | Original 10%-crop and mask/letterbox view generation |
| `scripts/verify_reproduction.py` | Integrity + 99-run CPU numerical replay |
| `scripts/compute_bootstrap_ci.py` | Optional explicitly defined paired bootstrap |
| `scripts/stage_local_inputs.py` | Stage frozen source metadata under local `data/` |
| `manifests/` | Two final split manifests, mapping decisions, source traceability records |
| `predictions_full_precision/` | All 99 original full-precision per-run `.npz` archives |
| `results/` | Frozen 99-run ledger and available NVIDIA L4 profiling evidence |
| `docs/` | Data/model provenance and numerical verification scope |

## Evidence boundaries and licensing

This release **does not** redistribute PlantDoc/FieldPlant photos or
pretrained weights, and does not include the 99 trained checkpoint NPZs.
Numerical validation from prediction archives and full model retraining are
distinct reproduction tiers. Original software is under MIT; original
author-generated records are under CC BY 4.0 to the extent they are
licensable, excluding upstream third-party rights. See [LICENSE](LICENSE),
[LICENSE-DATA](LICENSE-DATA) and [docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md).

## Citation

**Sole author:** Saed Alqaraleh (Mutah University, Jordan; [ORCID](https://orcid.org/0000-0002-7146-3905)).
The author is confirmed; the journal venue and article DOI remain pending.
See `CITATION.cff`. The software can be cited as:

```bibtex
@software{cisdnet_2026,
  title        = {CISD-Net: Counterfactual Invariant Subspace Decomposition Network},
  author       = {Alqaraleh, Saed},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/saed-alqaraleh/cisd-net-plant-disease}
}
```

For a persistent software citation, use the Zenodo DOI associated with the
published `v1.0.0` release once it is available. The journal article DOI
will be added separately after publication. Until then, cite the source
repository and version in the BibTeX entry above.
