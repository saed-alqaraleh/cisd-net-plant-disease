# Changelog

## v1.0.1 — Manuscript v6 alignment

- Added original table/figure regeneration script and frozen statistical reference JSON.
- Introduced GitHub Actions checks for numerical consistency against the archived reference.
- Corrected optimizer naming to Adam with coupled L2 on classifier W only (training unchanged).
- Clarified FieldPlant export-file-stem disjointness and disclosed perceptual near-duplicates.
- Replaced population with sample SD in the public results table.
- Documented Cayley transform attribution and exact swap, probe, and GeM semantics.
- Updated manuscript title and sole-author citation metadata.

## v1.0.0 — Initial research software release

- CISD-Net model and training code for frozen DINOv2 and ConvNeXt-Tiny features.
- Reproduction manifests for PlantDoc P1/P2 and FieldPlant FP (partitions disjoint by export file stem).
- Frozen full-precision predictions for 99 evaluations across three seeds.
- CPU numerical and SHA-256 verification, including exact conformal-set replay.
- Explicit distinction between numerical evidence replay and end-to-end image/model regeneration.
- Original source code under MIT; eligible author-generated research records under CC BY 4.0.

Raw source images, pretrained model weights, trained checkpoints and the
optional embedding cache are not included in the source distribution.
