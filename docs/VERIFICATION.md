# Numerical evidence verification and scope

Verification date: 2026-11-15. Source package:
`CISD_Net_v3_Real_Embeddings_Lightweight.zip`, associated frozen cache and
experiment ledger.

## Integrity and reproducibility checks

- All 99 archived prediction NPZ files match their per-run SHA-256 digests
  recorded in `v3_run_ledger_and_replay_audit.json`.
- All 99 LAC and clutter-modulated conformal prediction set matrices exactly
  match replay from stored float32 scores and calibration inputs.
- Recomputed Top-1, Macro-F1, all four clutter-quartile accuracies, AURCC,
  calibrated empirical coverage, mean set size, quantile thresholds, and
  selected gamma match the archived ledger.
- 33 configuration-level three-seed aggregate means and reported SDs match.
- The FieldPlant partitions are disjoint by export file stem (`stem_group`), not
  by perceptual-duplicate identity; the archived
  PlantDoc in-domain split manifest contains 2,185 unique image IDs.
- The frozen embedding cache is 75,604,736 bytes with SHA-256
  `736424459e298317ffb1f94838d45e7daad0ecb71f99d6f44ca54ad96c036205`.
- The full PlantDoc and FieldPlant final split manifests were regenerated
  from the frozen cached embeddings and source input inventories; both are
  byte-identical to the archived manifests.
- A P1 ERM baseline seed-42 rerun reproduced Top-1, Macro-F1, AURCC,
  LAC set size and Q4 exactly.
- A P1 CISD-Net single-stream seed-42 rerun reproduced the same five
  evaluation metrics exactly; Stiefel overlap and HSIC matched up to
  numerical floating-point variation.

- The v6 perceptual-hash audit identified 35 near-duplicate pairs (19 crossing
  partitions, affecting 10 test images). The reported sensitivity after
  removing those images is archived in `analysis/numbers_v3.json` but cannot
  be independently rerun without its optional earlier pHash input.

## Reproduction boundaries

- Checkpoint NPZ files and per-image projected subspace features are not in
  the compact release. Orthogonality, HSIC and gate weights are therefore
  authenticated ledger measurements, not quantities independently recomputed
  by the archival CPU verifier.
- Source photos, pretrained weights and the optional 75.6 MB cache are not
  committed to this Git tree.
- A full 99-run regenerated model-training suite and full raw-image GPU
  extraction are separate workflows, requiring external data and validation.
- The 95% paired bootstrap utility provides a clearly specified exploratory
  procedure, not a claim of replaying unpublished manuscript CIs.

Run `python scripts/verify_reproduction.py` after installation of the CPU
requirements to independently repeat all archive-only checks.
