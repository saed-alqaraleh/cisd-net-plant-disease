# Pre-publication release checklist

The repository should remain **private** until scientific, licensing and
publication checks are complete.

- [x] Preserve the frozen 99 prediction NPZ files byte-for-byte
- [x] Preserve both benchmark split manifests and the archived 99-run ledger
- [x] Include original view-creation and training math and clean local CLI
- [x] Verify evidence SHA-256s, recorded metrics, aggregates and 99 conformal replays
- [x] Confirm FP split has no repeated stem group across splits
- [x] Check for hardcoded personal Drive paths and excluded experimental outputs
- [x] Create the empty GitHub repository as **private**
- [x] Upload the exact vetted files in one initial commit and rerun checks on a fresh clone
- [ ] Confirm full training rerun on a compatible CPU/GPU installation with external datasets
- [ ] Review data redistribution rights, including metadata and embedding-cache licensing
- [ ] Verify whether missing NVIDIA L4 FP16 profiling log can be released
- [x] Confirm Saed Alqaraleh as the sole author in software citation metadata
- [ ] Add the journal citation and DOI after publication details are finalized
- [ ] Check manuscript/repository table and statistical reproducibility claims line by line
- [ ] Approve public release; create GitHub release tag and link Zenodo archive

**Do not publish or mint a DOI before all release decisions are authorized.**
