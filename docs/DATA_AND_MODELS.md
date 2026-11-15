# Data and model provenance

The study extracts features from **PlantDoc** and **FieldPlant v11**, using the
upstream PlantDoc object-detection repository pinned to Git commit
`4730a233a555b30ee98e0879c63ad25d82407455`:
https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset

Frozen backbone model names (timm):
- `vit_base_patch14_dinov2.lvd142m` — DINOv2-B/14, 768 dimensions
- `convnext_tiny.fb_in1k` — ConvNeXt-Tiny, 768 dimensions

Neither raw photos nor pretrained weights are in this repository. Users must
obtain them from their official distributors and obey their licenses.

## View and cohort derivation

`src/prepare_views.py` follows the original fixed view code: a 10%-enlarged
bbox crop, a full image with outside-box pixels set to mean RGB (124,116,104),
full-image resize, and full-image letterbox. The original pipeline renders
six (3 views × 2 policies) PNG variants for each of 3,721 targets; **four**
canonical variants are used for CISD-Net feature extraction. The 22,326-row
view metadata and pixel/file hashes are in
`manifests/provenance/views_manifest.csv`.

The source `manifests/mapping_decisions.json` explicitly records that
approximate mappings were adjudicated by the author using published
precedent and **were not independently reviewed by a pathologist**.
Descriptions elsewhere must not substitute a claim of expert sign-off.
Source inventory filenames ending in `_DRAFT` are frozen upstream filenames;
they are retained as provenance and do not change the final benchmark splits.

Main evaluation cohorts: 2,185 PlantDoc targets (19 in-domain classes), and
1,273 group-audited FieldPlant targets (6 primary classes). `targets.csv`
contains all 3,721 target rows before the PlantDoc in-domain restriction.
