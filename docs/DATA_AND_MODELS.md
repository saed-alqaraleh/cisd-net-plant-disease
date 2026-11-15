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

## Upstream licensing and distribution scope

- PlantDoc object-detection dataset: source and attribution information are
  available at https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.
  The dataset repository identifies a Creative Commons Attribution 4.0
  license; however, the creators report that some original images were
  collected from the web, so upstream image rights require independent care.
- FieldPlant v11: upstream dataset and its stated CC BY 4.0 license are at
  https://universe.roboflow.com/plant-disease-detection/fieldplant/dataset/11.
- Frozen image encoders: see the model distributors at
  https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m and
  https://huggingface.co/timm/convnext_tiny.fb_in1k.
  Verify the terms applicable to the exact model weights before redistribution.

This repository releases original scientific software and research evaluation
records; it does **not** redistribute third-party raw images or pretrained
weights. The optional 75.6 MB frozen embedding cache is not included in the
initial source release pending redistribution review. The published CPU
verification pathway is fully usable without this cache.
