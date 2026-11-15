#!/usr/bin/env python3
"""Stage released metadata into the dataset directory used by the original experiment."""
import argparse
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    "manifests/provenance/targets.csv": "field_eval/cohorts/targets.csv",
    "manifests/provenance/views_manifest.csv": "field_eval/views/views_manifest.csv",
    "manifests/provenance/plantdoc_eligible_parents_DRAFT.csv": "code/field_eval/inputs/plantdoc_eligible_parents_DRAFT.csv",
    "manifests/provenance/fieldplant_eligible_parents_DRAFT.csv": "code/field_eval/inputs/fieldplant_eligible_parents_DRAFT.csv",
    "manifests/mapping_decisions.json": "code/field_eval/inputs/mapping_decisions.json",
    "manifests/plantdoc_to_plantvillage_crosswalk.csv": "code/field_eval/inputs/plantdoc_to_plantvillage_crosswalk_DRAFT.csv",
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=Path("./data"))
    a=p.parse_args()
    for src, dst in SOURCES.items():
        source=ROOT/src; target=a.data_root/dst
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        print(f"{src} -> {target}")
    print("Inputs staged. External images, weights and embedding cache are not bundled.")

if __name__ == "__main__": main()
