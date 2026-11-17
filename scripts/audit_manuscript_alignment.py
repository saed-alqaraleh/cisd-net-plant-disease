#!/usr/bin/env python3
"""Verify manuscript-facing metadata, cohort counts and frozen key statistics.

Additive source-only check. Model/training/evaluation behavior is unchanged;
full numerical regeneration is performed by analysis/recompute_v3.py.
"""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "CISD-Net: Closed-Form Orthogonal Decomposition Eliminates Subspace Leakage in Field Plant Disease Recognition"

def read_csv(filename):
    with (ROOT / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

plantdoc = read_csv("manifests/plantdoc_P1_P2_manifest.csv")
fieldplant = read_csv("manifests/fieldplant_FP_grouped_manifest.csv")
assert len(plantdoc) == 2185
assert len(fieldplant) == 1273
assert len({r["label_19"] for r in plantdoc}) == 24
assert len({r["label_6"] for r in fieldplant}) == 5

expected = {
    "P1": {"train": 1492, "val": 245, "cal": 245, "test": 203},
    "P2": {"train": 1528, "val": 219, "cal": 219, "test": 219},
    "FP": {"train": 761, "val": 129, "cal": 126, "test": 257},
}
for label, rows, column in [
    ("P1", plantdoc, "split_P1_official"),
    ("P2", plantdoc, "split_P2_stratified"),
    ("FP", fieldplant, "split_FP_grouped"),
]:
    actual = dict(Counter(r[column] for r in rows))
    assert actual == expected[label], (label, actual)

groups = {}
for row in fieldplant:
    g, split = row["stem_group"], row["split_FP_grouped"]
    if g in groups:
        assert groups[g] == split, "FieldPlant export stem spans partitions"
    groups[g] = split

ledger = json.loads((ROOT / "results/v3_run_ledger_and_replay_audit.json").read_text())
original = json.loads((ROOT / "analysis/numbers_v3.json").read_text())
assert len(ledger["per_run_records"]) == 99
assert len(original["R"]) == 33
cfgs = {"cisd_net_single", "cisd_net_dualstream",
        "ablation_stiefel_no_cpi", "ablation_stiefel_no_adv"}
cayley = [r for r in ledger["per_run_records"] if r["config_name"] in cfgs]
assert len(cayley) == 36
maximum = max(r["ortho_cross_up_ub_fro"] for r in cayley)
assert abs(maximum - 2.0220069661553594e-15) < 1e-27
assert maximum < 2.1e-15

r = original["R"]["P1|cisd_net_single"]
for key, expected_value in {
    "top1": 0.7520525451559935,
    "top1_sd": 0.0028440899960080054,
    "f1": 0.7419193615619456,
    "f1_sd": 0.0022138545160892468,
}.items():
    assert abs(r[key] - expected_value) < 1e-12, key
paired = original["PT"]
for name, (delta, lo, hi) in {
    "P1|cisd_net_single|ablation_unconstrained|all":
        (4.269293924466338, 1.1494252873563224, 7.389162561576355),
    "P1|cisd_net_single|ablation_unconstrained|Q4":
        (7.189542483660132, 2.6143790849673207, 12.41830065359477),
}.items():
    obj = paired[name]
    for key, value in zip(("d", "lo", "hi"), (delta, lo, hi)):
        assert abs(obj[key] - value) < 1e-10, (name, key)

readme = (ROOT / "README.md").read_text(encoding="utf-8")
cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
assert readme.splitlines()[0] == "# " + TITLE
assert 'title: "' + TITLE + '"' in cff
assert zenodo["title"] == TITLE
assert 'family-names: "Alqaraleh"' in cff
assert len(zenodo["creators"]) == 1
assert zenodo["creators"][0]["name"] == "Alqaraleh, Saed"
assert 'version: "1.0.1"' in cff
assert "75.21 ± 0.28" in readme
assert "Figures 3–4" in readme

print("PASS: exact title, sole author, splits, label counts, 99 runs,")
print("      36 Cayley-frame measurements and headline comparisons agree.")
print("NOTE: FieldPlant pHash sensitivity requires an additional manifest.")
