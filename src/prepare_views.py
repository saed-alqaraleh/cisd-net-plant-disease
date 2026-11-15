#!/usr/bin/env python3
"""Deterministically prepare PlantDoc and FieldPlant photo cohorts and spatial views.

Stages: fetch (pin upstream PlantDoc); cohorts (verify source hashes and
selected mappings); views (six generated policies, of which four are consumed
by CISD-Net). Raw datasets are not bundled in this repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PD_REPO = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset"
PD_COMMIT = "4730a233a555b30ee98e0879c63ad25d82407455"
IMG = 224
MARGIN = 0.10
MEAN_RGB = (124, 116, 104)          # ImageNet mean x 255, rounded
VIEWS = ("F", "C", "M")
POLICIES = ("resize", "letterbox")
MIN_PER_CLASS = 30


# ------------------------------------------------------------------ helpers
def log(stage, msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{stage}] {msg}", flush=True)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(p: Path, obj):
    t = p.with_suffix(p.suffix + ".tmp"); t.write_text(json.dumps(obj, indent=1, default=str)); os.replace(t, p)


def atomic_csv(p: Path, rows):
    t = p.with_suffix(p.suffix + ".tmp")
    with open(t, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(t, p)


def done(d: Path) -> bool:
    return (d / "_STAGE_SUCCESS.json").exists()


def mark(d: Path, info: dict):
    atomic_json(d / "_STAGE_SUCCESS.json", {**info, "finished_utc": datetime.now(timezone.utc).isoformat()})


# ------------------------------------------------------------------ fetch
def stage_fetch(data_root: Path) -> Path:
    repo = data_root / "PlantDoc-Object-Detection-Dataset"
    if (repo / ".git").exists() and subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == PD_COMMIT:
        log("fetch", f"PlantDoc already at {PD_COMMIT[:8]}"); return repo
    data_root.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", PD_REPO, str(repo)], check=True)
    subprocess.run(["git", "checkout", "-q", PD_COMMIT], cwd=repo, check=True)
    log("fetch", f"PlantDoc at {PD_COMMIT[:8]}")
    return repo


# ------------------------------------------------------------------ cohorts
def load_decisions(p: Path, allow_pending: bool = False) -> dict:
    d = json.loads(p.read_text())
    pend = [k for k, v in d["approximate_mappings"].items() if v not in (True, False)]
    if pend and not allow_pending:
        raise SystemExit(f"approximate mapping decisions pending for: {pend}. Fill mapping_decisions.json (true/false).")
    return d


def stage_cohorts(pd_repo: Path, fp_root: Path | None, inputs: Path, out: Path, decisions: dict) -> Path:
    sdir = out / "cohorts"; sdir.mkdir(parents=True, exist_ok=True)
    tpath = sdir / "targets.csv"
    if done(sdir) and tpath.exists():
        log("cohorts", f"already complete: {tpath}"); return tpath
    appr = {k for k, v in decisions["approximate_mappings"].items() if v is True}
    rows = []
    # PlantDoc
    cw = {r["plantdoc_label"]: r for r in csv.DictReader(open(inputs / "plantdoc_to_plantvillage_crosswalk_DRAFT.csv"))}
    sparse = set(decisions["plantdoc_sparse_excluded"])
    approx_pd = set(decisions["plantdoc_approximate_labels"])
    for r in csv.DictReader(open(inputs / "plantdoc_eligible_parents_DRAFT.csv")):
        lab = r["labels"]
        if lab not in cw or lab in sparse:
            continue
        status = "approximate" if lab in approx_pd else "exact"
        rows.append({"dataset": "plantdoc", "relpath": f"{r['split'].upper()}/{r['file']}", "sha256": r["sha256"],
                     "label": lab, "pv_class": cw[lab]["plantvillage_class"], "mapping": status,
                     "width": int(r["w"]), "height": int(r["h"]),
                     "x1": float(r["box_x1"]), "y1": float(r["box_y1"]), "x2": float(r["box_x2"]), "y2": float(r["box_y2"]),
                     "box_frac": float(r["box_frac"]),
                     "primary": int(status == "exact"),
                     "sensitivity": int(status == "exact" or lab in appr)})
    # FieldPlant
    if fp_root is not None:
        fmap = decisions["fieldplant_map"]
        fp = [r for r in csv.DictReader(open(inputs / "fieldplant_eligible_parents_DRAFT.csv")) if r["label"] in fmap]
        ok_lab = {l for l, m in fmap.items() if m["mapping"] == "exact" or l in appr}
        counts = {}
        for r in fp:
            if r["label"] in ok_lab:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
        rep = {l for l, n in counts.items() if n >= MIN_PER_CLASS}
        exact_rep = {l for l in rep if fmap[l]["mapping"] == "exact"}
        for r in fp:
            lab = r["label"]
            rows.append({"dataset": "fieldplant", "relpath": r["file_name"], "sha256": r["sha256"], "label": lab,
                         "pv_class": fmap[lab]["pv_class"], "mapping": fmap[lab]["mapping"],
                         "width": int(r["width"]), "height": int(r["height"]),
                         "x1": float(r["box_x1"]), "y1": float(r["box_y1"]), "x2": float(r["box_x2"]), "y2": float(r["box_y2"]),
                         "box_frac": float(r["box_frac"]),
                         "primary": int(lab in rep), "sensitivity": int(lab in exact_rep)})
    rows = [r for r in rows if r["primary"] or r["sensitivity"]]   # unused rows (rejected mappings) are not evaluated
    rows.sort(key=lambda r: (r["dataset"], r["relpath"]))
    for i, r in enumerate(rows):
        r["target_id"] = f"{'PD' if r['dataset']=='plantdoc' else 'FP'}{i:05d}"
    # verify the image files are exactly the inventoried bytes
    bad = []
    for i, r in enumerate(rows, 1):
        root = pd_repo if r["dataset"] == "plantdoc" else fp_root
        p = root / r["relpath"]
        if not p.exists() or sha_file(p) != r["sha256"]:
            bad.append(r["relpath"])
        if i % 500 == 0:
            log("cohorts", f"hash-checked {i}/{len(rows)}")
    if bad:
        raise RuntimeError(f"{len(bad)} target files missing or changed, e.g. {bad[:3]}")
    rows = [{"target_id": r.pop("target_id"), **r} for r in rows]
    atomic_csv(tpath, rows)
    summ = {}
    for ds in ("plantdoc", "fieldplant"):
        for flag in ("primary", "sensitivity"):
            sel = [r for r in rows if r["dataset"] == ds and r[flag] == 1]
            if sel:
                per = {}
                for r in sel:
                    per[r["label"]] = per.get(r["label"], 0) + 1
                summ[f"{ds}_{flag}"] = {"n": len(sel), "n_labels": len(per), "per_label": dict(sorted(per.items()))}
    summ["approved_approximations"] = sorted(appr)
    summ["targets_sha256"] = sha_file(tpath)
    atomic_json(sdir / "cohort_summary.json", summ)
    log("cohorts", json.dumps({k: (v["n"] if isinstance(v, dict) and "n" in v else v) for k, v in summ.items()}))
    mark(sdir, {"targets_sha256": summ["targets_sha256"], "n_targets": len(rows)})
    return tpath


# ------------------------------------------------------------------ views
def make_views(im, box):
    """im: PIL RGB (raw orientation). box: (x1,y1,x2,y2) in pixels. Returns dict (view, policy) -> 224x224 PIL."""
    from PIL import Image, ImageDraw
    W, H = im.size
    x1, y1, x2, y2 = box
    mx, my = MARGIN * (x2 - x1), MARGIN * (y2 - y1)
    ex = (max(0, int(round(x1 - mx))), max(0, int(round(y1 - my))),
          min(W, int(round(x2 + mx))), min(H, int(round(y2 + my))))
    if ex[2] - ex[0] < 2 or ex[3] - ex[1] < 2:
        raise ValueError(f"degenerate enlarged box {ex}")
    full = im
    crop = im.crop(ex)
    masked = Image.new("RGB", (W, H), MEAN_RGB); masked.paste(im.crop(ex), (ex[0], ex[1]))
    out = {}
    for v, x in (("F", full), ("C", crop), ("M", masked)):
        out[(v, "resize")] = x.resize((IMG, IMG), Image.Resampling.BILINEAR)
        s = max(x.size)
        sq = Image.new("RGB", (s, s), MEAN_RGB); sq.paste(x, ((s - x.size[0]) // 2, (s - x.size[1]) // 2))
        out[(v, "letterbox")] = sq.resize((IMG, IMG), Image.Resampling.BILINEAR)
    return out, ex


def png_bytes(im) -> bytes:
    b = io.BytesIO(); im.save(b, format="PNG", optimize=False, compress_level=6); return b.getvalue()


def stage_views(pd_repo: Path, fp_root: Path | None, tpath: Path, out: Path) -> Path:
    import numpy as np
    from PIL import Image
    sdir = out / "views"; sdir.mkdir(parents=True, exist_ok=True)
    vpath = sdir / "views_manifest.csv"
    if done(sdir) and vpath.exists():
        log("views", f"already complete: {vpath}"); return vpath
    targets = list(csv.DictReader(open(tpath)))
    part = sdir / "views_manifest.partial.csv"
    have = {}
    if part.exists():
        for r in csv.DictReader(open(part)):
            have[(r["target_id"], r["view"], r["policy"])] = r
    fields = ["target_id", "view", "policy", "png", "png_sha256", "pixels_sha256", "ex_x1", "ex_y1", "ex_x2", "ex_y2", "src_w", "src_h"]
    new_file = not part.exists()
    f = open(part, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader()
    t0 = time.time()
    for i, t in enumerate(targets, 1):
        if all((t["target_id"], v, p) in have for v in VIEWS for p in POLICIES):
            continue
        if t["dataset"] == "fieldplant" and fp_root is None:
            raise SystemExit("FieldPlant targets present but --fieldplant-root not given")
        root = pd_repo if t["dataset"] == "plantdoc" else fp_root
        with Image.open(root / t["relpath"]) as im0:
            im = im0.convert("RGB")          # raw orientation; EXIF intentionally not applied
        if im.size != (int(t["width"]), int(t["height"])):
            raise RuntimeError(f"{t['relpath']}: size {im.size} != manifest {(t['width'], t['height'])}")
        views, ex = make_views(im, tuple(float(t[k]) for k in ("x1", "y1", "x2", "y2")))
        for (v, p), vim in views.items():
            rel = Path(t["dataset"]) / p / v / f"{t['target_id']}.png"
            dst = sdir / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            b = png_bytes(vim)
            tmp = dst.with_suffix(".tmp"); tmp.write_bytes(b); os.replace(tmp, dst)
            w.writerow({"target_id": t["target_id"], "view": v, "policy": p, "png": str(rel),
                        "png_sha256": hashlib.sha256(b).hexdigest(),
                        "pixels_sha256": hashlib.sha256(np.asarray(vim, dtype=np.uint8).tobytes()).hexdigest(),
                        "ex_x1": ex[0], "ex_y1": ex[1], "ex_x2": ex[2], "ex_y2": ex[3], "src_w": im.size[0], "src_h": im.size[1]})
        f.flush()
        if i % 100 == 0 or i == len(targets):
            log("views", f"{i}/{len(targets)} targets ({time.time()-t0:.0f}s)")
    f.close()
    rows = list(csv.DictReader(open(part)))
    keyset = {(r["target_id"], r["view"], r["policy"]) for r in rows}
    need = {(t["target_id"], v, p) for t in targets for v in VIEWS for p in POLICIES}
    if need - keyset:
        raise RuntimeError(f"{len(need - keyset)} views missing")
    # keep the last row per key (a crash can leave a duplicate); order deterministically
    last = {}
    for r in rows:
        last[(r["target_id"], r["view"], r["policy"])] = r
    rows = [last[k] for k in sorted(need)]
    for r in rows:   # verify files on disk
        if sha_file(sdir / r["png"]) != r["png_sha256"]:
            raise RuntimeError(f"hash mismatch {r['png']}")
    atomic_csv(vpath, rows)
    mark(sdir, {"views_manifest_sha256": sha_file(vpath), "n_views": len(rows)})
    log("views", f"{len(rows)} views written and verified")
    return vpath



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("./data/raw"))
    ap.add_argument("--fieldplant-root", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=Path("./data/code/field_eval/inputs"))
    ap.add_argument("--out", type=Path, default=Path("./data/field_eval"))
    ap.add_argument("--stages", type=str, default="fetch,cohorts,views")
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if set(stages) - {"fetch", "cohorts", "views"}:
        ap.error("Choose fetch,cohorts,views")
    pd_repo = args.data_root / "PlantDoc-Object-Detection-Dataset"
    if "fetch" in stages:
        pd_repo = stage_fetch(args.data_root)
    if "cohorts" in stages:
        decisions = load_decisions(args.inputs / "mapping_decisions.json")
        tpath = stage_cohorts(pd_repo, args.fieldplant_root, args.inputs, args.out, decisions)
    else:
        tpath = args.out / "cohorts" / "targets.csv"
    if "views" in stages:
        if not tpath.exists():
            raise SystemExit(f"Missing frozen cohort CSV: {tpath}")
        stage_views(pd_repo, args.fieldplant_root, tpath, args.out)


if __name__ == "__main__":
    main()
