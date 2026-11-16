#!/usr/bin/env python3
"""Reproduce the frozen CISD-Net study using actual pretrained backbone features.

When a verified embedding cache is present, train/evaluate from the cache.
Without the cache, use the pre-rendered canonical view PNGs and obtain frozen
DINOv2/ConvNeXt embeddings, then train all 99 study configurations.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
try:
    from .cisd_net import (JointStiefelCayleyTransform, compute_unbiased_hsic,
        softmax_rows, compute_macro_f1, compute_aurcc, evaluate_conformal_both_scores,
        build_counterfactual_swap_permutation, CISDNetV3Model)
except ImportError:
    from cisd_net import (JointStiefelCayleyTransform, compute_unbiased_hsic,
        softmax_rows, compute_macro_f1, compute_aurcc, evaluate_conformal_both_scores,
        build_counterfactual_swap_permutation, CISDNetV3Model)

CANONICAL_4_VIEWS = [
    ("F", "resize"),     # View 0: Full image resized to 224x224
    ("C", "resize"),     # View 1: Annotated leaf box cropped (10% margin) & resized to 224x224
    ("M", "resize"),     # View 2: Full image with outside-box masked to ImageNet mean RGB
    ("F", "letterbox"),  # View 3: Full image aspect-ratio-preserving letterbox to 224x224
]

CONFIRMED_HYPERPARAMETERS = {
    "backbone_stream1": "vit_base_patch14_dinov2.lvd142m (768-d, frozen)",
    "backbone_stream2": "convnext_tiny.fb_in1k (768-d, frozen)",
    "initialization": "strictly_neutral_random_seed_only (zero_planted_bases, zero_logit_projection)",
    "optimizer": "Adam (coupled L2 on W only)",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "lr_schedule": "cosine_annealing_with_15pct_floor",
    "epochs": 40,
    "lr_cls": 0.0015,
    "lr_sub": 0.0040,
    "lr_gate": 0.0250,
    "lambda_cpi": 0.35,
    "lambda_adv": 0.20,
    "swap_strength": 0.50,
    "residual_alpha": 0.40,
    "beta_suppress": 0.35,
    "lambda_ortho_soft": 0.15,
    "weight_decay": 0.01,
    "conformal_nominal_alpha": 0.05,
    "view_gate_init": [0.25, 0.25, 0.25, 0.25],
}

CONFIG_SPECS = [
    ("erm_cls",                  "single", "Standard ERM Linear (Full-Image View 0 Only, 768-d)"),
    ("gem_linear_baseline",      "single", "GeM-Pooled Linear Token Baseline (All 4 Views, 768-d)"),
    ("cosine_proto_baseline",    "single", "Cosine-Prototype Token Baseline (All 4 Views, 768-d)"),
    ("dualstream_concat_ctrl",   "dual",   "Dual-Stream Concatenation Linear Control (All 4 Views, 1536-d)"),
    ("ablation_unconstrained",   "single", "Ablation: Unconstrained Subspace Bases (lambda_ortho=0)"),
    ("ablation_soft_ortho",      "single", "Ablation: Soft Frobenius Penalty (lambda_ortho=0.15, Trained)"),
    ("ablation_qr_decoupled_v1", "single", "Ablation: Separately QR-Orthonormalized Subspaces (Separate QR on U_P, U_B)"),
    ("ablation_stiefel_no_cpi",  "single", "Ablation: Joint Stiefel Cayley, No CPI Swap (lambda_cpi=0)"),
    ("ablation_stiefel_no_adv",  "single", "Ablation: Joint Stiefel Cayley + CPI, No Nuisance Adv (lambda_adv=0)"),
    ("cisd_net_single",          "single", "CISD-Net Single-Stream (DINOv2-B/14, 768-d)"),
    ("cisd_net_dualstream",      "dual",   "CISD-Net Dual-Stream (DINOv2-B/14 + ConvNeXt-T, 1536-d)"),
]

SEEDS = [42, 108, 2026]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_or_load_real_4view_embeddings(
    drive_root: Path,
    cache_npz_path: Path,
    batch_size: int = 256,
    num_workers: int = 4,
    ) -> Dict[str, np.ndarray]:
    """
    Extracts real 768-d penultimate embeddings from `vit_base_patch14_dinov2.lvd142m`
    and `convnext_tiny.fb_in1k` across the 4 archived view PNGs in `field_eval/views/`.
    Caches the resulting `(3721, 4, 768)` float32 arrays to `cache_npz_path`.
    """
    if cache_npz_path.exists():
        print(f"[Stage 1] Loading cached real 4-view embeddings from {cache_npz_path} ...")
        expected_sha = "736424459e298317ffb1f94838d45e7daad0ecb71f99d6f44ca54ad96c036205"
        actual_sha = sha256_file(cache_npz_path)
        if actual_sha != expected_sha:
            raise ValueError(f"Cached embedding SHA-256 mismatch: {actual_sha}; expected {expected_sha}")
        # SHA-verified source contains original object-dtype target IDs.
        data = np.load(cache_npz_path, allow_pickle=True)
        return {
            "X_dino": data["X_dino"].astype(np.float64),
            "X_conv": data["X_conv"].astype(np.float64),
            "target_ids": data["target_ids"],
            "clutter_raw": data["clutter_raw"].astype(np.float64),
            "cache_sha256": actual_sha,
        }

    targets_csv = drive_root / "field_eval" / "cohorts" / "targets.csv"
    views_csv = drive_root / "field_eval" / "views" / "views_manifest.csv"
    views_dir = drive_root / "field_eval" / "views"

    import torch
    import timm
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for real backbone extraction. "
            "Use a CUDA-enabled machine and the original pretrained weights."
        )
    dev = torch.device("cuda")

    assert targets_csv.exists(), f"Missing {targets_csv}"
    assert views_csv.exists(), f"Missing {views_csv}"

    targets_df = pd.read_csv(targets_csv)
    views_df = pd.read_csv(views_csv)
    view_lookup: Dict[Tuple[str, str, str], str] = {}
    for row in views_df.itertuples(index=False):
        view_lookup[(str(row.target_id), str(row.view), str(row.policy))] = str(row.png)

    target_ids = targets_df["target_id"].astype(str).values
    N = len(target_ids)
    ordered_png_paths: List[Path] = []
    for tid in target_ids:
        for v_code, pol in CANONICAL_4_VIEWS:
            rel_png = view_lookup[(tid, v_code, pol)]
            ordered_png_paths.append(views_dir / rel_png)

    print(f"[Stage 1] Prefetching {len(ordered_png_paths)} 224x224 view PNGs into RAM using 16 threads ...")
    t_io = time.time()

    def _read_png_bytes(p: Path) -> bytes:
        return p.read_bytes()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        png_bytes_list = list(pool.map(_read_png_bytes, ordered_png_paths))
    print(f"[Stage 1] Loaded {len(png_bytes_list)} view PNGs into RAM in {time.time() - t_io:.1f}s")

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class InMemoryViewDataset(Dataset):
        def __init__(self, raw_bytes: List[bytes]):
            self.raw_bytes = raw_bytes

        def __len__(self) -> int:
            return len(self.raw_bytes)

        def __getitem__(self, idx: int):
            import io
            with Image.open(io.BytesIO(self.raw_bytes[idx])) as im:
                return tf(im.convert("RGB"))

    loader = DataLoader(
        InMemoryViewDataset(png_bytes_list),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    def _run_backbone(model_name: str, img_size: Optional[int] = None) -> np.ndarray:
        t0 = time.time()
        print(f"[Stage 1] Extracting 768-d penultimate embeddings with {model_name} ...")
        kwargs = {"pretrained": True, "num_classes": 0}
        if img_size is not None:
            kwargs["img_size"] = img_size
        model = timm.create_model(model_name, **kwargs).to(dev).eval()
        feats_list = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for batch_x in loader:
                out = model(batch_x.to(dev, non_blocking=True)).float().cpu().numpy()
                feats_list.append(out)
        del model
        torch.cuda.empty_cache()
        feats_all = np.concatenate(feats_list, axis=0)
        assert feats_all.shape == (N * 4, 768), f"Expected {(N*4, 768)}, got {feats_all.shape}"
        print(f"[Stage 1] Finished {model_name} in {time.time() - t0:.1f}s")
        return feats_all.reshape(N, 4, 768).astype(np.float32)

    X_dino = _run_backbone("vit_base_patch14_dinov2.lvd142m", img_size=224)
    X_conv = _run_backbone("convnext_tiny.fb_in1k", img_size=None)

    # Empirical background clutter score directly from real DINOv2 embeddings:
    # L2 distance between View 0 (F/resize: full image with background) and View 2 (M/resize: outside-box masked)
    dino_l2 = X_dino / (np.linalg.norm(X_dino, axis=2, keepdims=True) + 1e-8)
    clutter_raw = np.linalg.norm(dino_l2[:, 0, :] - dino_l2[:, 2, :], axis=1).astype(np.float32)

    cache_npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        X_dino=X_dino,
        X_conv=X_conv,
        target_ids=target_ids,
        clutter_raw=clutter_raw,
    )
    os.replace(tmp_path, cache_npz_path)
    print(f"[Stage 1] Saved real 4-view embeddings to {cache_npz_path}")

    return {
        "X_dino": X_dino.astype(np.float64),
        "X_conv": X_conv.astype(np.float64),
        "target_ids": target_ids,
        "clutter_raw": clutter_raw.astype(np.float64),
        "cache_sha256": sha256_file(cache_npz_path),
    }


# ==============================================================================
# 5. Stage 2: Build fixed split protocols (P1, P2, FP)
# ==============================================================================

def build_protocols_from_real_embeddings(
    drive_root: Path,
    emb_bundle: Dict[str, np.ndarray],
    out_dir: Path,
) -> Dict[str, Dict[str, object]]:
    manifests_dir = out_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    targets_csv = drive_root / "field_eval" / "cohorts" / "targets.csv"
    inputs_dir = drive_root / "code" / "field_eval" / "inputs"
    pd_parents_csv = inputs_dir / "plantdoc_eligible_parents_DRAFT.csv"
    fp_inv_csv = drive_root / "fieldplant" / "FieldPlant_inventory.csv"
    if not fp_inv_csv.exists():
        fp_inv_csv = inputs_dir / "fieldplant_eligible_parents_DRAFT.csv"

    targets_df = pd.read_csv(targets_csv)
    targets_df["target_idx"] = np.arange(len(targets_df), dtype=int)
    targets_df["image_id"] = targets_df["target_id"].astype(str)

    pd_parents_dedup = pd.read_csv(pd_parents_csv).drop_duplicates("sha256")
    fp_inv_df = pd.read_csv(fp_inv_csv).drop_duplicates("sha256")

    X_dino_all = emb_bundle["X_dino"]
    X_conv_all = emb_bundle["X_conv"]
    clutter_all = emb_bundle["clutter_raw"]

    pd_mask = (targets_df["dataset"] == "plantdoc").values
    fp_mask = (targets_df["dataset"] == "fieldplant").values

    pd_targets = targets_df[pd_mask].copy().reset_index(drop=True)
    fp_targets = targets_df[fp_mask].copy().reset_index(drop=True)

    pd_merged = pd_targets.merge(
        pd_parents_dedup[["sha256", "split", "n_boxes"]],
        on="sha256",
        how="left",
    )
    in_domain_mask = (pd_merged["primary"] == 1).values
    pd_in = pd_merged[in_domain_mask].copy().reset_index(drop=True)

    unique_pd_classes = sorted(pd_in["pv_class"].unique())
    pd_class_to_idx = {c: i for i, c in enumerate(unique_pd_classes)}
    pd_in["label_19"] = pd_in["pv_class"].map(pd_class_to_idx).astype(int)

    # Protocol P1: PlantDoc Official-Test Preserved (1,490 train / 246 val / 246 cal / 203 official test)
    p1_split = np.empty(len(pd_in), dtype=object)
    is_off_test = (pd_in["split"] == "test").values
    p1_split[is_off_test] = "test"

    rng_p1 = np.random.RandomState(20261109)
    train_pool_indices = np.where(~is_off_test)[0]
    for c_idx in range(len(unique_pd_classes)):
        c_pool = train_pool_indices[pd_in.loc[train_pool_indices, "label_19"].values == c_idx]
        perm = rng_p1.permutation(c_pool)
        n_c = len(perm)
        n_val = max(1, int(round(n_c * (246.0 / 1982.0))))
        n_cal = max(1, int(round(n_c * (246.0 / 1982.0))))
        p1_split[perm[:n_val]] = "val"
        p1_split[perm[n_val:n_val + n_cal]] = "cal"
        p1_split[perm[n_val + n_cal:]] = "train"
    pd_in["split_P1_official"] = p1_split

    # Protocol P2: PlantDoc Custom Stratified 70/10/10/10 (1,531 train / 218 val / 218 cal / 218 test)
    p2_split = np.empty(len(pd_in), dtype=object)
    rng_p2 = np.random.RandomState(20261110)
    for c_idx in range(len(unique_pd_classes)):
        c_all = np.where(pd_in["label_19"].values == c_idx)[0]
        perm = rng_p2.permutation(c_all)
        n_c = len(perm)
        n_test = max(1, int(round(n_c * 0.10)))
        n_cal = max(1, int(round(n_c * 0.10)))
        n_val = max(1, int(round(n_c * 0.10)))
        p2_split[perm[:n_test]] = "test"
        p2_split[perm[n_test:n_test + n_cal]] = "cal"
        p2_split[perm[n_test + n_cal:n_test + n_cal + n_val]] = "val"
        p2_split[perm[n_test + n_cal + n_val:]] = "train"
    pd_in["split_P2_stratified"] = p2_split

    pd_in_idx = pd_in["target_idx"].values
    pd_in["clutter_score"] = clutter_all[pd_in_idx]
    pd_in.to_csv(manifests_dir / "plantdoc_P1_P2_manifest.csv", index=False)

    # Protocol FP: FieldPlant v11 partitions disjoint by export file stem (1,273 images, 5 classes)
    fp_in = fp_targets[fp_targets["primary"] == 1].copy().reset_index(drop=True)
    if "stem_group" in fp_inv_df.columns:
        fp_in = fp_in.merge(fp_inv_df[["sha256", "stem_group"]], on="sha256", how="left")
    else:
        fp_in["stem_group"] = fp_in["relpath"].astype(str).apply(lambda s: s.split("_jpg")[0])

    unique_fp_classes = sorted(fp_in["label"].unique())
    fp_class_to_idx = {c: i for i, c in enumerate(unique_fp_classes)}
    fp_in["label_6"] = fp_in["label"].map(fp_class_to_idx).astype(int)

    fp_split = np.empty(len(fp_in), dtype=object)
    rng_fp = np.random.RandomState(20261111)
    for c_idx in range(len(unique_fp_classes)):
        c_rows = np.where(fp_in["label_6"].values == c_idx)[0]
        groups = fp_in.loc[c_rows, "stem_group"].unique()
        perm_groups = rng_fp.permutation(groups)
        n_total = len(c_rows)
        acc = 0
        for g in perm_groups:
            g_rows = c_rows[fp_in.loc[c_rows, "stem_group"].values == g]
            frac = acc / max(n_total, 1)
            if frac < 0.20:
                fp_split[g_rows] = "test"
            elif frac < 0.30:
                fp_split[g_rows] = "cal"
            elif frac < 0.40:
                fp_split[g_rows] = "val"
            else:
                fp_split[g_rows] = "train"
            acc += len(g_rows)
    fp_in["split_FP_grouped"] = fp_split

    fp_in_idx = fp_in["target_idx"].values
    fp_in["clutter_score"] = clutter_all[fp_in_idx]
    fp_in.to_csv(manifests_dir / "fieldplant_FP_grouped_manifest.csv", index=False)

    # Verify partitions are disjoint by export file stem; no pHash guarantee
    for s_a in ("train", "val", "cal", "test"):
        for s_b in ("train", "val", "cal", "test"):
            if s_a < s_b:
                ga = set(fp_in.loc[fp_in["split_FP_grouped"] == s_a, "stem_group"])
                gb = set(fp_in.loc[fp_in["split_FP_grouped"] == s_b, "stem_group"])
                assert len(ga.intersection(gb)) == 0, f"Group overlap between {s_a} and {s_b}"

    return {
        "P1": {
            "name": "PlantDoc_P1_Official",
            "X_dino": X_dino_all[pd_in_idx],
            "X_conv": X_conv_all[pd_in_idx],
            "y": pd_in["label_19"].values,
            "clutter": clutter_all[pd_in_idx],
            "splits": pd_in["split_P1_official"].values,
            "image_ids": pd_in["image_id"].values,
            "num_classes": len(unique_pd_classes),
        },
        "P2": {
            "name": "PlantDoc_P2_Stratified",
            "X_dino": X_dino_all[pd_in_idx],
            "X_conv": X_conv_all[pd_in_idx],
            "y": pd_in["label_19"].values,
            "clutter": clutter_all[pd_in_idx],
            "splits": pd_in["split_P2_stratified"].values,
            "image_ids": pd_in["image_id"].values,
            "num_classes": len(unique_pd_classes),
        },
        "FP": {
            "name": "FieldPlant_FP_Grouped",
            "X_dino": X_dino_all[fp_in_idx],
            "X_conv": X_conv_all[fp_in_idx],
            "y": fp_in["label_6"].values,
            "clutter": clutter_all[fp_in_idx],
            "splits": fp_in["split_FP_grouped"].values,
            "image_ids": fp_in["image_id"].values,
            "num_classes": len(unique_fp_classes),
        },
    }


# ==============================================================================
# 6. Stage 3: Strictly Neutral-Init Training & Evaluation of One Run
# ==============================================================================

def train_and_eval_one_run_v3(
    protocol_key: str,
    ds: Dict[str, object],
    config_name: str,
    stream_mode: str,
    seed: int,
    out_dir: Path,
) -> Dict[str, object]:
    X_dino = ds["X_dino"]
    X_conv = ds["X_conv"]
    y_all = ds["y"]
    clutter_all = ds["clutter"]
    splits = ds["splits"]
    image_ids = ds["image_ids"]
    num_classes = ds["num_classes"]

    tr_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    cal_idx = np.where(splits == "cal")[0]
    te_idx = np.where(splits == "test")[0]

    if stream_mode == "dual":
        X_raw = np.concatenate([X_dino, X_conv], axis=2)
    else:
        X_raw = X_dino.copy()

    # Standardize using ONLY training split statistics (zero leakage from val/cal/test)
    tr_mean = np.mean(X_raw[tr_idx, 0, :], axis=0, keepdims=True)
    tr_std = np.std(X_raw[tr_idx, 0, :], axis=0, keepdims=True) + 1e-6
    X_std = (X_raw - tr_mean[:, None, :]) / tr_std[:, None, :]
    # L2-normalize each view vector to unit sphere, scaled by sqrt(D)
    D = X_std.shape[2]
    X_norm = X_std / (np.linalg.norm(X_std, axis=2, keepdims=True) + 1e-8) * math.sqrt(D)

    c_tr_mean = float(np.mean(clutter_all[tr_idx]))
    c_tr_std = float(np.std(clutter_all[tr_idx])) + 1e-6
    c_norm = (clutter_all - c_tr_mean) / c_tr_std

    if stream_mode == "dual":
        d_P, d_B = 128, 32
    else:
        d_P, d_B = 64, 16

    model = CISDNetV3Model(
        config_name=config_name,
        D=D,
        num_classes=num_classes,
        d_P=d_P,
        d_B=d_B,
        seed=seed,
        lambda_cpi=CONFIRMED_HYPERPARAMETERS["lambda_cpi"],
        lambda_adv=CONFIRMED_HYPERPARAMETERS["lambda_adv"],
        swap_strength=CONFIRMED_HYPERPARAMETERS["swap_strength"],
        residual_alpha=CONFIRMED_HYPERPARAMETERS["residual_alpha"],
        beta_suppress=CONFIRMED_HYPERPARAMETERS["beta_suppress"],
        lambda_ortho_soft=CONFIRMED_HYPERPARAMETERS["lambda_ortho_soft"],
        weight_decay=CONFIRMED_HYPERPARAMETERS["weight_decay"],
    )

    # Record initial gate probabilities: strictly [0.25, 0.25, 0.25, 0.25]
    init_gate_probs = softmax_rows(model.b_gate[None, :])[0].tolist()

    # Standard ridge warm-start of W_cls on Training Split ONLY (identical across all linear & subspace models)
    f_tr, _ = model.pool_views(X_norm[tr_idx])
    if model.is_subspace_model:
        U_P_0, U_B_0 = model.get_subspace_bases()
        z_P_0 = f_tr @ U_P_0
        z_B_0 = f_tr @ U_B_0
        model.w_nuis_P = np.linalg.solve(
            z_P_0.T @ z_P_0 + 5.0 * np.eye(d_P), z_P_0.T @ c_norm[tr_idx]
        )
        model.w_nuis_B = np.linalg.solve(
            z_B_0.T @ z_B_0 + 5.0 * np.eye(d_B), z_B_0.T @ c_norm[tr_idx]
        )
        h_tr = (
            (1.0 - model.residual_alpha) * f_tr
            + model.residual_alpha * (z_P_0 @ U_P_0.T)
            - model.beta_suppress * (z_B_0 @ U_B_0.T)
        )
    else:
        h_tr = f_tr

    if config_name == "cosine_proto_baseline":
        # Class prototypes from training split only
        W_proto = np.zeros((D, num_classes), dtype=np.float64)
        for c in range(num_classes):
            c_mask = (y_all[tr_idx] == c)
            if np.any(c_mask):
                proto = np.mean(h_tr[c_mask], axis=0)
                W_proto[:, c] = proto / (np.linalg.norm(proto) + 1e-8)
        model.W_cls = W_proto
        model.b_cls = np.zeros(num_classes, dtype=np.float64)
    else:
        Y_tr_onehot = np.eye(num_classes, dtype=np.float64)[y_all[tr_idx]]
        ridge_lambda = 15.0 if stream_mode == "dual" else 10.0
        gram = h_tr.T @ h_tr + ridge_lambda * np.eye(D, dtype=np.float64)
        model.W_cls = np.linalg.solve(gram, h_tr.T @ (Y_tr_onehot - 1.0 / num_classes)) * 8.0
        model.b_cls = np.log(np.mean(Y_tr_onehot, axis=0) + 1e-4)

    epochs = CONFIRMED_HYPERPARAMETERS["epochs"]
    lr_cls = CONFIRMED_HYPERPARAMETERS["lr_cls"]
    lr_sub = CONFIRMED_HYPERPARAMETERS["lr_sub"]
    lr_gate = CONFIRMED_HYPERPARAMETERS["lr_gate"]
    beta1 = CONFIRMED_HYPERPARAMETERS["adam_beta1"]
    beta2 = CONFIRMED_HYPERPARAMETERS["adam_beta2"]
    eps_adam = CONFIRMED_HYPERPARAMETERS["adam_eps"]

    m_state: Dict[str, np.ndarray] = {}
    v_state: Dict[str, np.ndarray] = {}

    def _snapshot() -> Dict[str, np.ndarray]:
        snap = {
            "W_cls": model.W_cls.copy(),
            "b_cls": model.b_cls.copy(),
            "w_gate": model.w_gate.copy(),
            "b_gate": model.b_gate.copy(),
        }
        if model.use_joint_stiefel:
            snap["S_raw"] = model.stiefel.S_raw.copy()
            snap["B"] = model.stiefel.B.copy()
        elif model.is_subspace_model:
            snap["U_raw"] = model.U_raw.copy()
        if model.is_subspace_model:
            snap["w_nuis_P"] = model.w_nuis_P.copy()
            snap["b_nuis_P"] = np.array([model.b_nuis_P], dtype=np.float64)
            snap["w_nuis_B"] = model.w_nuis_B.copy()
            snap["b_nuis_B"] = np.array([model.b_nuis_B], dtype=np.float64)
        return snap

    def _restore(snap: Dict[str, np.ndarray]) -> None:
        model.W_cls = snap["W_cls"].copy()
        model.b_cls = snap["b_cls"].copy()
        model.w_gate = snap["w_gate"].copy()
        model.b_gate = snap["b_gate"].copy()
        if model.use_joint_stiefel:
            model.stiefel.S_raw = snap["S_raw"].copy()
            model.stiefel.B = snap["B"].copy()
        elif model.is_subspace_model:
            model.U_raw = snap["U_raw"].copy()
        if model.is_subspace_model:
            model.w_nuis_P = snap["w_nuis_P"].copy()
            model.b_nuis_P = float(snap["b_nuis_P"][0])
            model.w_nuis_B = snap["w_nuis_B"].copy()
            model.b_nuis_B = float(snap["b_nuis_B"][0])

    best_val_loss = 1e9
    best_val_acc = -1.0
    best_snap = _snapshot()

    for ep in range(1, epochs + 1):
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * (ep - 1) / epochs))
        cur_lr_cls = lr_cls * (0.15 + 0.85 * cos_decay)
        cur_lr_sub = lr_sub * (0.15 + 0.85 * cos_decay)
        cur_lr_gate = lr_gate * (0.15 + 0.85 * cos_decay)

        out_tr = model.forward_and_loss(
            X_norm[tr_idx], y_all[tr_idx], c_norm[tr_idx], compute_backward=True
        )
        grads = out_tr["grads"]

        for pname, g in grads.items():
            g_arr = np.array([g], dtype=np.float64) if isinstance(g, float) else g
            if pname not in m_state:
                m_state[pname] = np.zeros_like(g_arr)
                v_state[pname] = np.zeros_like(g_arr)
            m_state[pname] = beta1 * m_state[pname] + (1.0 - beta1) * g_arr
            v_state[pname] = beta2 * v_state[pname] + (1.0 - beta2) * (g_arr ** 2)
            m_hat = m_state[pname] / (1.0 - beta1 ** ep)
            v_hat = v_state[pname] / (1.0 - beta2 ** ep)
            step = m_hat / (np.sqrt(v_hat) + eps_adam)

            if pname in ("W_cls", "b_cls"):
                lr_use = cur_lr_cls
            elif pname in ("w_gate", "b_gate"):
                lr_use = cur_lr_gate
            else:
                lr_use = cur_lr_sub

            if pname == "W_cls":
                model.W_cls -= lr_use * step
            elif pname == "b_cls":
                model.b_cls -= lr_use * step
            elif pname == "w_gate":
                model.w_gate -= lr_use * step
            elif pname == "b_gate":
                model.b_gate -= lr_use * step
            elif pname == "S_raw":
                model.stiefel.S_raw -= lr_use * step
            elif pname == "B":
                model.stiefel.B -= lr_use * step
            elif pname == "U_raw":
                model.U_raw -= lr_use * step
            elif pname == "w_nuis_P":
                model.w_nuis_P -= lr_use * step
            elif pname == "b_nuis_P":
                model.b_nuis_P -= float(lr_use * step[0])
            elif pname == "w_nuis_B":
                model.w_nuis_B -= lr_use * step
            elif pname == "b_nuis_B":
                model.b_nuis_B -= float(lr_use * step[0])

        if ep % 4 == 0 or ep == epochs:
            out_val = model.forward_and_loss(
                X_norm[val_idx], y_all[val_idx], c_norm[val_idx], compute_backward=False
            )
            val_acc = float(np.mean(np.argmax(out_val["probs"], axis=1) == y_all[val_idx]))
            val_loss = float(out_val["loss_ce"])
            if (val_acc > best_val_acc + 1e-6) or (abs(val_acc - best_val_acc) <= 1e-6 and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_snap = _snapshot()

    _restore(best_snap)

    out_val = model.forward_and_loss(X_norm[val_idx], y_all[val_idx], c_norm[val_idx], compute_backward=False)
    out_cal = model.forward_and_loss(X_norm[cal_idx], y_all[cal_idx], c_norm[cal_idx], compute_backward=False)
    out_te = model.forward_and_loss(X_norm[te_idx], y_all[te_idx], c_norm[te_idx], compute_backward=False)

    # Temperature scaling on Validation NLL only
    best_temp = 1.0
    best_nll = 1e9
    for temp in (0.70, 0.85, 1.00, 1.15, 1.30):
        p_v = softmax_rows(out_val["logits"] / temp)
        nll = -float(np.mean(np.log(np.clip(p_v[np.arange(len(val_idx)), y_all[val_idx]], 1e-12, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    probs_val_f32 = softmax_rows(out_val["logits"] / best_temp).astype(np.float32)
    probs_cal_f32 = softmax_rows(out_cal["logits"] / best_temp).astype(np.float32)
    probs_test_f32 = softmax_rows(out_te["logits"] / best_temp).astype(np.float32)

    clut_val_f32 = clutter_all[val_idx].astype(np.float32)
    clut_cal_f32 = clutter_all[cal_idx].astype(np.float32)
    clut_test_f32 = clutter_all[te_idx].astype(np.float32)

    conf_res = evaluate_conformal_both_scores(
        probs_val=probs_val_f32.astype(np.float64),
        y_val=y_all[val_idx],
        clutter_val=clut_val_f32.astype(np.float64),
        probs_cal=probs_cal_f32.astype(np.float64),
        y_cal=y_all[cal_idx],
        clutter_cal=clut_cal_f32.astype(np.float64),
        probs_test=probs_test_f32.astype(np.float64),
        y_test=y_all[te_idx],
        clutter_test=clut_test_f32.astype(np.float64),
        alpha=0.05,
    )

    y_te = y_all[te_idx]
    preds_te = np.argmax(probs_test_f32, axis=1)
    top1_acc = float(np.mean(preds_te == y_te))
    macro_f1 = compute_macro_f1(y_te, preds_te, num_classes)
    aurcc = compute_aurcc(probs_test_f32.astype(np.float64), y_te)

    q_edges = np.quantile(clut_test_f32, [0.0, 0.25, 0.50, 0.75, 1.0])
    q_accs = []
    for q_i in range(4):
        if q_i < 3:
            q_mask = (clut_test_f32 >= q_edges[q_i]) & (clut_test_f32 < q_edges[q_i + 1])
        else:
            q_mask = (clut_test_f32 >= q_edges[q_i]) & (clut_test_f32 <= q_edges[q_i + 1])
        q_accs.append(float(np.mean(preds_te[q_mask] == y_te[q_mask])) if np.any(q_mask) else top1_acc)

    run_tag = f"{protocol_key}_{config_name}_seed{seed}"
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions_full_precision"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{run_tag}.npz"
    pred_path = pred_dir / f"{run_tag}_predictions.npz"

    ckpt_dict = {k: v for k, v in best_snap.items()}
    ckpt_dict["hyperparameters_json"] = np.array(json.dumps(CONFIRMED_HYPERPARAMETERS))
    ckpt_dict["temperature_val"] = np.array([best_temp], dtype=np.float64)
    np.savez_compressed(ckpt_path, **ckpt_dict)

    np.savez_compressed(
        pred_path,
        val_image_ids=image_ids[val_idx],
        val_y_true=y_all[val_idx].astype(np.int64),
        val_probs=probs_val_f32,
        val_clutter_scores=clut_val_f32,
        cal_image_ids=image_ids[cal_idx],
        cal_y_true=y_all[cal_idx].astype(np.int64),
        cal_probs=probs_cal_f32,
        cal_clutter_scores=clut_cal_f32,
        test_image_ids=image_ids[te_idx],
        test_y_true=y_te.astype(np.int64),
        test_probs=probs_test_f32,
        test_logits=(out_te["logits"] / best_temp).astype(np.float32),
        test_clutter_scores=clut_test_f32,
        test_sets_lac=conf_res["sets_lac"],
        test_sets_mod=conf_res["sets_mod"],
        lac_qhat=np.array([conf_res["lac_qhat"]], dtype=np.float64),
        mod_qhat=np.array([conf_res["mod_qhat"]], dtype=np.float64),
        mod_gamma=np.array([conf_res["mod_gamma"]], dtype=np.float64),
    )

    # Immediate replay audit from stored .npz
    reloaded = np.load(pred_path, allow_pickle=False)
    replay_conf = evaluate_conformal_both_scores(
        probs_val=reloaded["val_probs"].astype(np.float64),
        y_val=reloaded["val_y_true"],
        clutter_val=reloaded["val_clutter_scores"].astype(np.float64),
        probs_cal=reloaded["cal_probs"].astype(np.float64),
        y_cal=reloaded["cal_y_true"],
        clutter_cal=reloaded["cal_clutter_scores"].astype(np.float64),
        probs_test=reloaded["test_probs"].astype(np.float64),
        y_test=reloaded["test_y_true"],
        clutter_test=reloaded["test_clutter_scores"].astype(np.float64),
        alpha=0.05,
    )
    sets_exact_match = bool(
        np.array_equal(reloaded["test_sets_lac"], replay_conf["sets_lac"])
        and np.array_equal(reloaded["test_sets_mod"], replay_conf["sets_mod"])
    )
    assert sets_exact_match, f"Conformal replay mismatch on {run_tag}!"

    ortho_m = out_te["ortho_metrics"]
    return {
        "protocol": protocol_key,
        "config_name": config_name,
        "stream_mode": stream_mode,
        "seed": seed,
        "top1_acc": top1_acc,
        "macro_f1": macro_f1,
        "q1_acc": q_accs[0],
        "q2_acc": q_accs[1],
        "q3_acc": q_accs[2],
        "q4_acc": q_accs[3],
        "lac_coverage": conf_res["lac_coverage"],
        "lac_set_size": conf_res["lac_set_size"],
        "mod_coverage": conf_res["mod_coverage"],
        "mod_set_size": conf_res["mod_set_size"],
        "mod_gamma_val": conf_res["mod_gamma"],
        "aurcc": aurcc,
        "hsic_raw_unscaled": float(out_te["hsic_raw"]),
        "ortho_joint_stiefel_fro": float(ortho_m["joint_stiefel_fro_err"]),
        "ortho_cross_up_ub_fro": float(ortho_m["cross_orthogonality_fro_err"]),
        "init_gate_probs": init_gate_probs,
        "learned_gate_probs": np.asarray(out_te["attn_mean"], dtype=np.float64).tolist(),
        "temperature_val": float(best_temp),
        "checkpoint_file": f"checkpoints/{ckpt_path.name}",
        "checkpoint_sha256": sha256_file(ckpt_path),
        "prediction_file": f"predictions_full_precision/{pred_path.name}",
        "prediction_sha256": sha256_file(pred_path),
        "replay_conformal_sets_exact_match": sets_exact_match,
    }


# ==============================================================================
# 7. Main End-to-End Orchestrator
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("./data"),
                    help="Local data root with field_eval/cohorts and code/field_eval/inputs")
    ap.add_argument("--out-dir", type=Path, default=Path("./outputs"))
    ap.add_argument("--embeddings-npz", type=Path,
                    default=Path("./data/real_4view_backbone_embeddings_v3.npz"),
                    help="Verified frozen 4-view cache; if absent, extract from PNG views")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "code").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summaries_and_logs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), args.out_dir / "code" / "train_and_evaluate.py")
    emb_bundle = extract_or_load_real_4view_embeddings(
        drive_root=args.data_root,
        cache_npz_path=args.embeddings_npz,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    protocols = build_protocols_from_real_embeddings(
        drive_root=args.data_root,
        emb_bundle=emb_bundle,
        out_dir=args.out_dir,
    )
    per_run_records: List[Dict[str, object]] = []
    for p_key in ("P1", "P2", "FP"):
        ds = protocols[p_key]
        for cfg_name, stream_mode, _ in CONFIG_SPECS:
            for seed in SEEDS:
                rec = train_and_eval_one_run_v3(
                    protocol_key=p_key, ds=ds, config_name=cfg_name,
                    stream_mode=stream_mode, seed=seed, out_dir=args.out_dir,
                )
                per_run_records.append(rec)
            rlist = [r for r in per_run_records if r["protocol"] == p_key and r["config_name"] == cfg_name]
            print(f"[{p_key:2s}] {cfg_name:25s} Top-1={100*np.mean([r['top1_acc'] for r in rlist]):.2f}%")
    aggregated: Dict[str, Dict[str, object]] = {"P1": {}, "P2": {}, "FP": {}}
    for p_key in ("P1", "P2", "FP"):
        for cfg_name, stream_mode, cfg_desc in CONFIG_SPECS:
            rlist = [r for r in per_run_records if r["protocol"] == p_key and r["config_name"] == cfg_name]
            aggregated[p_key][cfg_name] = {
                "description": cfg_desc, "stream_mode": stream_mode,
                "top1_mean": float(np.mean([r["top1_acc"] for r in rlist])),
                "top1_std": float(np.std([r["top1_acc"] for r in rlist])),
                "macro_f1_mean": float(np.mean([r["macro_f1"] for r in rlist])),
                "macro_f1_std": float(np.std([r["macro_f1"] for r in rlist])),
                "q1_acc_mean": float(np.mean([r["q1_acc"] for r in rlist])),
                "q4_acc_mean": float(np.mean([r["q4_acc"] for r in rlist])),
                "lac_coverage_mean": float(np.mean([r["lac_coverage"] for r in rlist])),
                "lac_set_size_mean": float(np.mean([r["lac_set_size"] for r in rlist])),
                "mod_coverage_mean": float(np.mean([r["mod_coverage"] for r in rlist])),
                "mod_set_size_mean": float(np.mean([r["mod_set_size"] for r in rlist])),
                "aurcc_mean": float(np.mean([r["aurcc"] for r in rlist])),
                "ortho_cross_mean": float(np.mean([r["ortho_cross_up_ub_fro"] for r in rlist])),
                "ortho_cross_std": float(np.std([r["ortho_cross_up_ub_fro"] for r in rlist])),
                "hsic_raw_mean": float(np.mean([r["hsic_raw_unscaled"] for r in rlist])),
                "learned_gate_probs_mean": np.mean([r["learned_gate_probs"] for r in rlist], axis=0).tolist(),
            }
    exact_matches = sum(1 for r in per_run_records if r["replay_conformal_sets_exact_match"])
    ledger = {
        "package_version": "v3.0_true_backbone_embeddings_neutral_init",
        "training_script": "code/train_and_evaluate.py",
        "embedding_cache_sha256": emb_bundle["cache_sha256"],
        "confirmed_hyperparameters": CONFIRMED_HYPERPARAMETERS,
        "execution_elapsed_seconds": float(time.time() - t_start),
        "total_runs_executed": len(per_run_records),
        "conformal_sets_reproducible_runs_count": f"{exact_matches}/{len(per_run_records)}",
        "aggregated_3seed_results": aggregated,
        "per_run_records": per_run_records,
    }
    outfile = args.out_dir / "summaries_and_logs" / "v3_run_ledger_and_replay_audit.json"
    outfile.write_text(json.dumps(ledger, indent=2))
    print(f"Wrote {outfile}: {exact_matches}/{len(per_run_records)} conformal replay agreement")


if __name__ == "__main__":
    main()
