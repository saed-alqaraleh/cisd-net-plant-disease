#!/usr/bin/env python3
"""CISD-Net model, training objectives and numerical evaluation primitives.

Extracted verbatim from the frozen v3 experiment implementation.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

class JointStiefelCayleyTransform:
    """
    Parameterizes U = [U_P, U_B] in St(D, K) where K = d_P + d_B <= D via:
      U = (I_D - A) (I_D + A)^{-1} [I_K; 0]
    where A = [[S, -B^T], [B, 0]] is skew-symmetric (S = 0.5 * (S_raw - S_raw^T)).
    Guarantees ||U^T U - I_K||_F < 1e-13 and ||U_P^T U_B||_F < 1e-13 at every step.
    """

    def __init__(self, D: int, d_P: int, d_B: int, seed: int = 42, init_scale: float = 0.05):
        assert d_P + d_B <= D, f"Require d_P + d_B <= D, got {d_P}+{d_B} > {D}"
        self.D = D
        self.d_P = d_P
        self.d_B = d_B
        self.K = d_P + d_B
        rng = np.random.RandomState(seed)
        # Strictly neutral random initialization on the tangent space at a random permutation of I_D
        self.perm = rng.permutation(D)
        self.inv_perm = np.argsort(self.perm)
        self.S_raw = rng.randn(self.K, self.K).astype(np.float64) * init_scale
        self.B = rng.randn(self.D - self.K, self.K).astype(np.float64) * init_scale
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = 0.5 * (self.S_raw - self.S_raw.T)
        BtB = self.B.T @ self.B
        M = np.eye(self.K, dtype=np.float64) + S + BtB
        X = np.linalg.solve(M, np.eye(self.K, dtype=np.float64))
        U_top = 2.0 * X - np.eye(self.K, dtype=np.float64)
        if self.D > self.K:
            U_bot = -2.0 * (self.B @ X)
            U_canon = np.vstack([U_top, U_bot])
        else:
            U_canon = U_top
        U = U_canon[self.inv_perm, :]
        self._cache = {"S": S, "M": M, "X": X, "U_canon": U_canon, "U": U}
        U_P = U[:, :self.d_P]
        U_B = U[:, self.d_P:]
        return U, U_P, U_B

    def backward(self, grad_U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._cache["X"]
        grad_U_canon = grad_U[self.perm, :]
        grad_U_top = grad_U_canon[:self.K, :]
        if self.D > self.K:
            grad_U_bot = grad_U_canon[self.K:, :]
            grad_X = 2.0 * grad_U_top - 2.0 * (self.B.T @ grad_U_bot)
        else:
            grad_U_bot = np.zeros((0, self.K), dtype=np.float64)
            grad_X = 2.0 * grad_U_top

        grad_M = -(X.T @ grad_X @ X.T)
        grad_S_raw = 0.5 * (grad_M - grad_M.T)
        if self.D > self.K:
            grad_B = -2.0 * (grad_U_bot @ X.T) + self.B @ (grad_M + grad_M.T)
        else:
            grad_B = np.zeros_like(self.B)
        return grad_S_raw, grad_B

    @staticmethod
    def measure_orthogonality(U_P: np.ndarray, U_B: np.ndarray) -> Dict[str, float]:
        U = np.hstack([U_P, U_B])
        K = U.shape[1]
        d_P = U_P.shape[1]
        d_B = U_B.shape[1]
        return {
            "joint_stiefel_fro_err": float(np.linalg.norm(U.T @ U - np.eye(K), ord="fro")),
            "up_ortho_fro_err": float(np.linalg.norm(U_P.T @ U_P - np.eye(d_P), ord="fro")),
            "ub_ortho_fro_err": float(np.linalg.norm(U_B.T @ U_B - np.eye(d_B), ord="fro")),
            "cross_orthogonality_fro_err": float(np.linalg.norm(U_P.T @ U_B, ord="fro")),
        }


# ==============================================================================
# 2. Unbiased Empirical HSIC (Song et al., JMLR 2012, Eq. 5) & Metrics
# ==============================================================================

def compute_unbiased_hsic(Z_P: np.ndarray, Z_B: np.ndarray, eps: float = 1e-8) -> float:
    n = Z_P.shape[0]
    if n < 4:
        return 0.0
    sq_P = np.sum((Z_P[:, None, :] - Z_P[None, :, :]) ** 2, axis=-1)
    sq_B = np.sum((Z_B[:, None, :] - Z_B[None, :, :]) ** 2, axis=-1)
    triu_idx = np.triu_indices(n, k=1)
    sigma2_P = max(float(np.median(sq_P[triu_idx])), eps)
    sigma2_B = max(float(np.median(sq_B[triu_idx])), eps)
    K = np.exp(-sq_P / (2.0 * sigma2_P))
    L = np.exp(-sq_B / (2.0 * sigma2_B))
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    one = np.ones((n, 1), dtype=np.float64)
    term1 = float(np.trace(K @ L))
    term2 = float((one.T @ K @ one @ one.T @ L @ one)[0, 0]) / ((n - 1.0) * (n - 2.0))
    term3 = 2.0 * float((one.T @ K @ L @ one)[0, 0]) / (n - 2.0)
    hsic = (term1 + term2 - term3) / (n * (n - 3.0))
    return float(abs(hsic))


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        if np.sum(y_true == c) > 0:
            f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def compute_aurcc(probs: np.ndarray, y_true: np.ndarray) -> float:
    conf = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    errors = (preds != y_true).astype(np.float64)
    order = np.argsort(-conf, kind="mergesort")
    sorted_err = errors[order]
    cum_risk = np.cumsum(sorted_err) / np.arange(1, len(errors) + 1, dtype=np.float64)
    return float(np.mean(cum_risk))


def evaluate_conformal_both_scores(
    probs_val: np.ndarray,
    y_val: np.ndarray,
    clutter_val: np.ndarray,
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    clutter_cal: np.ndarray,
    probs_test: np.ndarray,
    y_test: np.ndarray,
    clutter_test: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, object]:
    c_min = float(np.min(clutter_val))
    c_max = float(np.max(clutter_val))
    denom = max(c_max - c_min, 1e-8)
    c_val_norm = np.clip((clutter_val - c_min) / denom, 0.0, 1.0)
    c_cal_norm = np.clip((clutter_cal - c_min) / denom, 0.0, 1.0)
    c_test_norm = np.clip((clutter_test - c_min) / denom, 0.0, 1.0)

    n_cal = len(y_cal)
    k_idx = min(max(int(math.ceil((n_cal + 1) * (1.0 - alpha))) - 1, 0), n_cal - 1)

    # 1. Standard LAC
    s_cal_lac = 1.0 - probs_cal[np.arange(n_cal), y_cal]
    q_lac = float(np.sort(s_cal_lac, kind="mergesort")[k_idx])
    sets_lac = (1.0 - probs_test) <= q_lac
    empty_lac = np.sum(sets_lac, axis=1) == 0
    if np.any(empty_lac):
        sets_lac[empty_lac, np.argmax(probs_test[empty_lac], axis=1)] = True

    # 2. Clutter-modulated LAC (gamma tuned exclusively on Val)
    best_gamma = 0.0
    best_val_sz = 1e9
    n_val = len(y_val)
    k_val = min(max(int(math.ceil((n_val + 1) * (1.0 - alpha))) - 1, 0), n_val - 1)
    for gamma in (0.0, 0.05, 0.10, 0.20):
        s_v = (1.0 - probs_val[np.arange(n_val), y_val]) / (1.0 + gamma * c_val_norm)
        q_v = float(np.sort(s_v, kind="mergesort")[k_val])
        sets_v = ((1.0 - probs_val) / (1.0 + gamma * c_val_norm[:, None])) <= q_v
        sz_v = float(np.mean(np.sum(sets_v, axis=1)))
        if sz_v < best_val_sz:
            best_val_sz = sz_v
            best_gamma = gamma

    s_cal_mod = (1.0 - probs_cal[np.arange(n_cal), y_cal]) / (1.0 + best_gamma * c_cal_norm)
    q_mod = float(np.sort(s_cal_mod, kind="mergesort")[k_idx])
    sets_mod = ((1.0 - probs_test) / (1.0 + best_gamma * c_test_norm[:, None])) <= q_mod
    empty_mod = np.sum(sets_mod, axis=1) == 0
    if np.any(empty_mod):
        sets_mod[empty_mod, np.argmax(probs_test[empty_mod], axis=1)] = True

    return {
        "lac_qhat": q_lac,
        "lac_coverage": float(np.mean(sets_lac[np.arange(len(y_test)), y_test])),
        "lac_set_size": float(np.mean(np.sum(sets_lac, axis=1))),
        "sets_lac": sets_lac,
        "mod_gamma": float(best_gamma),
        "mod_qhat": q_mod,
        "mod_coverage": float(np.mean(sets_mod[np.arange(len(y_test)), y_test])),
        "mod_set_size": float(np.mean(np.sum(sets_mod, axis=1))),
        "sets_mod": sets_mod,
    }


def build_counterfactual_swap_permutation(
    clutter_scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    identity_mode: bool = False,
) -> np.ndarray:
    n = clutter_scores.shape[0]
    if identity_mode or n <= 1:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(clutter_scores, kind="mergesort")
    rev_order = order[::-1]
    perm = np.empty(n, dtype=np.int64)
    perm[order] = rev_order
    fixed = np.where(perm == np.arange(n))[0]
    for idx in fixed:
        swap_partner = order[(np.where(order == idx)[0][0] + 1) % n]
        perm[idx], perm[swap_partner] = perm[swap_partner], perm[idx]
    return perm


# ==============================================================================
# 3. CISD-Net v3 Neutral-Initialization Model Class
# ==============================================================================

class CISDNetV3Model:
    """
    Strictly neutral-initialization CISD-Net model operating on real 4-view
    backbone embeddings X in R^{N x 4 x D}.
    """

    def __init__(
        self,
        config_name: str,
        D: int,
        num_classes: int,
        d_P: int = 64,
        d_B: int = 32,
        seed: int = 42,
        lambda_cpi: float = 0.35,
        lambda_adv: float = 0.20,
        swap_strength: float = 0.50,
        residual_alpha: float = 0.40,
        beta_suppress: float = 0.35,
        lambda_ortho_soft: float = 0.15,
        weight_decay: float = 0.01,
    ):
        self.config_name = config_name
        self.D = D
        self.num_classes = num_classes
        self.d_P = d_P
        self.d_B = d_B
        self.K = d_P + d_B
        self.seed = seed
        self.lambda_cpi = float(lambda_cpi)
        self.lambda_adv = float(lambda_adv)
        self.swap_strength = float(swap_strength)
        self.residual_alpha = float(residual_alpha)
        self.beta_suppress = float(beta_suppress)
        self.lambda_ortho_soft = float(lambda_ortho_soft)
        self.weight_decay = float(weight_decay)

        rng = np.random.RandomState(seed)

        self.use_joint_stiefel = config_name in (
            "ablation_stiefel_no_cpi",
            "ablation_stiefel_no_adv",
            "cisd_net_single",
            "cisd_net_dualstream",
        )
        self.use_soft_ortho = (config_name == "ablation_soft_ortho")
        self.use_qr_decoupled = (config_name == "ablation_qr_decoupled_v1")
        self.use_unconstrained_subspace = (config_name == "ablation_unconstrained")
        self.is_subspace_model = (
            self.use_joint_stiefel
            or self.use_soft_ortho
            or self.use_qr_decoupled
            or self.use_unconstrained_subspace
        )

        # Strictly uniform gate initialization: w_gate = 0, b_gate = [0, 0, 0, 0] -> [0.25, 0.25, 0.25, 0.25]
        self.w_gate = np.zeros(D, dtype=np.float64)
        self.b_gate = np.zeros(4, dtype=np.float64)
        self.gem_p = 2.25

        # Strictly neutral subspace initialization from `rng = np.random.RandomState(seed)` ONLY
        if self.use_joint_stiefel:
            self.stiefel = JointStiefelCayleyTransform(D=D, d_P=d_P, d_B=d_B, seed=seed, init_scale=0.05)
            self.U_raw = None
        elif self.is_subspace_model:
            self.stiefel = None
            self.U_raw = rng.randn(D, self.K).astype(np.float64) * (1.0 / math.sqrt(D))
        else:
            self.stiefel = None
            self.U_raw = None

        self.W_cls = rng.randn(D, num_classes).astype(np.float64) * (1.0 / math.sqrt(D))
        self.b_cls = np.zeros(num_classes, dtype=np.float64)
        self.cosine_scale = 12.0

        if self.is_subspace_model:
            self.w_nuis_P = rng.randn(d_P).astype(np.float64) * (1.0 / math.sqrt(d_P))
            self.b_nuis_P = 0.0
            self.w_nuis_B = rng.randn(d_B).astype(np.float64) * (1.0 / math.sqrt(d_B))
            self.b_nuis_B = 0.0

    def get_subspace_bases(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_joint_stiefel:
            _, U_P, U_B = self.stiefel.forward()
            return U_P, U_B
        elif self.use_qr_decoupled:
            Q_P, _ = np.linalg.qr(self.U_raw[:, :self.d_P])
            Q_B, _ = np.linalg.qr(self.U_raw[:, self.d_P:])
            return Q_P, Q_B
        else:
            return self.U_raw[:, :self.d_P], self.U_raw[:, self.d_P:]

    def pool_views(self, X_views: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N, V, D = X_views.shape
        if self.config_name == "erm_cls":
            attn = np.zeros((N, V), dtype=np.float64)
            attn[:, 0] = 1.0
            return X_views[:, 0, :], attn
        elif self.config_name == "gem_linear_baseline":
            attn = np.full((N, V), 1.0 / V, dtype=np.float64)
            shift = np.min(X_views, axis=(1, 2), keepdims=True) - 1e-3
            pos = X_views - shift
            pooled = np.mean(pos ** self.gem_p, axis=1) ** (1.0 / self.gem_p) + shift[:, 0, :]
            return pooled, attn
        elif self.config_name in ("cosine_proto_baseline", "dualstream_concat_ctrl"):
            attn = np.full((N, V), 1.0 / V, dtype=np.float64)
            pooled = np.mean(X_views, axis=1)
            return pooled, attn
        else:
            scores = (X_views @ self.w_gate) + self.b_gate[None, :]
            attn = softmax_rows(scores)
            pooled = (attn[:, None, :] @ X_views)[:, 0, :]
            return pooled, attn

    def forward_and_loss(
        self,
        X_views: np.ndarray,
        y: np.ndarray,
        clutter: np.ndarray,
        compute_backward: bool = True,
        cpi_identity_check: bool = False,
    ) -> Dict[str, object]:
        N = X_views.shape[0]
        f, attn = self.pool_views(X_views)

        if not self.is_subspace_model:
            if self.config_name == "cosine_proto_baseline":
                f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                W_norm = self.W_cls / (np.linalg.norm(self.W_cls, axis=0, keepdims=True) + 1e-8)
                logits = self.cosine_scale * (f_norm @ W_norm) + self.b_cls[None, :]
            else:
                logits = f @ self.W_cls + self.b_cls[None, :]

            probs = softmax_rows(logits)
            loss_ce = -float(np.mean(np.log(np.clip(probs[np.arange(N), y], 1e-12, 1.0))))
            total_loss = loss_ce + 0.5 * self.weight_decay * float(np.sum(self.W_cls ** 2))

            grads = {}
            if compute_backward:
                dlogits = probs.copy()
                dlogits[np.arange(N), y] -= 1.0
                dlogits /= N
                grads["W_cls"] = f.T @ dlogits + self.weight_decay * self.W_cls
                grads["b_cls"] = np.sum(dlogits, axis=0)

            return {
                "logits": logits,
                "probs": probs,
                "attn_mean": np.mean(attn, axis=0),
                "loss_total": total_loss,
                "loss_ce": loss_ce,
                "loss_cpi": 0.0,
                "loss_adv": 0.0,
                "loss_ortho": 0.0,
                "hsic_raw": 0.0,
                "ortho_metrics": {
                    "joint_stiefel_fro_err": 0.0,
                    "up_ortho_fro_err": 0.0,
                    "ub_ortho_fro_err": 0.0,
                    "cross_orthogonality_fro_err": 0.0,
                },
                "grads": grads,
            }

        U_P, U_B = self.get_subspace_bases()
        z_P = f @ U_P
        z_B = f @ U_B

        f_P = z_P @ U_P.T
        f_B = z_B @ U_B.T
        h = (1.0 - self.residual_alpha) * f + self.residual_alpha * f_P - self.beta_suppress * f_B
        logits = h @ self.W_cls + self.b_cls[None, :]
        probs = softmax_rows(logits)

        loss_ce = -float(np.mean(np.log(np.clip(probs[np.arange(N), y], 1e-12, 1.0))))

        active_cpi = self.lambda_cpi if self.config_name != "ablation_stiefel_no_cpi" else 0.0
        perm = build_counterfactual_swap_permutation(clutter, y, identity_mode=cpi_identity_check)
        delta_z_B = z_B[perm] - z_B
        h_swap = h + self.swap_strength * (delta_z_B @ U_B.T)
        logits_swap = h_swap @ self.W_cls + self.b_cls[None, :]
        probs_swap = softmax_rows(logits_swap)

        kl_per_sample = np.sum(
            probs * (np.log(np.clip(probs, 1e-12, 1.0)) - np.log(np.clip(probs_swap, 1e-12, 1.0))),
            axis=1,
        )
        loss_cpi = float(np.mean(kl_per_sample))

        active_adv = self.lambda_adv if self.config_name != "ablation_stiefel_no_adv" else 0.0
        pred_nuis_P = z_P @ self.w_nuis_P + self.b_nuis_P
        pred_nuis_B = z_B @ self.w_nuis_B + self.b_nuis_B
        err_P = pred_nuis_P - clutter
        err_B = pred_nuis_B - clutter
        loss_nuis_P = 0.5 * float(np.mean(err_P ** 2))
        loss_nuis_B = 0.5 * float(np.mean(err_B ** 2))
        loss_adv = loss_nuis_B - loss_nuis_P

        U_joint = np.hstack([U_P, U_B])
        gram_diff = (U_joint.T @ U_joint) - np.eye(self.K, dtype=np.float64)
        loss_ortho = 0.5 * float(np.sum(gram_diff ** 2)) if self.use_soft_ortho else 0.0

        total_loss = (
            loss_ce
            + active_cpi * loss_cpi
            + active_adv * loss_adv
            + (self.lambda_ortho_soft * loss_ortho if self.use_soft_ortho else 0.0)
            + 0.5 * self.weight_decay * float(np.sum(self.W_cls ** 2))
        )

        grads: Dict[str, np.ndarray] = {}
        if compute_backward:
            dlogits_ce = probs.copy()
            dlogits_ce[np.arange(N), y] -= 1.0
            dlogits_ce /= N
            dlogits_swap = active_cpi * (probs_swap - probs) / N

            grads["W_cls"] = (
                h.T @ dlogits_ce
                + h_swap.T @ dlogits_swap
                + self.weight_decay * self.W_cls
            )
            grads["b_cls"] = np.sum(dlogits_ce + dlogits_swap, axis=0)

            dh = (dlogits_ce + dlogits_swap) @ self.W_cls.T
            dh_swap_only = dlogits_swap @ self.W_cls.T

            df_P = self.residual_alpha * dh
            dU_P = f.T @ (df_P @ U_P) + df_P.T @ z_P

            df_B = -self.beta_suppress * dh
            dU_B_suppress = f.T @ (df_B @ U_B) + df_B.T @ z_B

            G_swap = self.swap_strength * dh_swap_only
            dU_B_term1 = G_swap.T @ delta_z_B
            d_delta_z_B = G_swap @ U_B
            dU_B_term2 = (f[perm] - f).T @ d_delta_z_B
            dU_B = dU_B_suppress + dU_B_term1 + dU_B_term2

            d_err_P = err_P / N
            d_err_B = err_B / N
            grads["w_nuis_P"] = z_P.T @ d_err_P + 1e-3 * self.w_nuis_P
            grads["b_nuis_P"] = float(np.sum(d_err_P))
            grads["w_nuis_B"] = z_B.T @ d_err_B + 1e-3 * self.w_nuis_B
            grads["b_nuis_B"] = float(np.sum(d_err_B))

            if active_adv > 0.0:
                dU_P += f.T @ (-active_adv * np.outer(d_err_P, self.w_nuis_P))
                dU_B += f.T @ (active_adv * np.outer(d_err_B, self.w_nuis_B))

            dU_joint = np.hstack([dU_P, dU_B])
            if self.use_soft_ortho:
                dU_joint += self.lambda_ortho_soft * (2.0 * (U_joint @ gram_diff))

            if self.use_joint_stiefel:
                grad_S_raw, grad_B = self.stiefel.backward(dU_joint)
                grads["S_raw"] = grad_S_raw
                grads["B"] = grad_B
            elif self.use_qr_decoupled:
                # Project gradient onto decoupled QR tangent spaces
                Q_P, Q_B = U_P, U_B
                gP = dU_P - Q_P @ (0.5 * (Q_P.T @ dU_P + dU_P.T @ Q_P))
                gB = dU_B - Q_B @ (0.5 * (Q_B.T @ dU_B + dU_B.T @ Q_B))
                grads["U_raw"] = np.hstack([gP, gB])
            else:
                grads["U_raw"] = dU_joint

            inv_perm_swap = np.argsort(perm)
            df_from_swap = (d_delta_z_B[inv_perm_swap] - d_delta_z_B) @ U_B.T

            df_total = (
                (1.0 - self.residual_alpha) * dh
                + (df_P @ U_P) @ U_P.T
                + (df_B @ U_B) @ U_B.T
                + df_from_swap
            )
            d_attn = (X_views @ df_total[:, :, None])[:, :, 0]
            d_scores = attn * (d_attn - np.sum(attn * d_attn, axis=1, keepdims=True))
            grads["w_gate"] = np.sum((d_scores[:, None, :] @ X_views)[:, 0, :], axis=0) + 1e-4 * self.w_gate
            grads["b_gate"] = np.sum(d_scores, axis=0)

        ortho_metrics = (
            {"joint_stiefel_fro_err": 0.0, "up_ortho_fro_err": 0.0, "ub_ortho_fro_err": 0.0, "cross_orthogonality_fro_err": 0.0}
            if compute_backward
            else JointStiefelCayleyTransform.measure_orthogonality(U_P, U_B)
        )
        hsic_raw = 0.0 if compute_backward else compute_unbiased_hsic(z_P[:min(N, 128)], z_B[:min(N, 128)])

        return {
            "logits": logits,
            "probs": probs,
            "attn_mean": np.mean(attn, axis=0),
            "z_P": z_P,
            "z_B": z_B,
            "loss_total": total_loss,
            "loss_ce": loss_ce,
            "loss_cpi": loss_cpi,
            "loss_adv": loss_adv,
            "loss_ortho": loss_ortho,
            "hsic_raw": hsic_raw,
            "ortho_metrics": ortho_metrics,
            "grads": grads,
        }
