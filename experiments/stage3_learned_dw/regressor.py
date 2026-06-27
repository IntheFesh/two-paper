"""
regressor.py
============

Tiny MLP regressor for D_w-hat -- instance-shape-agnostic.

Used by Phase 0 (corridor instances) and by Phase 1 (grid-trap instances)
with different feature extractors but the same model architecture and
training loop. The point of factoring out the model is so that "the
estimator architecture" is one knob, not two.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class DwMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # D_w >= 0 by construction, but a saturating ReLU on the OUTPUT
        # kills gradients when the optimiser initially sends predictions
        # negative -- training never starts. We let the network emit any
        # value during training and clamp at inference (LearnedDwEstimator).
        return self.net(x).squeeze(-1)


def train_dw_regressor(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden: int = 64,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
    verbose: bool = True,
    pos_weight: float = 1.0,
) -> Tuple[DwMLP, Dict[str, float]]:
    """Train the D_w-hat regressor.

    pos_weight: per-sample weight applied to positive (label > 0) examples
    during the MSE loss. Default 1.0 (unweighted). Use ~10-20 when the
    label distribution is heavily zero-biased (Phase 1 grid samples have
    ~0.5% positives) so the gradient does not collapse to "predict 0".
    """
    torch.manual_seed(seed)
    in_dim = X_train.shape[1]
    model = DwMLP(in_dim, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def weighted_mse(pred, target):
        w = torch.where(target > 1e-6,
                        torch.full_like(target, float(pos_weight)),
                        torch.ones_like(target))
        return ((pred - target) ** 2 * w).mean()

    n = X_train.shape[0]
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = weighted_mse(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.shape[0]
        ep_loss /= n
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            # Report PLAIN MSE for monitoring (so it's comparable across
            # pos_weight settings).
            val_mse = nn.functional.mse_loss(val_pred, y_val).item()
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}: train wMSE {ep_loss:.4f}  val MSE {val_mse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy()
    val_true = y_val.numpy()
    metrics = regression_metrics(val_pred, val_true)
    if verbose:
        print(f"  val metrics: {metrics}")
    return model, metrics


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(pred - true)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(((pred - true) ** 2).mean()))
    mask = true > 1e-6
    if mask.sum() > 0:
        rel = abs_err[mask] / true[mask]
        median_rel_err = float(np.median(rel))
        mean_rel_err = float(rel.mean())
    else:
        median_rel_err = 0.0
        mean_rel_err = 0.0
    pos = pred[mask]
    zero = pred[~mask]
    if len(pos) > 0 and len(zero) > 0:
        diff = pos[:, None] - zero[None, :]
        wins = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        auc = (wins + 0.5 * ties) / (len(pos) * len(zero))
    else:
        auc = 1.0
    return {
        "mae": mae,
        "rmse": rmse,
        "median_rel_err_on_positives": median_rel_err,
        "mean_rel_err_on_positives": mean_rel_err,
        "rank_auc_positive_vs_zero": float(auc),
        "n_positives": int(mask.sum()),
        "n_zeros": int((~mask).sum()),
    }
