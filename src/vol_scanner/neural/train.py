"""Training loop for the residual MLP, CPU only."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .residual_net import ResidualMLP


@dataclass
class TrainingResult:
    model: ResidualMLP
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    final_rmse: float = 0.0
    seed: int = 42


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_residual(
    features: np.ndarray,
    target: np.ndarray,
    cfg: dict,
) -> TrainingResult:
    """Train the MLP on (k, t, atm_vol) -> residual IV."""
    _set_seed(int(cfg.get("seed", 42)))

    features = features.astype(np.float32)
    target = target.astype(np.float32)

    n = features.shape[0]
    n_train = int(float(cfg["train_fraction"]) * n)
    perm = np.random.permutation(n)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:]

    x_train = torch.from_numpy(features[idx_train])
    y_train = torch.from_numpy(target[idx_train])
    x_val = torch.from_numpy(features[idx_val])
    y_val = torch.from_numpy(target[idx_val])

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
    )

    model = ResidualMLP(cfg.get("hidden_units"))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    loss_fn = nn.MSELoss()

    epochs = int(cfg["epochs"])
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _epoch in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            val_loss = float(loss_fn(pred_val, y_val).item())
        val_losses.append(val_loss)

    model.eval()
    with torch.no_grad():
        pred_val = model(x_val)
        rmse = float(torch.sqrt(torch.mean((pred_val - y_val) ** 2)).item())

    return TrainingResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        final_rmse=rmse,
        seed=int(cfg.get("seed", 42)),
    )


def predict_residual(
    model: ResidualMLP,
    k: np.ndarray,
    t: np.ndarray,
    atm_vol: np.ndarray,
) -> np.ndarray:
    model.eval()
    x = np.stack([k, t, atm_vol], axis=-1).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(x)).cpu().numpy()
    return out
