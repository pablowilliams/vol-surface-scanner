"""Small MLP for learning the residual on top of the SVI fit."""
from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, hidden_units: list[int] | None = None) -> None:
        super().__init__()
        hidden_units = list(hidden_units or [64, 64])
        layers: list[nn.Module] = []
        in_dim = 3
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
