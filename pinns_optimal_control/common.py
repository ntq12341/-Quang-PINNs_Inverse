from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: int = 4,
        hidden_width: int = 50,
    ) -> None:
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_width))
            layers.append(nn.Tanh())
            last = hidden_width
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gradients(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    g = y
    for _ in range(order):
        g = torch.autograd.grad(
            g,
            x,
            grad_outputs=torch.ones_like(g),
            create_graph=True,
            retain_graph=True,
        )[0]
    return g


def sample_uniform(n: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, 1, device=device)


@dataclass
class TrainLog:
    losses: list[float]
    pde: list[float]
    bc: list[float]
    obj: list[float]


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def analytic_burgers(x: np.ndarray, t: np.ndarray, nu: float = 0.01) -> np.ndarray:
    num = 2.0 * nu * math.pi * np.exp(-(math.pi**2) * nu * (t - 5.0)) * np.sin(math.pi * x)
    den = 2.0 + np.exp(-(math.pi**2) * nu * (t - 5.0)) * np.cos(math.pi * x)
    return num / den


def ks_initial_condition(x: np.ndarray, L: float = 50.0) -> np.ndarray:
    return np.cos(2.0 * np.pi * x / 10.0) + 1.0 / np.cosh((x - L / 2.0) / 5.0)

