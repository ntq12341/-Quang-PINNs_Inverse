from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.common import MLP


def lhs_2d(n: int, low: tuple[float, float], high: tuple[float, float], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = (rng.random((n, 2)) + np.arange(n)[:, None]) / n
    out = np.empty_like(u)
    for dim in range(2):
        out[:, dim] = u[rng.permutation(n), dim]
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    return low_arr + (high_arr - low_arr) * out


class NormalizedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        hidden_layers: int,
        hidden_width: int,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, -1))
        self.register_buffer("std", std.reshape(1, -1))
        self.net = MLP(in_dim=in_dim, out_dim=out_dim, hidden_layers=hidden_layers, hidden_width=hidden_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


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


def burgers_residual(u: torch.Tensor, xt: torch.Tensor, nu: float) -> torch.Tensor:
    grad_u = gradients(u, xt, order=1)
    u_x = grad_u[:, :1]
    u_t = grad_u[:, 1:2]
    u_xx = gradients(u_x, xt, order=1)[:, :1]
    return u_t + u * u_x - nu * u_xx


def analytic_burgers(x: torch.Tensor, t: torch.Tensor, nu: float = 0.1) -> torch.Tensor:
    expo = torch.exp(-(math.pi**2) * nu * t)
    num = 2.0 * nu * math.pi * expo * torch.sin(math.pi * x)
    den = 2.0 + expo * torch.cos(math.pi * x)
    return num / den


def analytic_initial_condition(x: torch.Tensor, nu: float = 0.1) -> torch.Tensor:
    return 2.0 * nu * math.pi * torch.sin(math.pi * x) / (2.0 + torch.cos(math.pi * x))


def relative_l2(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    num = torch.linalg.norm(u_pred - u_true).item()
    den = torch.linalg.norm(u_true).item()
    return num / (den + 1e-12)


@dataclass
class TrainHistory:
    total: list[float]
    pde: list[float]
    bc: list[float]
    ic: list[float]
    obj: list[float]


def save_named_columns_csv(path: Path | str, columns: dict[str, np.ndarray | list[float] | list[int] | float | int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns.keys())
    arrays = [np.asarray(columns[name]).reshape(-1) for name in names]
    n_rows = len(arrays[0]) if arrays else 0
    for arr in arrays:
        if len(arr) != n_rows:
            raise ValueError("All columns must have the same length.")
    header = ",".join(names)
    if n_rows == 0:
        path.write_text(header + "\n", encoding="utf-8")
        return
    np.savetxt(path, np.column_stack(arrays), delimiter=",", header=header, comments="", fmt="%.12g")


def flatten_field_to_rows(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {"x": np.asarray(x_grid).reshape(-1), "t": np.asarray(t_grid).reshape(-1)}
    for key, arr in values.items():
        rows[key] = np.asarray(arr).reshape(-1)
    return rows


def grid_l2_error(u_pred: np.ndarray, u_true: np.ndarray, x: np.ndarray, t: np.ndarray) -> float:
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    return float(np.sqrt(np.sum((u_pred - u_true) ** 2) * dx * dt))
