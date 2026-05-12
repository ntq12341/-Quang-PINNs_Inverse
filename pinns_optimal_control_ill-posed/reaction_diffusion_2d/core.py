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


def lhs_nd(n: int, low: tuple[float, ...], high: tuple[float, ...], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = len(low)
    u = (rng.random((n, dim)) + np.arange(n)[:, None]) / n
    out = np.empty_like(u)
    for j in range(dim):
        out[:, j] = u[rng.permutation(n), j]
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


def reaction_diffusion_residual(
    u: torch.Tensor,
    xyt: torch.Tensor,
    forcing: torch.Tensor,
    c: float,
) -> torch.Tensor:
    grad_u = gradients(u, xyt, order=1)
    u_x = grad_u[:, :1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]
    u_xx = gradients(u_x, xyt, order=1)[:, :1]
    u_yy = gradients(u_y, xyt, order=1)[:, 1:2]
    return u_t - u_xx - u_yy + c * u - forcing


def analytic_initial_condition(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def analytic_reaction_diffusion(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, c: float) -> torch.Tensor:
    return torch.exp(-(c + 2.0 * math.pi**2) * t) * analytic_initial_condition(x, y)


def analytic_terminal_target(x: torch.Tensor, y: torch.Tensor, c: float, T: float = 1.0) -> torch.Tensor:
    return analytic_reaction_diffusion(x, y, torch.full_like(x, T), c)


def analytic_min_energy_control(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x + y + t)


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


@dataclass
class OptimalControlHistory:
    total: list[float]
    pde: list[float]
    bc: list[float]
    ic: list[float]
    objective: list[float]
    regularization: list[float]
    alpha_scan: list[dict[str, float]]


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
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {
        "x": np.asarray(x_grid).reshape(-1),
        "y": np.asarray(y_grid).reshape(-1),
        "t": np.asarray(t_grid).reshape(-1),
    }
    for key, arr in values.items():
        rows[key] = np.asarray(arr).reshape(-1)
    return rows


def grid_l2_error_3d(u_pred: np.ndarray, u_true: np.ndarray, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    return float(np.sqrt(np.sum((u_pred - u_true) ** 2) * dx * dy * dt))


class ReactionDiffusion2DForwardSolver:
    def __init__(self, nx: int = 31, ny: int = 31, nt: int = 200, c: float = 1.0, T: float = 1.0) -> None:
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.c = c
        self.T = T
        self.x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
        self.y = np.linspace(0.0, 1.0, ny, dtype=np.float64)
        self.t = np.linspace(0.0, T, nt + 1, dtype=np.float64)
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        self.dt = T / nt
        self.nx_int = nx - 2
        self.ny_int = ny - 2
        self.n_int = self.nx_int * self.ny_int
        self.system_matrix_inv = np.linalg.inv(self._build_backward_euler_matrix())

    def _idx(self, i: int, j: int) -> int:
        return j * self.nx_int + i

    def _build_backward_euler_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.n_int, self.n_int), dtype=np.float64)
        rx = self.dt / (self.dx**2)
        ry = self.dt / (self.dy**2)
        diag = 1.0 + 2.0 * rx + 2.0 * ry + self.dt * self.c
        for j in range(self.ny_int):
            for i in range(self.nx_int):
                row = self._idx(i, j)
                matrix[row, row] = diag
                if i > 0:
                    matrix[row, self._idx(i - 1, j)] = -rx
                if i < self.nx_int - 1:
                    matrix[row, self._idx(i + 1, j)] = -rx
                if j > 0:
                    matrix[row, self._idx(i, j - 1)] = -ry
                if j < self.ny_int - 1:
                    matrix[row, self._idx(i, j + 1)] = -ry
        return matrix

    def rollout(self, forcing: np.ndarray | None = None, u0: np.ndarray | None = None) -> np.ndarray:
        if forcing is None:
            force = np.zeros((self.nt + 1, self.ny, self.nx), dtype=np.float64)
        else:
            force = np.asarray(forcing, dtype=np.float64)
            if force.shape == (self.nt, self.ny, self.nx):
                force = np.vstack([force, force[-1:]])
            if force.shape != (self.nt + 1, self.ny, self.nx):
                raise ValueError(f"forcing must have shape {(self.nt, self.ny, self.nx)} or {(self.nt + 1, self.ny, self.nx)}")
        xx, yy = np.meshgrid(self.x, self.y, indexing="xy")
        states = np.zeros((self.nt + 1, self.ny, self.nx), dtype=np.float64)
        states[0] = np.sin(math.pi * xx) * np.sin(math.pi * yy) if u0 is None else np.asarray(u0, dtype=np.float64)
        for n in range(self.nt):
            rhs = states[n, 1:-1, 1:-1] + self.dt * force[n + 1, 1:-1, 1:-1]
            states[n + 1, 1:-1, 1:-1] = (self.system_matrix_inv @ rhs.reshape(-1)).reshape(self.ny_int, self.nx_int)
        return states

    def terminal_objective(self, terminal_state: np.ndarray, target: np.ndarray) -> float:
        diff = np.asarray(terminal_state, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        return 0.5 * self.dx * self.dy * float(np.sum(diff**2))

    def h1_control_norm(self, forcing: np.ndarray) -> float:
        force = np.asarray(forcing, dtype=np.float64)
        if force.shape == (self.nt, self.ny, self.nx):
            force = np.vstack([force, force[-1:]])
        fx = np.zeros_like(force)
        fy = np.zeros_like(force)
        ft = np.zeros_like(force)
        fx[:, :, 1:-1] = (force[:, :, 2:] - force[:, :, :-2]) / (2.0 * self.dx)
        fy[:, 1:-1, :] = (force[:, 2:, :] - force[:, :-2, :]) / (2.0 * self.dy)
        ft[1:-1, :, :] = (force[2:, :, :] - force[:-2, :, :]) / (2.0 * self.dt)
        integrand = force**2 + fx**2 + fy**2 + ft**2
        return 0.5 * self.dx * self.dy * self.dt * float(np.sum(integrand))
