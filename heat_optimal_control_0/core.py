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


SQRT2 = math.sqrt(2.0)


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


def heat_residual(u: torch.Tensor, xt: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
    grad_u = gradients(u, xt, order=1)
    u_x = grad_u[:, :1]
    u_t = grad_u[:, 1:2]
    u_xx = gradients(u_x, xt, order=1)[:, :1]
    return u_t - u_xx - forcing


def _solve_temporal_profile_coeffs() -> tuple[float, float, float]:
    matrix = np.array(
        [
            [SQRT2, -SQRT2, 1.0],
            [SQRT2 * math.exp(SQRT2), -SQRT2 * math.exp(-SQRT2), math.e],
            [
                (math.exp(SQRT2 + 1.0) - 1.0) / (SQRT2 + 1.0),
                (math.exp(1.0 - SQRT2) - 1.0) / (1.0 - SQRT2),
                0.5 * (math.e**2 - 1.0),
            ],
        ],
        dtype=np.float64,
    )
    rhs = np.array([0.0, 0.0, math.e], dtype=np.float64)
    coeffs = np.linalg.solve(matrix, rhs)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


COEFF_A, COEFF_B, COEFF_C = _solve_temporal_profile_coeffs()


def analytic_initial_condition(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def analytic_terminal_target(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


def analytic_temporal_profile(t: torch.Tensor) -> torch.Tensor:
    return (
        COEFF_A * torch.exp(SQRT2 * t)
        + COEFF_B * torch.exp(-SQRT2 * t)
        + COEFF_C * torch.exp(t)
    )


def analytic_state_amplitude(t: torch.Tensor) -> torch.Tensor:
    return (
        COEFF_A * (torch.exp(SQRT2 * t) - torch.exp(-t)) / (SQRT2 + 1.0)
        + COEFF_B * (torch.exp(-SQRT2 * t) - torch.exp(-t)) / (1.0 - SQRT2)
        + 0.5 * COEFF_C * (torch.exp(t) - torch.exp(-t))
    )


def analytic_optimal_u(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return analytic_state_amplitude(t) * torch.sin(x)


def analytic_optimal_f(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return analytic_temporal_profile(t) * torch.sin(x)


def relative_l2(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    num = torch.linalg.norm(u_pred - u_true).item()
    den = torch.linalg.norm(u_true).item()
    return num / (den + 1e-12)


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
    t_grid: np.ndarray,
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {"x": np.asarray(x_grid).reshape(-1), "t": np.asarray(t_grid).reshape(-1)}
    for key, arr in values.items():
        rows[key] = np.asarray(arr).reshape(-1)
    return rows


class HeatForwardSolver:
    def __init__(self, nx: int = 101, nt: int = 200, L: float = math.pi, T: float = 1.0) -> None:
        self.nx = nx
        self.nt = nt
        self.L = L
        self.T = T
        self.x = np.linspace(0.0, L, nx, dtype=np.float64)
        self.t = np.linspace(0.0, T, nt + 1, dtype=np.float64)
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.r = self.dt / (self.dx**2)
        self.n_int = nx - 2
        main = (1.0 + 2.0 * self.r) * np.ones(self.n_int, dtype=np.float64)
        off = -self.r * np.ones(self.n_int - 1, dtype=np.float64)
        self.a = off.copy()
        self.b = main.copy()
        self.c = off.copy()

    def _solve_tridiagonal(self, rhs: np.ndarray) -> np.ndarray:
        a = self.a.copy()
        b = self.b.copy()
        c = self.c.copy()
        d = rhs.astype(np.float64).copy()
        for i in range(1, len(b)):
            m = a[i - 1] / b[i - 1]
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]
        x = np.zeros_like(d)
        x[-1] = d[-1] / b[-1]
        for i in range(len(b) - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
        return x

    def rollout(self, forcing: np.ndarray | None = None, u0: np.ndarray | None = None) -> np.ndarray:
        if forcing is None:
            force = np.zeros((self.nt + 1, self.nx), dtype=np.float64)
        else:
            force = np.asarray(forcing, dtype=np.float64)
            if force.shape == (self.nt, self.nx):
                force = np.vstack([force, force[-1:]])
            if force.shape != (self.nt + 1, self.nx):
                raise ValueError(f"forcing must have shape {(self.nt, self.nx)} or {(self.nt + 1, self.nx)}")
        states = np.zeros((self.nt + 1, self.nx), dtype=np.float64)
        states[0] = np.zeros_like(self.x) if u0 is None else np.asarray(u0, dtype=np.float64)
        for n in range(self.nt):
            rhs = states[n, 1:-1] + self.dt * force[n + 1, 1:-1]
            states[n + 1, 1:-1] = self._solve_tridiagonal(rhs)
        return states

    def terminal_objective(self, terminal_state: np.ndarray, target: np.ndarray | None = None) -> float:
        if target is None:
            x_t = torch.tensor(self.x, dtype=torch.float32).reshape(-1, 1)
            target_vec = analytic_terminal_target(x_t).detach().cpu().numpy().reshape(-1)
        else:
            target_vec = np.asarray(target, dtype=np.float64)
        diff = np.asarray(terminal_state, dtype=np.float64) - target_vec
        return 0.5 * self.dx * float(np.sum(diff**2))

    def h1_control_norm(self, forcing: np.ndarray) -> float:
        force = np.asarray(forcing, dtype=np.float64)
        if force.shape == (self.nt, self.nx):
            force = np.vstack([force, force[-1:]])
        fx = np.zeros_like(force)
        ft = np.zeros_like(force)
        fx[:, 1:-1] = (force[:, 2:] - force[:, :-2]) / (2.0 * self.dx)
        ft[1:-1, :] = (force[2:, :] - force[:-2, :]) / (2.0 * self.dt)
        integrand = force**2 + fx**2 + ft**2
        return 0.5 * self.dx * self.dt * float(np.sum(integrand))
