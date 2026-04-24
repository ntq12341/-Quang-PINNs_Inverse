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


def burgers_residual(u: torch.Tensor, xt: torch.Tensor, nu: float, forcing: torch.Tensor) -> torch.Tensor:
    grad_u = gradients(u, xt, order=1)
    u_x = grad_u[:, :1]
    u_t = grad_u[:, 1:2]
    u_xx = gradients(u_x, xt, order=1)[:, :1]
    return u_t + u * u_x - nu * u_xx - forcing


def analytic_initial_condition(x: torch.Tensor, nu: float = 0.1) -> torch.Tensor:
    return 2.0 * nu * math.pi * torch.sin(math.pi * x) / (2.0 + torch.cos(math.pi * x))


def analytic_forward_u(x: torch.Tensor, t: torch.Tensor, nu: float = 0.1) -> torch.Tensor:
    expo = torch.exp(-(math.pi**2) * nu * t)
    return 2.0 * nu * math.pi * expo * torch.sin(math.pi * x) / (2.0 + expo * torch.cos(math.pi * x))


def analytic_optimal_u(x: torch.Tensor, t: torch.Tensor, nu: float = 0.1) -> torch.Tensor:
    expo = torch.exp(-t)
    return 2.0 * nu * math.pi * expo * torch.sin(math.pi * x) / (2.0 + torch.cos(math.pi * x))


def target_step_function(x: torch.Tensor) -> torch.Tensor:
    """u_d(x) = 0.5 on [0.3, 0.7], and 0 elsewhere."""
    return torch.where(
        (x >= 0.3) & (x <= 0.7),
        0.5 * torch.ones_like(x),
        torch.zeros_like(x),
    )


def analytic_terminal_target(x: torch.Tensor, T: float = 1.0, nu: float = 0.1) -> torch.Tensor:
    return target_step_function(x)


def analytic_optimal_f(x: torch.Tensor, t: torch.Tensor, nu: float = 0.1) -> torch.Tensor:
    c = torch.cos(math.pi * x)
    s = torch.sin(math.pi * x)
    den = 2.0 + c
    exp_t = torch.exp(t)
    term = exp_t * (2.0 * (math.pi**2) * nu * c - 2.0 * (math.pi**2) * nu + c**2 + 4.0 * c + 4.0)
    term = term - 2.0 * (math.pi**2) * nu * (2.0 * c + 1.0)
    return -2.0 * math.pi * nu * torch.exp(-2.0 * t) * s * term / (den**3)


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


class BurgersForwardSolver:
    def __init__(self, nx: int = 101, nt: int = 400, L: float = 1.0, T: float = 1.0, nu: float = 0.1, substeps: int = 10) -> None:
        self.nx = nx
        self.nt = nt
        self.L = L
        self.T = T
        self.nu = nu
        self.x = np.linspace(0.0, L, nx, dtype=np.float64)
        self.t = np.linspace(0.0, T, nt + 1, dtype=np.float64)
        self.dx = L / (nx - 1)
        self.dt = T / nt
        self.substeps = substeps

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
        if u0 is None:
            states[0] = analytic_initial_condition(torch.tensor(self.x, dtype=torch.float64), nu=self.nu).detach().cpu().numpy()
        else:
            states[0] = np.asarray(u0, dtype=np.float64)
        states[:, 0] = 0.0
        states[:, -1] = 0.0
        for n in range(self.nt):
            u = states[n].copy()
            dt_sub = self.dt / self.substeps
            forcing_n = force[n].copy()
            for _ in range(self.substeps):
                ux = np.zeros_like(u)
                uxx = np.zeros_like(u)
                ux[1:-1] = (u[2:] - u[:-2]) / (2.0 * self.dx)
                uxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (self.dx**2)
                u[1:-1] = (
                    u[1:-1]
                    - dt_sub * u[1:-1] * ux[1:-1]
                    + dt_sub * self.nu * uxx[1:-1]
                    + dt_sub * forcing_n[1:-1]
                )
                u = np.nan_to_num(u, nan=0.0, posinf=10.0, neginf=-10.0)
                u = np.clip(u, -10.0, 10.0)
                u[0] = 0.0
                u[-1] = 0.0
            states[n + 1] = u
            states[n + 1, 0] = 0.0
            states[n + 1, -1] = 0.0
        return states

    def terminal_objective(self, terminal_state: np.ndarray, target: np.ndarray) -> float:
        diff = np.asarray(terminal_state, dtype=np.float64) - np.asarray(target, dtype=np.float64)
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
