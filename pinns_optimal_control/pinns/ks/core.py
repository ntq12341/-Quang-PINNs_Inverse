from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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


def ks_initial_condition_np(x: np.ndarray, L: float = 50.0) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    return np.cos(2.0 * math.pi * x_arr / 10.0) + 1.0 / np.cosh((x_arr - L / 2.0) / 5.0)


def ks_initial_condition_torch(x: torch.Tensor, L: float = 50.0) -> torch.Tensor:
    return torch.cos(2.0 * math.pi * x / 10.0) + 1.0 / torch.cosh((x - L / 2.0) / 5.0)


def ks_residual(u: torch.Tensor, xt: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
    grad_u = gradients(u, xt, order=1)
    u_x = grad_u[:, :1]
    u_t = grad_u[:, 1:2]
    u_xx = gradients(u_x, xt, order=1)[:, :1]
    u_xxx = gradients(u_xx, xt, order=1)[:, :1]
    u_xxxx = gradients(u_xxx, xt, order=1)[:, :1]
    return u_t + u * u_x + u_xx + u_xxxx - forcing


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


class KSSpectralSolver:
    def __init__(
        self,
        nx: int = 128,
        L: float = 50.0,
        T: float = 10.0,
        dt: float = 0.01,
        sigma: float = 1.0,
        max_substep_dt: float = 0.002,
    ) -> None:
        self.nx = nx
        self.L = L
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.max_substep_dt = max_substep_dt
        self.nt = int(round(T / dt))
        self.dx = L / nx
        self.x = np.linspace(0.0, L, nx, endpoint=False)
        self.t = np.linspace(0.0, T, self.nt + 1)

        self.k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.ik = 1j * self.k
        self.k2 = self.k**2
        self.k4 = self.k**4
        self.linear = self.k2 - self.k4
        self.den = 1.0 - dt * self.linear
        self.n_substeps = max(1, int(np.ceil(self.dt / self.max_substep_dt)))
        self.dt_sub = self.dt / self.n_substeps
        self.den_sub = 1.0 - self.dt_sub * self.linear

    def spectral_derivative(self, u: np.ndarray) -> np.ndarray:
        return np.fft.ifft(self.ik * np.fft.fft(u)).real

    def step(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        u_next = u.copy()
        for _ in range(self.n_substeps):
            nonlin = 0.5 * self.spectral_derivative(u_next**2)
            rhs = u_next + self.dt_sub * (-nonlin + f)
            u_hat = np.fft.fft(rhs)
            u_next = np.fft.ifft(u_hat / self.den_sub).real
        return u_next

    def rollout(self, forcing: np.ndarray | None = None, u0: np.ndarray | None = None) -> np.ndarray:
        if forcing is None:
            forcing_arr = np.zeros((self.nt, self.nx), dtype=np.float64)
        else:
            forcing_arr = np.asarray(forcing, dtype=np.float64)
            if forcing_arr.shape == (self.nt + 1, self.nx):
                forcing_arr = forcing_arr[:-1]
            if forcing_arr.shape != (self.nt, self.nx):
                raise ValueError(f"forcing must have shape {(self.nt, self.nx)} or {(self.nt + 1, self.nx)}")

        if u0 is None:
            u = ks_initial_condition_np(self.x, L=self.L)
        else:
            u = np.asarray(u0, dtype=np.float64).copy()

        states = np.zeros((self.nt + 1, self.nx), dtype=np.float64)
        states[0] = u
        for n in range(self.nt):
            u = self.step(u, forcing_arr[n])
            states[n + 1] = u
        return states

    def objective(self, states: np.ndarray, forcing: np.ndarray) -> float:
        forcing_arr = np.asarray(forcing, dtype=np.float64)
        if forcing_arr.shape == (self.nt + 1, self.nx):
            forcing_arr = forcing_arr[:-1]
        integrand = states[:-1] ** 2 + self.sigma * forcing_arr**2
        return 0.5 * self.dt * self.dx * float(np.sum(integrand))

    def adjoint(self, states: np.ndarray) -> np.ndarray:
        mu = np.zeros_like(states, dtype=np.float64)
        for m in range(self.nt):
            u_rev = states[self.nt - m]
            lam = mu[m].copy()
            for _ in range(self.n_substeps):
                conv = u_rev * self.spectral_derivative(lam)
                rhs = lam + self.dt_sub * (conv - u_rev)
                lam_hat = np.fft.fft(rhs)
                lam = np.fft.ifft(lam_hat / self.den_sub).real
            mu[m + 1] = lam
        lambdas = np.zeros_like(states, dtype=np.float64)
        for n in range(self.nt + 1):
            lambdas[n] = mu[self.nt - n]
        return lambdas
