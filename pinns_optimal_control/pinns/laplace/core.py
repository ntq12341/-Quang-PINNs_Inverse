from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pinns_optimal_control.common import MLP


def lhs_2d(n: int, low: float = 0.0, high: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Simple Latin hypercube sampling in 2D."""
    rng = np.random.default_rng(seed)
    u = (rng.random((n, 2)) + np.arange(n)[:, None]) / n
    out = np.empty_like(u)
    for dim in range(2):
        out[:, dim] = u[rng.permutation(n), dim]
    return low + (high - low) * out


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
        self.net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=hidden_layers,
            hidden_width=hidden_width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.net(x_norm)


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


def laplace_residual(u: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    grad_u = gradients(u, xy, order=1)
    u_x = grad_u[:, :1]
    u_y = grad_u[:, 1:2]
    u_xx = gradients(u_x, xy, order=1)[:, :1]
    u_yy = gradients(u_y, xy, order=1)[:, 1:2]
    return u_xx + u_yy


def analytic_forward_u(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Solution for u(x,1)=sin(pi x), u(x,0)=u(0,y)=u(1,y)=0.
    return torch.sin(math.pi * x) * torch.sinh(math.pi * y) / math.sinh(math.pi)


def control_bottom_bc(x: torch.Tensor) -> torch.Tensor:
    # Periodic optimal-control case: u(x, 0) = sin(2 pi x).
    return torch.sin(2.0 * math.pi * x)


def desired_flux(x: torch.Tensor) -> torch.Tensor:
    # Consistent with the analytical optimal solution and periodicity in x.
    return torch.cos(2.0 * math.pi * x)


def analytic_optimal_f(x: torch.Tensor) -> torch.Tensor:
    sech = 1.0 / torch.cosh(torch.tensor(2.0 * math.pi, dtype=x.dtype, device=x.device))
    return sech * torch.sin(2.0 * math.pi * x) + (math.tanh(2.0 * math.pi) / (2.0 * math.pi)) * torch.cos(
        2.0 * math.pi * x
    )


def analytic_optimal_u(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sech = 1.0 / torch.cosh(torch.tensor(2.0 * math.pi, dtype=x.dtype, device=x.device))
    term1 = 0.5 * sech * torch.sin(2.0 * math.pi * x) * (
        torch.exp(2.0 * math.pi * (y - 1.0)) + torch.exp(2.0 * math.pi * (1.0 - y))
    )
    term2 = (sech / (4.0 * math.pi)) * torch.cos(2.0 * math.pi * x) * (
        torch.exp(2.0 * math.pi * y) - torch.exp(-2.0 * math.pi * y)
    )
    return term1 + term2


def relative_l2(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    num = torch.linalg.norm(u_pred - u_true).item()
    den = torch.linalg.norm(u_true).item()
    return num / (den + 1e-12)


@dataclass
class TrainHistory:
    total: list[float]
    pde: list[float]
    bc: list[float]
    obj: list[float]


def save_csv_bundle(outdir: Path | str, arrays: dict[str, np.ndarray | float | int]) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_lines = ["name,shape,dtype,csv_file"]

    for key, value in arrays.items():
        arr = np.asarray(value)
        csv_path = outdir / f"{key}.csv"
        if arr.ndim == 0:
            np.savetxt(csv_path, arr.reshape(1, 1), delimiter=",", fmt="%.12g")
        elif arr.ndim == 1:
            np.savetxt(csv_path, arr.reshape(-1, 1), delimiter=",", fmt="%.12g")
        elif arr.ndim == 2:
            np.savetxt(csv_path, arr, delimiter=",", fmt="%.12g")
        else:
            np.savetxt(csv_path, arr.reshape(arr.shape[0], -1), delimiter=",", fmt="%.12g")

        shape = "x".join(str(v) for v in arr.shape) if arr.shape else "scalar"
        manifest_lines.append(f"{key},{shape},{arr.dtype},{key}.csv")

    (outdir / "manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def load_csv_bundle(indir: Path | str) -> dict[str, np.ndarray]:
    indir = Path(indir)
    manifest = (indir / "manifest.csv").read_text(encoding="utf-8").strip().splitlines()
    arrays: dict[str, np.ndarray] = {}

    for line in manifest[1:]:
        name, shape_str, _dtype, csv_file = line.split(",")
        arr = np.loadtxt(indir / csv_file, delimiter=",")
        if shape_str == "scalar":
            arrays[name] = np.asarray(arr).reshape(())
            continue

        shape = tuple(int(v) for v in shape_str.split("x"))
        arrays[name] = np.asarray(arr).reshape(shape)

    return arrays


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

    matrix = np.column_stack(arrays)
    np.savetxt(path, matrix, delimiter=",", header=header, comments="", fmt="%.12g")


def flatten_field_to_rows(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {
        "x": np.asarray(x_grid).reshape(-1),
        "y": np.asarray(y_grid).reshape(-1),
    }
    for key, arr in values.items():
        rows[key] = np.asarray(arr).reshape(-1)
    return rows
