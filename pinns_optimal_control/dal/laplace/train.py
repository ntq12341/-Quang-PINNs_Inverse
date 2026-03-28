from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.pinns.laplace.core import (
    analytic_optimal_f,
    flatten_field_to_rows,
    save_named_columns_csv,
)


@dataclass
class DalHistory:
    objective: list[float]
    grad_norm: list[float]
    step_size: list[float]


class LaplaceDalOptimizer:
    """
    DAL for section 3.1.2:
      - state solve: A u = b(f)
      - adjoint solve: A^T lambda = dJ/du
      - gradient: dJ/df = J_f + (db/df)^T lambda
    """

    def __init__(self, nx: int, ny: int) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / nx
        self.dy = 1.0 / (ny - 1)
        self.x = np.linspace(0.0, 1.0, nx, endpoint=False)
        self._n_int = nx * (ny - 2)
        self._top_slice = slice((ny - 3) * nx, (ny - 2) * nx)
        self._rhs_top_coeff = -1.0 / (self.dy**2)
        self.A = self._build_matrix()
        self._lu = spla.splu(self.A.tocsc())

        self.bottom_bc = np.sin(2.0 * np.pi * self.x)
        self.qd = np.cos(2.0 * np.pi * self.x)
        self.b0 = self._build_rhs_base()

    def _idx(self, i: int, j: int) -> int:
        return (j - 1) * self.nx + i

    def _build_matrix(self) -> sp.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        c0 = -2.0 / (self.dx**2) - 2.0 / (self.dy**2)

        for j in range(1, self.ny - 1):
            for i in range(self.nx):
                k = self._idx(i, j)
                rows += [k, k, k]
                cols += [k, self._idx((i - 1) % self.nx, j), self._idx((i + 1) % self.nx, j)]
                data += [c0, 1.0 / (self.dx**2), 1.0 / (self.dx**2)]
                if j > 1:
                    rows.append(k)
                    cols.append(self._idx(i, j - 1))
                    data.append(1.0 / (self.dy**2))
                if j < self.ny - 2:
                    rows.append(k)
                    cols.append(self._idx(i, j + 1))
                    data.append(1.0 / (self.dy**2))
        return sp.csr_matrix((data, (rows, cols)), shape=(self._n_int, self._n_int))

    def _build_rhs_base(self) -> np.ndarray:
        b0 = np.zeros(self._n_int, dtype=np.float64)
        for i in range(self.nx):
            b0[self._idx(i, 1)] += -self.bottom_bc[i] / (self.dy**2)
        return b0

    def _solve_state(self, f: np.ndarray) -> np.ndarray:
        b = self.b0.copy()
        b[self._top_slice] += self._rhs_top_coeff * f
        return self._lu.solve(b)

    def objective_and_gradient(self, f: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        u_int = self._solve_state(f)
        u_top_minus = u_int[self._top_slice]
        q_top = (f - u_top_minus) / self.dy
        residual = q_top - self.qd
        j_val = float(np.mean(residual**2))

        # Direct part wrt f.
        j_f = (2.0 / self.nx) * residual * (1.0 / self.dy)

        # Adjoint rhs dJ/du (only top interior layer contributes).
        j_u = np.zeros(self._n_int, dtype=np.float64)
        j_u[self._top_slice] = (2.0 / self.nx) * residual * (-1.0 / self.dy)

        lam = self._lu.solve(j_u, trans="T")
        grad = j_f + self._rhs_top_coeff * lam[self._top_slice]
        return j_val, grad, u_int, q_top

    def optimize(
        self,
        max_iters: int,
        init_step: float,
        backtrack: float,
        min_step: float,
        grad_tol: float,
        print_every: int,
    ) -> tuple[np.ndarray, DalHistory, np.ndarray, np.ndarray]:
        f = np.zeros(self.nx, dtype=np.float64)
        hist = DalHistory(objective=[], grad_norm=[], step_size=[])

        for it in range(1, max_iters + 1):
            j_val, grad, u_int, q_top = self.objective_and_gradient(f)
            gnorm = float(np.linalg.norm(grad))

            step = init_step
            f_new = f - step * grad
            j_new, _, _, _ = self.objective_and_gradient(f_new)
            while j_new > j_val and step > min_step:
                step *= backtrack
                f_new = f - step * grad
                j_new, _, _, _ = self.objective_and_gradient(f_new)

            f = f_new
            hist.objective.append(j_new)
            hist.grad_norm.append(gnorm)
            hist.step_size.append(step)

            if it % print_every == 0 or it == 1 or it == max_iters:
                print(
                    f"[DAL] iter={it:5d} J={j_new:.3e} "
                    f"|grad|={gnorm:.3e} step={step:.3e}"
                )

            if gnorm < grad_tol:
                print(f"[DAL] early stop at iter={it} because grad norm < {grad_tol}")
                break

        j_val, grad, u_int, q_top = self.objective_and_gradient(f)
        _ = grad
        return f, hist, u_int, q_top

    def reconstruct_field(self, f: np.ndarray, u_int: np.ndarray) -> np.ndarray:
        u = np.zeros((self.nx, self.ny), dtype=np.float64)
        u[:, 0] = self.bottom_bc
        u[:, -1] = f
        for j in range(1, self.ny - 1):
            s = (j - 1) * self.nx
            e = j * self.nx
            u[:, j] = u_int[s:e]
        return u


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAL optimal control for Laplace example (paper section 3.1.2).")
    parser.add_argument("--nx", type=int, default=40, help="Number of points along x (periodic).")
    parser.add_argument("--ny", type=int, default=40, help="Number of points along y (includes top/bottom boundaries).")
    parser.add_argument("--max-iters", type=int, default=400)
    parser.add_argument("--init-step", type=float, default=1e-2)
    parser.add_argument("--backtrack", type=float, default=0.5)
    parser.add_argument("--min-step", type=float, default=1e-10)
    parser.add_argument("--grad-tol", type=float, default=1e-8)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/laplace/dal")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    opt = LaplaceDalOptimizer(nx=args.nx, ny=args.ny)
    f_opt, hist, u_int, q_top = opt.optimize(
        max_iters=args.max_iters,
        init_step=args.init_step,
        backtrack=args.backtrack,
        min_step=args.min_step,
        grad_tol=args.grad_tol,
        print_every=args.print_every,
    )
    u = opt.reconstruct_field(f_opt, u_int)

    x_t = np.asarray(opt.x, dtype=np.float32).reshape(-1, 1)
    f_ref = analytic_optimal_f(torch.tensor(x_t, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    rel_f = np.linalg.norm(f_opt - f_ref) / (np.linalg.norm(f_ref) + 1e-12)
    final_j = hist.objective[-1]
    print(f"[DAL] final J = {final_j:.3e}")
    print(f"[DAL] relative L2(f, f_analytic) = {rel_f:.3e}")

    outdir = Path(args.outdir) / "dal_laplace_optimal_control"
    outdir.mkdir(parents=True, exist_ok=True)
    x_grid, y_grid = np.meshgrid(opt.x, np.linspace(0.0, 1.0, opt.ny), indexing="xy")
    save_named_columns_csv(
        outdir / "field_u.csv",
        flatten_field_to_rows(
            x_grid,
            y_grid,
            {
                "u": u.T,
            },
        ),
    )
    save_named_columns_csv(
        outdir / "control_f.csv",
        {
            "x": opt.x,
            "f_ref": f_ref,
            "f_opt": f_opt,
            "abs_error": np.abs(f_opt - f_ref),
            "q_top": q_top,
            "q_d": opt.qd,
        },
    )
    save_named_columns_csv(
        outdir / "history.csv",
        {
            "iter": np.arange(1, len(hist.objective) + 1, dtype=np.int32),
            "J": np.array(hist.objective, dtype=np.float64),
            "grad_norm": np.array(hist.grad_norm, dtype=np.float64),
            "step": np.array(hist.step_size, dtype=np.float64),
        },
    )
    save_named_columns_csv(
        outdir / "summary.csv",
        {
            "final_J": np.array([final_j], dtype=np.float64),
            "rel_f": np.array([rel_f], dtype=np.float64),
        },
    )


if __name__ == "__main__":
    main()
