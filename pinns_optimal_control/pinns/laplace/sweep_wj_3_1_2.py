from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.pinns.laplace import optimal_control_3_1_2 as control_mod
from pinns_optimal_control.pinns.laplace.core import save_named_columns_csv


class LaplaceForwardHF:
    """High-fidelity finite-difference evaluator for J(u) with given top control f(x)."""

    def __init__(self, nx: int = 120, ny: int = 120) -> None:
        self.nx = nx
        self.ny = ny
        self.x = np.linspace(0.0, 1.0, nx, endpoint=False)
        self.dx = 1.0 / nx
        self.dy = 1.0 / (ny - 1)
        self.n_int = nx * (ny - 2)
        self.top_slice = slice((ny - 3) * nx, (ny - 2) * nx)
        self.bottom_bc = np.sin(2.0 * np.pi * self.x)
        self.qd = np.cos(2.0 * np.pi * self.x)
        self.A = self._build_matrix()
        self.lu = spla.splu(self.A.tocsc())
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
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_int, self.n_int))

    def _build_rhs_base(self) -> np.ndarray:
        b0 = np.zeros(self.n_int, dtype=np.float64)
        for i in range(self.nx):
            b0[self._idx(i, 1)] += -self.bottom_bc[i] / (self.dy**2)
        return b0

    def objective(self, f_top: np.ndarray) -> float:
        f_interp = np.interp(self.x, np.linspace(0.0, 1.0, len(f_top), endpoint=False), f_top, period=1.0)
        b = self.b0.copy()
        b[self.top_slice] += -f_interp / (self.dy**2)
        u_int = self.lu.solve(b)
        u_top_minus = u_int[self.top_slice]
        q_top = (f_interp - u_top_minus) / self.dy
        return float(np.mean((q_top - self.qd) ** 2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep wJ for Laplace PINN optimal control (paper Fig.3 a,b).")
    parser.add_argument("--wj-list", type=float, nargs="*", default=None, help="Explicit list of wJ values.")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--batch-residual", type=int, default=1000)
    parser.add_argument("--n-boundary", type=int, default=40)
    parser.add_argument("--n-objective", type=int, default=41)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/laplace/optimal_control/sweep_wj")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    wj_values = args.wj_list if args.wj_list else list(np.logspace(-3, 7, 11))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf = LaplaceForwardHF(nx=120, ny=120)
    loss_fb: list[float] = []
    loss_j: list[float] = []
    j_hf: list[float] = []
    f_paths: list[str] = []

    for wj in wj_values:
        run_dir = outdir / f"wJ_{wj:.3e}"
        run_args = argparse.Namespace(
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            lr=1e-3,
            lr_drop_epoch=max(1, args.epochs // 2),
            lr_drop_factor=0.1,
            n_residual=args.n_residual,
            batch_residual=args.batch_residual,
            n_boundary=args.n_boundary,
            n_objective=args.n_objective,
            wJ=float(wj),
            print_every=args.print_every,
            outdir=str(run_dir),
        )
        u_net, f_net, hist = control_mod.train(run_args)
        control_mod.evaluate_and_save(u_net, f_net, hist, run_args)
        loss_hist = np.loadtxt(run_dir / "optimal_control_results" / "loss_history.csv", delimiter=",", skiprows=1)
        control_f = np.loadtxt(run_dir / "optimal_control_results" / "control_f.csv", delimiter=",", skiprows=1)
        loss_fb.append(float(loss_hist[-1, 1] + loss_hist[-1, 2]))
        loss_j.append(float(loss_hist[-1, 3]))
        f_pred = control_f[:, 2].reshape(-1)
        j_hf.append(hf.objective(f_pred))
        f_paths.append(str(run_dir / "optimal_control_results"))
        print(
            f"[sweep] wJ={wj:.3e} "
            f"LF/B={loss_fb[-1]:.3e} LJ={loss_j[-1]:.3e} J_HF={j_hf[-1]:.3e}"
        )

    result_dir = outdir / "sweep_wj_results"
    save_named_columns_csv(
        result_dir / "summary.csv",
        {
            "wJ": np.asarray(wj_values, dtype=np.float64),
            "loss_fb": np.asarray(loss_fb, dtype=np.float64),
            "loss_j": np.asarray(loss_j, dtype=np.float64),
            "j_hf": np.asarray(j_hf, dtype=np.float64),
            "run_index": np.arange(len(f_paths), dtype=np.int32),
        },
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "run_paths.csv").write_text(
        "index,run_path\n" + "\n".join(f"{i},{path}" for i, path in enumerate(f_paths)) + "\n",
        encoding="utf-8",
    )
    best = int(np.argmin(j_hf))
    print(f"[sweep] best wJ by J_HF: {wj_values[best]:.3e}")


if __name__ == "__main__":
    main()
