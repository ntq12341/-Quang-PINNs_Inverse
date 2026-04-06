from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.pinns.ks.core import KSSpectralSolver, flatten_field_to_rows, save_named_columns_csv


@dataclass
class DalHistory:
    objective: list[float]
    grad_norm: list[float]
    step_size: list[float]


class KSDalOptimizer:
    def __init__(self, nx: int = 128, L: float = 50.0, T: float = 10.0, dt: float = 0.01, sigma: float = 1.0) -> None:
        self.solver = KSSpectralSolver(nx=nx, L=L, T=T, dt=dt, sigma=sigma)
        self.sigma = sigma

    def gradient(self, states: np.ndarray, forcing: np.ndarray) -> np.ndarray:
        lam = self.solver.adjoint(states)
        return self.sigma * forcing - lam[:-1]

    def optimize(
        self,
        max_iters: int,
        init_step: float,
        backtrack: float,
        min_step: float,
        grad_tol: float,
        print_every: int,
    ) -> tuple[np.ndarray, np.ndarray, DalHistory]:
        forcing = np.zeros((self.solver.nt, self.solver.nx), dtype=np.float64)
        hist = DalHistory(objective=[], grad_norm=[], step_size=[])

        for it in range(1, max_iters + 1):
            states = self.solver.rollout(forcing=forcing)
            j_val = self.solver.objective(states, forcing)
            grad = self.gradient(states, forcing)
            gnorm = float(np.linalg.norm(grad) / np.sqrt(grad.size))

            step = init_step
            forcing_new = forcing - step * grad
            states_new = self.solver.rollout(forcing=forcing_new)
            j_new = self.solver.objective(states_new, forcing_new)
            while j_new > j_val and step > min_step:
                step *= backtrack
                forcing_new = forcing - step * grad
                states_new = self.solver.rollout(forcing=forcing_new)
                j_new = self.solver.objective(states_new, forcing_new)

            forcing = forcing_new
            hist.objective.append(j_new)
            hist.grad_norm.append(gnorm)
            hist.step_size.append(step)

            if it % print_every == 0 or it == 1 or it == max_iters:
                print(f"[ks DAL] iter={it:5d} J={j_new:.3e} |grad|={gnorm:.3e} step={step:.3e}")
            if gnorm < grad_tol:
                break

        states = self.solver.rollout(forcing=forcing)
        return forcing, states, hist


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAL optimal control for Kuramoto-Sivashinsky example (paper section 3.3.2).")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--L", type=float, default=50.0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=60)
    parser.add_argument("--init-step", type=float, default=1e-2)
    parser.add_argument("--backtrack", type=float, default=0.5)
    parser.add_argument("--min-step", type=float, default=1e-8)
    parser.add_argument("--grad-tol", type=float, default=1e-6)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/ks/dal")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    optimizer = KSDalOptimizer(nx=args.nx, L=args.L, T=args.T, dt=args.dt, sigma=args.sigma)
    forcing, states, history = optimizer.optimize(
        max_iters=args.max_iters,
        init_step=args.init_step,
        backtrack=args.backtrack,
        min_step=args.min_step,
        grad_tol=args.grad_tol,
        print_every=args.print_every,
    )

    solver = optimizer.solver
    final_j = solver.objective(states, forcing)
    final_norm = float(np.sqrt(solver.dx * np.sum(states[-1] ** 2)))
    print(f"[ks DAL] final J = {final_j:.3e}")
    print(f"[ks DAL] final-state L2 norm = {final_norm:.3e}")

    outdir = Path(args.outdir) / "dal_ks_optimal_control"
    outdir.mkdir(parents=True, exist_ok=True)
    xx, tt = np.meshgrid(solver.x, solver.t, indexing="xy")
    save_named_columns_csv(
        outdir / "field_u.csv",
        flatten_field_to_rows(
            xx,
            tt,
            {
                "u": states,
            },
        ),
    )
    save_named_columns_csv(
        outdir / "control_f.csv",
        flatten_field_to_rows(
            xx[:-1],
            tt[:-1],
            {
                "f": forcing,
            },
        ),
    )
    save_named_columns_csv(
        outdir / "terminal_state.csv",
        {
            "x": solver.x,
            "uT": states[-1],
        },
    )
    save_named_columns_csv(
        outdir / "history.csv",
        {
            "iter": np.arange(1, len(history.objective) + 1, dtype=np.int32),
            "J": np.array(history.objective, dtype=np.float64),
            "grad_norm": np.array(history.grad_norm, dtype=np.float64),
            "step": np.array(history.step_size, dtype=np.float64),
        },
    )
    save_named_columns_csv(
        outdir / "summary.csv",
        {
            "final_J": np.array([final_j], dtype=np.float64),
            "final_state_l2": np.array([final_norm], dtype=np.float64),
        },
    )


if __name__ == "__main__":
    main()
