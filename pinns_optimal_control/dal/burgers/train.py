from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.pinns.burgers.core import (
    analytic_initial_condition,
    analytic_terminal_target,
    save_named_columns_csv,
)


@dataclass
class DalHistory:
    objective: list[float]
    grad_norm: list[float]
    step_size: list[float]


class BurgersSpectralDAL:
    def __init__(self, nx: int = 256, L: float = 4.0, T: float = 5.0, nu: float = 0.01, dt: float = 1e-3) -> None:
        self.nx = nx
        self.L = L
        self.T = T
        self.nu = nu
        self.dt = dt
        self.nt = int(round(T / dt))
        self.x = np.linspace(0.0, L, nx, endpoint=False)
        self.dx = L / nx
        self.k = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.ik = 1j * self.k
        self.k2 = self.k**2
        self.den = 1.0 + nu * self.k2 * dt
        self.target = analytic_terminal_target(
            __import__("torch").tensor(self.x.reshape(-1, 1), dtype=__import__("torch").float32),
            T=T,
            nu=nu,
        ).detach().cpu().numpy().reshape(-1)

    def spectral_derivative(self, u: np.ndarray) -> np.ndarray:
        return np.fft.ifft(self.ik * np.fft.fft(u)).real

    def forward(self, u0: np.ndarray) -> np.ndarray:
        states = np.zeros((self.nt + 1, self.nx), dtype=np.float64)
        states[0] = u0
        u = u0.copy()
        for n in range(self.nt):
            nonlin = u * self.spectral_derivative(u)
            u_hat = np.fft.fft(u - self.dt * nonlin)
            u = np.fft.ifft(u_hat / self.den).real
            states[n + 1] = u
        return states

    def adjoint(self, states: np.ndarray) -> np.ndarray:
        mu = np.zeros_like(states)
        mu[0] = -states[-1] + self.target
        for m in range(self.nt):
            u = states[self.nt - m]
            lam = mu[m]
            conv = u * self.spectral_derivative(lam)
            lam_hat = np.fft.fft(lam + self.dt * conv)
            mu[m + 1] = np.fft.ifft(lam_hat / self.den).real
        return mu

    def objective(self, u_final: np.ndarray) -> float:
        return 0.5 * float(np.mean((u_final - self.target) ** 2))

    def optimize(
        self,
        max_iters: int,
        init_step: float,
        backtrack: float,
        min_step: float,
        grad_tol: float,
        print_every: int,
    ) -> tuple[np.ndarray, np.ndarray, DalHistory]:
        u0 = np.zeros(self.nx, dtype=np.float64)
        history = DalHistory(objective=[], grad_norm=[], step_size=[])

        for it in range(1, max_iters + 1):
            states = self.forward(u0)
            j_val = self.objective(states[-1])
            mu = self.adjoint(states)
            grad = -mu[-1]
            gnorm = float(np.linalg.norm(grad) / np.sqrt(self.nx))

            step = init_step
            u0_new = u0 - step * grad
            j_new = self.objective(self.forward(u0_new)[-1])
            while j_new > j_val and step > min_step:
                step *= backtrack
                u0_new = u0 - step * grad
                j_new = self.objective(self.forward(u0_new)[-1])

            u0 = u0_new
            history.objective.append(j_new)
            history.grad_norm.append(gnorm)
            history.step_size.append(step)

            if it % print_every == 0 or it == 1 or it == max_iters:
                print(f"[burgers DAL] iter={it:5d} J={j_new:.3e} |grad|={gnorm:.3e} step={step:.3e}")
            if gnorm < grad_tol:
                break

        states = self.forward(u0)
        return u0, states, history


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAL optimal control for Burgers example (paper section 3.2.2).")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--L", type=float, default=4.0)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--init-step", type=float, default=1.0)
    parser.add_argument("--backtrack", type=float, default=0.5)
    parser.add_argument("--min-step", type=float, default=1e-8)
    parser.add_argument("--grad-tol", type=float, default=1e-8)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/burgers/dal")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    solver = BurgersSpectralDAL(nx=args.nx, L=args.L, T=args.T, nu=args.nu, dt=args.dt)
    u0_opt, states, history = solver.optimize(
        max_iters=args.max_iters,
        init_step=args.init_step,
        backtrack=args.backtrack,
        min_step=args.min_step,
        grad_tol=args.grad_tol,
        print_every=args.print_every,
    )

    x = solver.x
    t = np.linspace(0.0, solver.T, solver.nt + 1)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    u_field = states.T
    u0_true = analytic_initial_condition(
        __import__("torch").tensor(x.reshape(-1, 1), dtype=__import__("torch").float32),
        nu=args.nu,
    ).detach().cpu().numpy().reshape(-1)
    target = solver.target
    final_state = states[-1]
    rel_u0 = np.linalg.norm(u0_opt - u0_true) / (np.linalg.norm(u0_true) + 1e-12)
    final_j = history.objective[-1]
    print(f"[burgers DAL] final J = {final_j:.3e}")
    print(f"[burgers DAL] relative L2(u0, analytic IC) = {rel_u0:.3e}")

    outdir = Path(args.outdir) / "dal_burgers_optimal_control"
    outdir.mkdir(parents=True, exist_ok=True)
    save_named_columns_csv(
        outdir / "field_u.csv",
        {
            "x": xx.reshape(-1),
            "t": tt.reshape(-1),
            "u": u_field.reshape(-1),
        },
    )
    save_named_columns_csv(
        outdir / "control_u0.csv",
        {
            "x": x,
            "u0_true": u0_true,
            "u0_opt": u0_opt,
            "abs_error": np.abs(u0_opt - u0_true),
        },
    )
    save_named_columns_csv(
        outdir / "terminal_state.csv",
        {
            "x": x,
            "uT_target": target,
            "uT_pred": final_state,
            "abs_error": np.abs(final_state - target),
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
            "rel_u0": np.array([rel_u0], dtype=np.float64),
        },
    )


if __name__ == "__main__":
    main()

