from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.pinns.burgers import optimal_control_3_2_2 as control_mod
from pinns_optimal_control.pinns.burgers.core import save_named_columns_csv


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep wJ for Burgers PINN optimal control.")
    parser.add_argument("--wj-list", type=float, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-residual", type=int, default=20000)
    parser.add_argument("--batch-residual", type=int, default=2000)
    parser.add_argument("--n-boundary-time", type=int, default=41)
    parser.add_argument("--n-initial", type=int, default=41)
    parser.add_argument("--n-objective", type=int, default=41)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/burgers/optimal_control/sweep_wj")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    wj_values = args.wj_list if args.wj_list else list(np.logspace(-3, 6, 10))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[float, float, float, float]] = []
    paths: list[str] = []
    for wj in wj_values:
        run_dir = outdir / f"wJ_{wj:.3e}"
        run_args = argparse.Namespace(
            seed=args.seed,
            device=args.device,
            L=4.0,
            T=5.0,
            nu=0.01,
            epochs=args.epochs,
            lr=1e-3,
            lr_drop_epochs=[max(1, int(2 * args.epochs / 3)), max(1, int(5 * args.epochs / 6))],
            lr_drop_factor=0.1,
            n_residual=args.n_residual,
            batch_residual=args.batch_residual,
            n_boundary_time=args.n_boundary_time,
            n_initial=args.n_initial,
            n_objective=args.n_objective,
            wJ=float(wj),
            print_every=args.print_every,
            outdir=str(run_dir),
        )
        u_net, u0_net, history = control_mod.train(run_args)
        control_mod.evaluate_and_save(u_net, u0_net, history, run_args)

        loss_hist = np.loadtxt(run_dir / "optimal_control_results" / "loss_history.csv", delimiter=",", skiprows=1)
        summary = np.loadtxt(run_dir / "optimal_control_results" / "summary.csv", delimiter=",", skiprows=1)
        loss_fbi = float(loss_hist[-1, 2] + loss_hist[-1, 3] + loss_hist[-1, 4])
        loss_j = float(loss_hist[-1, 5])
        obj = float(summary[3] if summary.ndim == 1 else summary[0, 3])
        rows.append((wj, loss_fbi, loss_j, obj))
        paths.append(str(run_dir / "optimal_control_results"))
        print(f"[burgers sweep] wJ={wj:.3e} LFBI={loss_fbi:.3e} LJ={loss_j:.3e} J={obj:.3e}")

    arr = np.asarray(rows, dtype=np.float64)
    result_dir = outdir / "sweep_wj_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_named_columns_csv(
        result_dir / "summary.csv",
        {
            "wJ": arr[:, 0],
            "loss_fbi": arr[:, 1],
            "loss_j": arr[:, 2],
            "objective": arr[:, 3],
        },
    )
    (result_dir / "run_paths.csv").write_text(
        "index,run_path\n" + "\n".join(f"{i},{p}" for i, p in enumerate(paths)) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
