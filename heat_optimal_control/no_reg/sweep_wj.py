from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

import optimal_control as control_mod


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep wJ for heat PINN optimal control without regularization.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--L", type=float, default=np.pi)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epochs", type=int, nargs="*", default=[5000])
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--batch-residual", type=int, default=1000)
    parser.add_argument("--n-boundary-time", type=int, default=40)
    parser.add_argument("--n-initial", type=int, default=40)
    parser.add_argument("--n-terminal", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=50)
    parser.add_argument("--wJ-list", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3])
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-nx", type=int, default=100)
    parser.add_argument("--eval-nt", type=int, default=100)
    parser.add_argument("--rollout-nx", type=int, default=101)
    parser.add_argument("--rollout-nt", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/no_reg/sweep_wj")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    outdir = Path(args.outdir)
    rows: list[tuple[float, float, float, float]] = []
    paths: list[str] = []

    for wj in args.wJ_list:
        run_dir = outdir / f"wJ_{wj:.3e}"
        run_args = argparse.Namespace(
            seed=args.seed,
            device=args.device,
            L=float(args.L),
            T=float(args.T),
            epochs=args.epochs,
            lr=args.lr,
            lr_drop_epochs=args.lr_drop_epochs,
            lr_drop_factor=args.lr_drop_factor,
            n_residual=args.n_residual,
            batch_residual=args.batch_residual,
            n_boundary_time=args.n_boundary_time,
            n_initial=args.n_initial,
            n_terminal=args.n_terminal,
            hidden_layers=args.hidden_layers,
            hidden_width=args.hidden_width,
            wJ=float(wj),
            print_every=args.print_every,
            eval_nx=args.eval_nx,
            eval_nt=args.eval_nt,
            rollout_nx=args.rollout_nx,
            rollout_nt=args.rollout_nt,
            outdir=str(run_dir),
        )
        u_net, f_net, history = control_mod.train(run_args)
        control_mod.evaluate_and_save(u_net, f_net, history, run_args)
        loss_hist = np.atleast_2d(np.loadtxt(run_dir / "optimal_control_results" / "loss_history.csv", delimiter=",", skiprows=1))
        summary = np.atleast_2d(np.loadtxt(run_dir / "optimal_control_results" / "summary.csv", delimiter=",", skiprows=1))
        rows.append((wj, float(loss_hist[-1, 2] + loss_hist[-1, 3] + loss_hist[-1, 4]), float(loss_hist[-1, 5]), float(summary[-1, 4])))
        paths.append(str(run_dir / "optimal_control_results"))
        print(f"[heat no_reg sweep] wJ={wj:.3e} LFBI={rows[-1][1]:.3e} LJ={rows[-1][2]:.3e} Jroll={rows[-1][3]:.3e}")

    result_dir = outdir / "sweep_wj_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(rows, dtype=np.float64)
    control_mod.save_named_columns_csv(result_dir / "summary.csv", {"wJ": arr[:, 0], "loss_fbi": arr[:, 1], "loss_j": arr[:, 2], "objective_rollout": arr[:, 3]})
    run_paths_path = result_dir / "run_paths.csv"
    run_paths_path.write_text(
        "index,run_path\n" + "\n".join(f"{idx},{path}" for idx, path in enumerate(paths)) + "\n",
        encoding="utf-8",
    )

    best = int(np.argmin(arr[:, 3]))
    canonical_dir = outdir / "optimal_control_results"
    source_dir = Path(paths[best])
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    shutil.copytree(source_dir, canonical_dir)
    print(f"[heat no_reg sweep] best wJ by rollout objective: {arr[best, 0]:.3e}")


if __name__ == "__main__":
    main()
