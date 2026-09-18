from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

import optimal_control as control_mod
from core import save_named_columns_csv


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep wJ for heat PINN optimal control.")
    parser.add_argument("--wj-list", type=float, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=float, default=np.pi)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--batch-residual", type=int, default=1000)
    parser.add_argument("--n-boundary-time", type=int, default=40)
    parser.add_argument("--n-initial", type=int, default=40)
    parser.add_argument("--n-terminal", type=int, default=100)
    parser.add_argument("--n-tikhonov", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--alpha-method", choices=["fixed", "lcurve"], default="lcurve")
    parser.add_argument("--alpha-list", type=float, nargs="*", default=None)
    parser.add_argument("--alpha-scan-epochs", type=int, default=3000)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-nx", type=int, default=100)
    parser.add_argument("--eval-nt", type=int, default=100)
    parser.add_argument("--rollout-nx", type=int, default=101)
    parser.add_argument("--rollout-nt", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/sweep_wj")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    # Range [0.01, 10]: giới hạn trên tránh wJ quá lớn khiến PDE loss
    # bị lấn át bởi objective, dẫn đến nghiệm không thỏa mãn PDE.
    wj_values = args.wj_list if args.wj_list else list(np.logspace(-3, 3, 7))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[float, float, float, float, float, float, float]] = []
    paths: list[str] = []
    for wj in wj_values:
        run_dir = outdir / f"wJ_{wj:.3e}"
        run_args = argparse.Namespace(
            seed=args.seed, device=args.device, L=args.L, T=args.T, epochs=args.epochs, lr=1e-3,
            lr_drop_epochs=[max(1, int(args.epochs / 2))], lr_drop_factor=0.1,
            n_residual=args.n_residual, batch_residual=args.batch_residual,
            n_boundary_time=args.n_boundary_time, n_initial=args.n_initial, n_terminal=args.n_terminal, n_tikhonov=args.n_tikhonov,
            hidden_layers=args.hidden_layers, hidden_width=args.hidden_width,
            wJ=float(wj), alpha=float(args.alpha), alpha_method=args.alpha_method,
            alpha_list=args.alpha_list, alpha_scan_epochs=args.alpha_scan_epochs,
            print_every=args.print_every, eval_nx=args.eval_nx, eval_nt=args.eval_nt,
            rollout_nx=args.rollout_nx, rollout_nt=args.rollout_nt, outdir=str(run_dir),
        )
        u_net, f_net, history, alpha = control_mod.train(run_args)
        control_mod.evaluate_and_save(u_net, f_net, history, alpha, run_args)
        loss_hist = np.atleast_2d(np.loadtxt(run_dir / "optimal_control_results" / "loss_history.csv", delimiter=",", skiprows=1))
        summary = np.atleast_2d(np.loadtxt(run_dir / "optimal_control_results" / "summary.csv", delimiter=",", skiprows=1))
        rows.append((wj, alpha, float(loss_hist[-1, 2] + loss_hist[-1, 3] + loss_hist[-1, 4]), float(loss_hist[-1, 5]), float(loss_hist[-1, 6]), float(summary[-1, 5]), float(summary[-1, 7])))
        paths.append(str(run_dir / "optimal_control_results"))
        print(f"[heat sweep] wJ={wj:.3e} alpha={alpha:.3e} LFBI={rows[-1][2]:.3e} LJ={rows[-1][3]:.3e} Reg={rows[-1][4]:.3e} Jroll={rows[-1][5]:.3e} Tik={rows[-1][6]:.3e}")
    arr = np.asarray(rows, dtype=np.float64)
    result_dir = outdir / "sweep_wj_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_named_columns_csv(result_dir / "summary.csv", {"wJ": arr[:, 0], "alpha": arr[:, 1], "loss_fbi": arr[:, 2], "loss_j": arr[:, 3], "loss_reg": arr[:, 4], "objective_rollout": arr[:, 5], "tikhonov_rollout": arr[:, 6]})
    (result_dir / "run_paths.csv").write_text("index,run_path\n" + "\n".join(f"{i},{p}" for i, p in enumerate(paths)) + "\n", encoding="utf-8")
    # Chọn best wJ: rollout objective thấp nhất TRONG SỐ các run có PDE loss
    # hội tụ tốt (loss_fbi < ngưỡng). Tránh chọn wJ quá lớn khiến PDE không
    # được thỏa mãn dù rollout objective trông thấp.
    pde_threshold = np.percentile(arr[:, 2], 50)  # loại nửa trên PDE loss
    valid_mask = arr[:, 2] <= pde_threshold
    if not np.any(valid_mask):
        valid_mask = np.ones(len(arr), dtype=bool)
    best = int(np.flatnonzero(valid_mask)[np.argmin(arr[valid_mask, 5])])
    best_run_dir = Path(paths[best])
    canonical_dir = outdir.parent / "optimal_control_results"
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    shutil.copytree(best_run_dir, canonical_dir)
    print(f"[heat sweep] copied best run to: {canonical_dir}")
    print(f"[heat sweep] best wJ by rollout objective: {arr[best, 0]:.3e}")


if __name__ == "__main__":
    main()