from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]


def run_step(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full KS pipeline end-to-end.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-epochs", type=int, default=30000)
    parser.add_argument("--sweep-epochs", type=int, default=30000)
    parser.add_argument("--control-epochs", type=int, default=30000)
    parser.add_argument("--dal-iters", type=int, default=200)
    parser.add_argument("--n-residual", type=int, default=80000)
    parser.add_argument("--batch-residual", type=int, default=4000)
    parser.add_argument("--n-boundary-time", type=int, default=82)
    parser.add_argument("--n-initial", type=int, default=41)
    parser.add_argument("--eval-nx", type=int, default=128)
    parser.add_argument("--eval-dt", type=float, default=0.01)
    parser.add_argument("--dal-dt", type=float, default=0.01)
    parser.add_argument("--dal-step", type=float, default=1e-2)
    parser.add_argument("--wj", type=float, default=1e-3)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    py = args.python
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("[ks pipeline] CUDA requested but not available in this PyTorch build. Falling back to CPU.")
        device = "cpu"

    # run_step(
    #     [
    #         py,
    #         "pinns_optimal_control/pinns/ks/train_forward.py",
    #         "--device",
    #         device,
    #         "--epochs",
    #         str(args.forward_epochs),
    #         "--n-residual",
    #         str(args.n_residual),
    #         "--batch-residual",
    #         str(args.batch_residual),
    #         "--n-boundary-time",
    #         str(args.n_boundary_time),
    #         "--n-initial",
    #         str(args.n_initial),
    #         "--lr",
    #         "1e-3",
    #         "--lr-drop-epochs",
    #         "10000",
    #         "20000",
    #         "--lr-drop-factor",
    #         "0.1",
    #         "--ref-nx",
    #         str(args.eval_nx),
    #         "--ref-dt",
    #         str(args.eval_dt),
    #     ]
    # )

    run_step(
        [
            py,
            "pinns_optimal_control/pinns/ks/sweep_wj_3_3_2.py",
            "--device",
            device,
            "--wj-list",
            "1e-4",
            "1e-3",
            "1e-2",
            "1e-1",
            "1e0",
            "--epochs",
            str(args.sweep_epochs),
            "--n-residual",
            str(args.n_residual),
            "--batch-residual",
            str(args.batch_residual),
            "--n-boundary-time",
            str(args.n_boundary_time),
            "--n-initial",
            str(args.n_initial),
            "--eval-nx",
            str(args.eval_nx),
            "--eval-dt",
            str(args.eval_dt),
        ]
    )

    run_step(
        [
            py,
            "pinns_optimal_control/pinns/ks/train.py",
            "--device",
            device,
            "--wJ",
            str(args.wj),
            "--epochs",
            str(args.control_epochs),
            "--n-residual",
            str(args.n_residual),
            "--batch-residual",
            str(args.batch_residual),
            "--n-boundary-time",
            str(args.n_boundary_time),
            "--n-initial",
            str(args.n_initial),
            "--lr",
            "1e-3",
            "--lr-drop-epochs",
            "10000",
            "20000",
            "--lr-drop-factor",
            "0.1",
            "--eval-nx",
            str(args.eval_nx),
            "--eval-dt",
            str(args.eval_dt),
        ]
    )

    run_step(
        [
            py,
            "pinns_optimal_control/dal/ks/train.py",
            "--nx",
            str(args.eval_nx),
            "--dt",
            str(args.dal_dt),
            "--max-iters",
            str(args.dal_iters),
            "--init-step",
            str(args.dal_step),
        ]
    )

    run_step([py, "pinns_optimal_control/pinns/ks/plot_figures.py"])


if __name__ == "__main__":
    main()
