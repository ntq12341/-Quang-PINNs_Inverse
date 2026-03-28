from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220


def plot_fig2_forward(forward_dir: Path, out_png: Path) -> None:
    field = np.loadtxt(forward_dir / "field_u.csv", delimiter=",", skiprows=1)
    loss = np.loadtxt(forward_dir / "loss_history.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(forward_dir / "test_error_history.csv", delimiter=",", skiprows=1)
    x = field[:, 0]
    y = field[:, 1]
    u_true = field[:, 2]
    u_pred = field[:, 3]
    err = np.abs(u_pred - u_true)
    n_side = int(round(np.sqrt(len(x))))
    x = x.reshape(n_side, n_side)
    y = y.reshape(n_side, n_side)
    u_true = u_true.reshape(n_side, n_side)
    u_pred = u_pred.reshape(n_side, n_side)
    err = err.reshape(n_side, n_side)
    loss_total = loss[:, 1]
    loss_pde = loss[:, 2]
    loss_bc = loss[:, 3]
    rel_epochs = test[:, 0]
    rel_hist = test[:, 1]

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = axs[0, 0]
    ax.semilogy(np.arange(1, len(loss_total) + 1), loss_total, label="total")
    ax.semilogy(np.arange(1, len(loss_pde) + 1), loss_pde, label="PDE")
    ax.semilogy(np.arange(1, len(loss_bc) + 1), loss_bc, label="BC")
    ax.set_title("(a) Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axs[0, 1]
    ax.semilogy(rel_epochs, rel_hist, color="tab:red")
    ax.set_title("(b) Relative L2 Test Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||u-u_a||/||u_a||")

    ax = axs[0, 2]
    c = ax.contourf(x, y, u_true, levels=40, cmap="viridis")
    ax.set_title("(c) Analytical $u_a$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c, ax=ax)

    ax = axs[1, 0]
    c = ax.contourf(x, y, u_pred, levels=40, cmap="viridis")
    ax.set_title("(d) PINN $u$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c, ax=ax)

    ax = axs[1, 1]
    c = ax.contourf(x, y, err, levels=40, cmap="magma")
    ax.set_title("(e) Absolute Error $|u-u_a|$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(c, ax=ax)

    axs[1, 2].axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_optimal(control_dir: Path, dal_dir: Path, sweep_dir: Path | None, out_png: Path) -> None:
    ctl_field = np.loadtxt(control_dir / "field_u.csv", delimiter=",", skiprows=1)
    ctl_f = np.loadtxt(control_dir / "control_f.csv", delimiter=",", skiprows=1)
    ctl_loss = np.loadtxt(control_dir / "loss_history.csv", delimiter=",", skiprows=1)
    dal_f = np.loadtxt(dal_dir / "control_f.csv", delimiter=",", skiprows=1)
    dal_hist = np.loadtxt(dal_dir / "history.csv", delimiter=",", skiprows=1)
    sweep = np.loadtxt(sweep_dir / "summary.csv", delimiter=",", skiprows=1) if sweep_dir and sweep_dir.exists() else None

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))

    ax = axs[0, 0]
    if sweep is not None:
        ax.loglog(sweep[:, 0], sweep[:, 1], marker="o", label=r"$L_{F/B}$")
        ax.loglog(sweep[:, 0], sweep[:, 2], marker="s", label=r"$L_J$")
        ax.set_xlabel(r"$w_J$")
        ax.set_ylabel("Loss at end of training")
        ax.set_title("(a) Step-1 Line Search")
        ax.legend()
    else:
        ax.text(0.1, 0.5, "Run sweep_wj_3_1_2.py\nfor panel (a).", fontsize=11)
        ax.axis("off")

    ax = axs[0, 1]
    if sweep is not None:
        wj = sweep[:, 0]
        j_hf = sweep[:, 3]
        best = int(np.argmin(j_hf))
        ax.semilogx(wj, j_hf, marker="o")
        ax.scatter([wj[best]], [j_hf[best]], color="red", zorder=3, label="best")
        ax.set_xlabel(r"$w_J$")
        ax.set_ylabel(r"$J(u_{HF}, c^*)$")
        ax.set_title("(b) Step-2 Cost Evaluation")
        ax.legend()
    else:
        ax.text(0.1, 0.5, "Run sweep_wj_3_1_2.py\nfor panel (b).", fontsize=11)
        ax.axis("off")

    ax = axs[0, 2]
    epochs = ctl_loss[:, 0]
    ax.semilogy(epochs, ctl_loss[:, 1], label="total")
    ax.semilogy(epochs, ctl_loss[:, 2], label="PDE")
    ax.semilogy(epochs, ctl_loss[:, 3], label="BC")
    ax.semilogy(epochs, ctl_loss[:, 4], label="J")
    ax.set_title("(c) PINN Training (Selected $w_J$)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.semilogy(dal_hist[:, 0], dal_hist[:, 1], color="tab:green")
    ax.set_title("(d) DAL Objective History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("J")

    ax = axs[1, 1]
    ax.plot(ctl_f[:, 0], ctl_f[:, 2], label="PINN")
    ax.plot(dal_f[:, 0], dal_f[:, 2], label="DAL")
    ax.plot(ctl_f[:, 0], ctl_f[:, 1], "--", label="Analytical")
    ax.set_title("(e) Optimal Top-Wall Potential")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()

    axs[1, 2].axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-style figures for Laplace example.")
    parser.add_argument(
        "--forward-dir",
        type=str,
        default="pinns_optimal_control/outputs/laplace/forward/forward_results",
    )
    parser.add_argument(
        "--control-dir",
        type=str,
        default="pinns_optimal_control/outputs/laplace/optimal_control/optimal_control_results",
    )
    parser.add_argument(
        "--dal-dir",
        type=str,
        default="pinns_optimal_control/outputs/laplace/dal/dal_laplace_optimal_control",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default="pinns_optimal_control/outputs/laplace/optimal_control/sweep_wj/sweep_wj_results",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="pinns_optimal_control/outputs/laplace/figures",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    plot_fig2_forward(Path(args.forward_dir), outdir / "fig2_forward_laplace.png")
    plot_fig3_optimal(
        Path(args.control_dir),
        Path(args.dal_dir),
        Path(args.sweep_dir) if args.sweep_dir else None,
        outdir / "fig3_optimal_laplace.png",
    )
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
