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


def plot_fig4_forward(forward_dir: Path, out_png: Path) -> None:
    field = np.loadtxt(forward_dir / "field_u.csv", delimiter=",", skiprows=1)
    loss = np.loadtxt(forward_dir / "loss_history.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(forward_dir / "test_error_history.csv", delimiter=",", skiprows=1)
    x = field[:, 0]
    t = field[:, 1]
    u_true = field[:, 2]
    u_pred = field[:, 3]
    n_x = len(np.unique(x))
    n_t = len(np.unique(t))
    xx = x.reshape(n_x, n_t)
    tt = t.reshape(n_x, n_t)
    u_true = u_true.reshape(n_x, n_t)
    u_pred = u_pred.reshape(n_x, n_t)
    err = np.abs(u_pred - u_true)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    axs[0, 0].semilogy(loss[:, 0], loss[:, 1], label="total")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].set_title("(a) Loss Components")
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].semilogy(test[:, 0], test[:, 1], color="tab:red")
    axs[0, 1].set_title("(b) Relative L2 Test Error")

    x_last = xx[:, -1]
    axs[0, 2].plot(x_last, u_true[:, -1], label="Analytical")
    axs[0, 2].plot(x_last, u_pred[:, -1], "--", label="PINN")
    axs[0, 2].set_title("(c) Final-Time Snapshot")
    axs[0, 2].legend(fontsize=8)

    c = axs[1, 0].contourf(xx, tt, u_true, levels=40, cmap="viridis")
    axs[1, 0].set_title("(d) Analytical $u_a$")
    fig.colorbar(c, ax=axs[1, 0])

    c = axs[1, 1].contourf(xx, tt, u_pred, levels=40, cmap="viridis")
    axs[1, 1].set_title("(e) PINN $u$")
    fig.colorbar(c, ax=axs[1, 1])

    c = axs[1, 2].contourf(xx, tt, err, levels=40, cmap="magma")
    axs[1, 2].set_title("(f) Absolute Error")
    fig.colorbar(c, ax=axs[1, 2])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_fig5_optimal(control_dir: Path, dal_dir: Path, sweep_dir: Path | None, out_png: Path) -> None:
    ctl_u0 = np.loadtxt(control_dir / "control_u0.csv", delimiter=",", skiprows=1)
    ctl_loss = np.loadtxt(control_dir / "loss_history.csv", delimiter=",", skiprows=1)
    ctl_term = np.loadtxt(control_dir / "terminal_state.csv", delimiter=",", skiprows=1)
    dal_u0 = np.loadtxt(dal_dir / "control_u0.csv", delimiter=",", skiprows=1)
    dal_hist = np.loadtxt(dal_dir / "history.csv", delimiter=",", skiprows=1)
    dal_term = np.loadtxt(dal_dir / "terminal_state.csv", delimiter=",", skiprows=1)
    sweep = np.loadtxt(sweep_dir / "summary.csv", delimiter=",", skiprows=1) if sweep_dir and sweep_dir.exists() else None

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    if sweep is not None:
        axs[0, 0].loglog(sweep[:, 0], sweep[:, 1], marker="o", label=r"$L_{F/B/I}$")
        axs[0, 0].loglog(sweep[:, 0], sweep[:, 2], marker="s", label=r"$L_J$")
        axs[0, 0].set_title("(a) Step-1 Line Search")
        axs[0, 0].legend(fontsize=8)

        best = int(np.argmin(sweep[:, 3]))
        axs[0, 1].semilogx(sweep[:, 0], sweep[:, 3], marker="o")
        axs[0, 1].scatter([sweep[best, 0]], [sweep[best, 3]], color="red", zorder=3, label="best")
        axs[0, 1].set_title("(b) Objective by $w_J$")
        axs[0, 1].legend(fontsize=8)
    else:
        axs[0, 0].axis("off")
        axs[0, 1].axis("off")

    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 1], label="total")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 2], label="PDE")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 3], label="BC")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 4], label="IC")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 5], label="J")
    axs[0, 2].set_title("(c) PINN Training")
    axs[0, 2].legend(fontsize=8)

    axs[1, 0].semilogy(dal_hist[:, 0], dal_hist[:, 1], color="tab:green")
    axs[1, 0].set_title("(d) DAL Objective History")

    axs[1, 1].plot(ctl_u0[:, 0], ctl_u0[:, 2], label="PINN")
    axs[1, 1].plot(dal_u0[:, 0], dal_u0[:, 2], label="DAL")
    axs[1, 1].plot(ctl_u0[:, 0], ctl_u0[:, 1], "--", label="Analytical IC")
    axs[1, 1].set_title("(e) Optimal Initial Condition")
    axs[1, 1].legend(fontsize=8)

    axs[1, 2].plot(ctl_term[:, 0], ctl_term[:, 2], label="PINN")
    axs[1, 2].plot(dal_term[:, 0], dal_term[:, 2], label="DAL")
    axs[1, 2].plot(ctl_term[:, 0], ctl_term[:, 1], "--", label="Target")
    axs[1, 2].set_title("(f) Final State")
    axs[1, 2].legend(fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-style figures for Burgers example.")
    parser.add_argument("--forward-dir", default="pinns_optimal_control/outputs/burgers/forward/forward_results")
    parser.add_argument(
        "--control-dir",
        default="pinns_optimal_control/outputs/burgers/optimal_control/optimal_control_results",
    )
    parser.add_argument("--dal-dir", default="pinns_optimal_control/outputs/burgers/dal/dal_burgers_optimal_control")
    parser.add_argument(
        "--sweep-dir",
        default="pinns_optimal_control/outputs/burgers/optimal_control/sweep_wj/sweep_wj_results",
    )
    parser.add_argument("--outdir", default="pinns_optimal_control/outputs/burgers/figures")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    plot_fig4_forward(Path(args.forward_dir), outdir / "fig4_forward_burgers.png")
    plot_fig5_optimal(Path(args.control_dir), Path(args.dal_dir), Path(args.sweep_dir), outdir / "fig5_optimal_burgers.png")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()

