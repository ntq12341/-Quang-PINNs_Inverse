from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220


def _load_csv_2d(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))


def _reshape_field(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = table[:, 0]
    y = table[:, 1]
    t = table[:, 2]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    t_unique = np.unique(t)
    n_y = len(y_unique)
    n_x = len(x_unique)
    n_t = len(t_unique)
    shape = (n_y, n_x, n_t)
    xx = x.reshape(shape)
    yy = y.reshape(shape)
    tt = t.reshape(shape)
    u_true = table[:, 3].reshape(shape)
    u_pred = table[:, 4].reshape(shape)
    err = table[:, 5].reshape(shape)
    return xx, yy, tt, u_true, u_pred, err


def plot_forward(forward_dir: Path, out_png: Path, snapshot_time: float) -> None:
    field = _load_csv_2d(forward_dir / "field_u.csv")
    loss = _load_csv_2d(forward_dir / "loss_history.csv")
    test = _load_csv_2d(forward_dir / "test_error_history.csv")
    summary = _load_csv_2d(forward_dir / "summary.csv")

    xx, yy, _tt, u_true, u_pred, err = _reshape_field(field)
    t_values = _tt[0, 0, :]
    snapshot_idx = int(np.argmin(np.abs(t_values - snapshot_time)))
    snapshot_t = float(t_values[snapshot_idx])
    mid_y_idx = u_true.shape[0] // 2

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))

    axs[0, 0].semilogy(loss[:, 0], loss[:, 1], label="total")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].set_title("(a) Loss Components")
    axs[0, 0].set_xlabel("epoch")
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].semilogy(test[:, 0], test[:, 1], color="tab:red")
    axs[0, 1].set_title("(b) Relative L2 Test Error")
    axs[0, 1].set_xlabel("epoch")

    x_line = xx[mid_y_idx, :, snapshot_idx]
    y_line = yy[mid_y_idx, 0, snapshot_idx]
    axs[0, 2].plot(x_line, u_true[mid_y_idx, :, snapshot_idx], label="Analytical")
    axs[0, 2].plot(x_line, u_pred[mid_y_idx, :, snapshot_idx], "--", label="PINN")
    axs[0, 2].set_title(f"(c) Slice y={y_line:.2f}, t={snapshot_t:.2f}")
    axs[0, 2].set_xlabel("x")
    axs[0, 2].legend(fontsize=8)
    rel_l2 = float(summary[0, 1]) if summary.shape[1] > 1 else float("nan")
    grid_l2 = float(summary[0, 2]) if summary.shape[1] > 2 else float("nan")
    axs[0, 2].text(
        0.03,
        0.05,
        f"rel L2 = {rel_l2:.2e}\ngrid L2 = {grid_l2:.2e}",
        transform=axs[0, 2].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.3"},
    )

    panels = [
        (axs[1, 0], u_true[:, :, snapshot_idx], f"(d) Analytical $u_a$, t={snapshot_t:.2f}", "viridis"),
        (axs[1, 1], u_pred[:, :, snapshot_idx], f"(e) PINN $u$, t={snapshot_t:.2f}", "viridis"),
        (axs[1, 2], err[:, :, snapshot_idx], f"(f) Absolute Error, t={snapshot_t:.2f}", "magma"),
    ]
    for ax, values, title, cmap in panels:
        c = ax.contourf(xx[:, :, snapshot_idx], yy[:, :, snapshot_idx], values, levels=40, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(c, ax=ax)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for the 2D reaction-diffusion forward example.")
    parser.add_argument("--forward-dir", default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d/forward/forward_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d/figures")
    parser.add_argument("--snapshot-time", type=float, default=0.25)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    plot_forward(Path(args.forward_dir), outdir / "fig_forward_reaction_diffusion_2d.png", args.snapshot_time)
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
