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


def _reshape_field(table: np.ndarray, value_cols: list[int]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    x = table[:, 0]
    t = table[:, 1]
    x_unique = np.unique(x)
    t_unique = np.unique(t)
    n_x = len(x_unique)
    n_t = len(t_unique)
    xx = x.reshape(n_t, n_x)
    tt = t.reshape(n_t, n_x)
    values = [table[:, col].reshape(n_t, n_x) for col in value_cols]
    return xx, tt, values


def plot_forward(forward_dir: Path, out_png: Path) -> None:
    field = _load_csv_2d(forward_dir / "field_u.csv")
    loss = _load_csv_2d(forward_dir / "loss_history.csv")
    test = _load_csv_2d(forward_dir / "test_error_history.csv")
    summary = _load_csv_2d(forward_dir / "summary.csv")

    xx, tt, values = _reshape_field(field, [2, 3, 4])
    u_true, u_pred, err = values

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))

    axs[0, 0].semilogy(loss[:, 0], loss[:, 1], label="total")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].set_title("(a) Loss Components")
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].semilogy(test[:, 0], test[:, 1], color="tab:red")
    axs[0, 1].set_title("(b) Relative L2 Test Error")

    axs[0, 2].plot(xx[-1], u_true[-1], label="Analytical")
    axs[0, 2].plot(xx[-1], u_pred[-1], "--", label="PINN")
    axs[0, 2].set_title("(c) Final-Time Snapshot")
    axs[0, 2].legend(fontsize=8)

    rel_l2 = float(summary[0, 0])
    grid_l2 = float(summary[0, 1]) if summary.shape[1] > 1 else float("nan")
    axs[0, 2].text(
        0.03,
        0.05,
        f"rel L2 = {rel_l2:.2e}\ngrid L2 = {grid_l2:.2e}",
        transform=axs[0, 2].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.3"},
    )

    c = axs[1, 0].contourf(xx, tt, u_true, levels=40, cmap="viridis")
    axs[1, 0].set_title("(d) Analytical $u_a$")
    fig.colorbar(c, ax=axs[1, 0])

    c = axs[1, 1].contourf(xx, tt, u_pred, levels=40, cmap="viridis")
    axs[1, 1].set_title("(e) PINN $u$")
    fig.colorbar(c, ax=axs[1, 1])

    c = axs[1, 2].contourf(xx, tt, err, levels=40, cmap="magma")
    axs[1, 2].set_title("(f) Absolute Error")
    fig.colorbar(c, ax=axs[1, 2])

    for ax in axs.flat:
        if ax in (axs[1, 0], axs[1, 1], axs[1, 2]):
            ax.set_xlabel("x")
            ax.set_ylabel("t")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for the heat forward example.")
    parser.add_argument("--forward-dir", default="pinns_optimal_control_ill-posed/outputs/heat/forward/forward_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/heat/figures")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    plot_forward(Path(args.forward_dir), outdir / "fig_forward_heat.png")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
