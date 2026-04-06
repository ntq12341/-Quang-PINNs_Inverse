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


def plot_forward(results_dir: Path, out_png: Path) -> None:
    field_u = _load_csv_2d(results_dir / "field_u.csv")
    loss = _load_csv_2d(results_dir / "loss_history.csv")
    test_err = _load_csv_2d(results_dir / "test_error_history.csv")
    summary = _load_csv_2d(results_dir / "summary.csv")

    xx, tt, u_vals = _reshape_field(field_u, [2, 3, 4])
    u_true, u_pred, u_err = u_vals
    final_idx = -1

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))

    axs[0, 0].semilogy(loss[:, 0], loss[:, 1], label="total")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].set_title("(a) Loss History")
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].semilogy(test_err[:, 0], test_err[:, 1], color="tab:red")
    axs[0, 1].set_title("(b) Relative L2 Error")
    axs[0, 1].set_xlabel("epoch")

    axs[0, 2].plot(xx[final_idx, :], u_true[final_idx, :], color="black", lw=2.0, label="true")
    axs[0, 2].plot(xx[final_idx, :], u_pred[final_idx, :], "--", color="tab:orange", lw=1.8, label="PINN")
    axs[0, 2].set_title(
        f"(c) Final-Time Slice\nrel L2={summary[0, 0]:.3e}, grid L2={summary[0, 1]:.3e}"
    )
    axs[0, 2].legend(fontsize=8)

    c = axs[1, 0].contourf(xx, tt, u_true, levels=40, cmap="viridis")
    axs[1, 0].set_title("(d) True State")
    fig.colorbar(c, ax=axs[1, 0])

    c = axs[1, 1].contourf(xx, tt, u_pred, levels=40, cmap="viridis")
    axs[1, 1].set_title("(e) PINN State")
    fig.colorbar(c, ax=axs[1, 1])

    c = axs[1, 2].contourf(xx, tt, u_err, levels=40, cmap="magma")
    axs[1, 2].set_title("(f) Absolute Error")
    fig.colorbar(c, ax=axs[1, 2])

    for ax in axs[1, :]:
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    fig.subplots_adjust(wspace=0.28, hspace=0.30)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for the Burgers forward example.")
    parser.add_argument("--results-dir", default="pinns_optimal_control_ill-posed/outputs/burgers/forward/forward_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/burgers/figures")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    plot_forward(results_dir, outdir / "fig_forward_burgers.png")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
