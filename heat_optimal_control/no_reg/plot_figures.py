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


def plot_overview(control_dir: Path, out_png: Path) -> None:
    field_u = _load_csv_2d(control_dir / "field_u.csv")
    field_f = _load_csv_2d(control_dir / "control_f.csv")
    loss = _load_csv_2d(control_dir / "loss_history.csv")
    term = _load_csv_2d(control_dir / "terminal_state.csv")
    roll_term = _load_csv_2d(control_dir / "rollout_terminal_state.csv")

    xx_u, tt_u, u_vals = _reshape_field(field_u, [2, 3, 4])
    u_true, u_pred, u_err = u_vals
    xx_f, tt_f, f_vals = _reshape_field(field_f, [2, 3, 4])
    f_true, f_pred, f_err = f_vals
    f_term_x = xx_f[-1]
    f_term_true = f_true[-1]
    f_term_pred = f_pred[-1]
    f_term_err = f_err[-1]

    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].set_title("(a) PDE/BC/IC Loss")
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].semilogy(loss[:, 0], loss[:, 5], label="J")
    axs[0, 1].set_title("(b) Objective Loss")
    axs[0, 1].legend(fontsize=8)

    axs[0, 2].plot(term[:, 0], term[:, 1], color="black", lw=2.2, label="target")
    axs[0, 2].plot(
        term[:, 0],
        term[:, 2],
        color="tab:orange",
        lw=1.8,
        ls="--",
        marker="o",
        markevery=max(1, len(term) // 16),
        ms=3.5,
        label="PINN",
    )
    axs[0, 2].plot(
        roll_term[:, 0],
        roll_term[:, 2],
        color="tab:green",
        lw=1.8,
        ls="-.",
        marker="s",
        markevery=max(1, len(roll_term) // 16),
        ms=3.2,
        label="rollout",
    )
    axs[0, 2].set_title("(c) Terminal State")
    axs[0, 2].legend(fontsize=8)

    c = axs[1, 0].contourf(xx_u, tt_u, u_true, levels=40, cmap="viridis")
    axs[1, 0].set_title("(d) True State $u^*$")
    fig.colorbar(c, ax=axs[1, 0])

    c = axs[1, 1].contourf(xx_u, tt_u, u_pred, levels=40, cmap="viridis")
    axs[1, 1].set_title("(e) PINN State")
    fig.colorbar(c, ax=axs[1, 1])

    c = axs[1, 2].contourf(xx_u, tt_u, u_err, levels=40, cmap="magma")
    axs[1, 2].set_title("(f) State Error")
    fig.colorbar(c, ax=axs[1, 2])

    axs[2, 0].plot(f_term_x, f_term_true, color="black", lw=2.0, label="true")
    axs[2, 0].plot(f_term_x, f_term_pred, color="tab:red", lw=1.8, ls="--", label="PINN")
    axs[2, 0].set_title(r"(g) PINN Control at $t=T$")
    axs[2, 0].legend(fontsize=8)

    axs[2, 1].plot(f_term_x, f_term_err, color="tab:purple", lw=2.0)
    axs[2, 1].set_title(r"(h) Control Error at $t=T$")

    axs[2, 2].axis("off")

    for ax in (axs[1, 0], axs[1, 1], axs[1, 2]):
        ax.set_xlabel("x")
        ax.set_ylabel("t")
    for ax in (axs[2, 0], axs[2, 1]):
        ax.set_xlabel("x")

    fig.subplots_adjust(wspace=0.30, hspace=0.28)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_wj_sweep(sweep_dir: Path, out_png: Path) -> None:
    summary_path = sweep_dir / "summary.csv"
    if not summary_path.exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.1, 0.5, "Missing sweep_wj summary.csv", fontsize=12)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return

    sweep = _load_csv_2d(summary_path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

    axs[0].loglog(sweep[:, 0], sweep[:, 1], marker="o", label=r"$L_{F/B/I}$")
    axs[0].loglog(sweep[:, 0], sweep[:, 2], marker="s", label=r"$L_J$")
    axs[0].set_title("(a) Loss Terms vs $w_J$")
    axs[0].legend(fontsize=8)

    best = int(np.argmin(sweep[:, 3]))
    axs[1].semilogx(sweep[:, 0], sweep[:, 3], marker="o", label="rollout objective")
    axs[1].scatter([sweep[best, 0]], [sweep[best, 3]], color="red", zorder=3, label="best")
    axs[1].set_title("(b) Rollout Objective")
    axs[1].legend(fontsize=8)

    for ax in axs:
        ax.set_xlabel(r"$w_J$")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for the heat no-reg optimal-control example.")
    parser.add_argument("--control-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/no_reg/optimal_control_results")
    parser.add_argument("--sweep-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/no_reg/sweep_wj/sweep_wj_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/no_reg/figures")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    control_dir = Path(args.control_dir)
    sweep_dir = Path(args.sweep_dir)

    plot_overview(control_dir, outdir / "fig_heat_no_reg_optimal_overview.png")
    plot_wj_sweep(sweep_dir, outdir / "fig_heat_no_reg_wj_sweep.png")
    print(f"Using control results from: {control_dir}")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
