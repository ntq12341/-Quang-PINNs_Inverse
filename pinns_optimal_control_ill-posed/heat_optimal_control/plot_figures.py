from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220


def _load_csv_2d(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))


def _resolve_best_control_dir(control_dir: Path, sweep_dir: Path, explicit_control_dir: bool) -> Path:
    if explicit_control_dir:
        return control_dir

    summary_path = sweep_dir / "summary.csv"
    run_paths_path = sweep_dir / "run_paths.csv"
    if not summary_path.exists() or not run_paths_path.exists():
        return control_dir

    sweep = _load_csv_2d(summary_path)
    best = int(np.argmin(sweep[:, 5]))
    run_paths = np.genfromtxt(run_paths_path, delimiter=",", skip_header=1, dtype=str)
    run_paths = np.atleast_2d(run_paths)
    best_run = Path(run_paths[best, 1])
    return best_run


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
    summary = _load_csv_2d(control_dir / "summary.csv")

    xx_u, tt_u, u_vals = _reshape_field(field_u, [2, 3, 4])
    u_true, u_pred, u_err = u_vals
    xx_f, tt_f, f_vals = _reshape_field(field_f, [2, 3, 4])
    f_true, f_pred, f_err = f_vals

    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].set_title("(a) PDE/BC/IC Loss")
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].semilogy(loss[:, 0], loss[:, 5], label="J")
    axs[0, 1].semilogy(loss[:, 0], loss[:, 6], label="H1")
    axs[0, 1].set_title("(b) Objective/Regularization")
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

    c = axs[2, 0].contourf(xx_f, tt_f, f_true, levels=40, cmap="coolwarm")
    axs[2, 0].set_title("(g) True Control $f^*$")
    fig.colorbar(c, ax=axs[2, 0])

    c = axs[2, 1].contourf(xx_f, tt_f, f_pred, levels=40, cmap="coolwarm")
    axs[2, 1].set_title("(h) PINN Control")
    fig.colorbar(c, ax=axs[2, 1])

    c = axs[2, 2].contourf(xx_f, tt_f, f_err, levels=40, cmap="magma")
    axs[2, 2].set_title("(i) Control Error")
    fig.colorbar(c, ax=axs[2, 2])

    for ax in (axs[1, 0], axs[1, 1], axs[1, 2], axs[2, 0], axs[2, 1], axs[2, 2]):
        ax.set_xlabel("x")
        ax.set_ylabel("t")

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
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))

    axs[0].loglog(sweep[:, 0], sweep[:, 2], marker="o", label=r"$L_{F/B/I}$")
    axs[0].loglog(sweep[:, 0], sweep[:, 3], marker="s", label=r"$L_J$")
    axs[0].loglog(sweep[:, 0], sweep[:, 4], marker="^", label=r"$L_{H^1}$")
    axs[0].set_title("(a) Loss Terms vs $w_J$")
    axs[0].legend(fontsize=8)

    best = int(np.argmin(sweep[:, 5]))
    axs[1].semilogx(sweep[:, 0], sweep[:, 5], marker="o", label="rollout objective")
    axs[1].scatter([sweep[best, 0]], [sweep[best, 5]], color="red", zorder=3, label="best")
    axs[1].set_title("(b) Rollout Objective")
    axs[1].legend(fontsize=8)

    axs[2].semilogx(sweep[:, 0], sweep[:, 1], marker="o", label=r"$\alpha(w_J)$")
    axs[2].set_title("(c) Chosen Alpha")
    axs[2].legend(fontsize=8)

    for ax in axs:
        ax.set_xlabel(r"$w_J$")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_lcurve(control_dir: Path, out_png: Path) -> None:
    path = control_dir / "alpha_lcurve.csv"
    if not path.exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.1, 0.5, "Missing alpha_lcurve.csv\n(run with --alpha-method lcurve)", fontsize=12)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return

    table = _load_csv_2d(path)
    alpha = table[:, 0]
    residual = table[:, 1]
    regularization = table[:, 2]

    summary = _load_csv_2d(control_dir / "summary.csv")
    alpha_star = float(summary[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(residual, regularization, "b.-", lw=1.4, label="L-curve")

    idx = int(np.argmin(np.abs(alpha - alpha_star)))
    ax.loglog(residual[idx], regularization[idx], "ro", ms=8, label=rf"$\alpha^* = {alpha_star:.3e}$")
    ax.set_xlabel("Residual norm")
    ax.set_ylabel("H1 regularization norm")
    ax.set_title("L-curve for Alpha Selection")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for the heat optimal-control example.")
    parser.add_argument("--control-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/optimal_control_results")
    parser.add_argument("--sweep-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/sweep_wj/sweep_wj_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control/figures")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    control_dir = Path(args.control_dir)
    sweep_dir = Path(args.sweep_dir)
    explicit_control_dir = "--control-dir" in sys.argv
    control_dir = _resolve_best_control_dir(control_dir, sweep_dir, explicit_control_dir)

    plot_overview(control_dir, outdir / "fig_heat_optimal_overview.png")
    plot_lcurve(control_dir, outdir / "fig_heat_alpha_lcurve.png")
    plot_wj_sweep(sweep_dir, outdir / "fig_heat_wj_sweep.png")
    print(f"Using control results from: {control_dir}")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
