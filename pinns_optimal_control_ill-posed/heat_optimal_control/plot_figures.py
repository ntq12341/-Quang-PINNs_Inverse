from __future__ import annotations

import argparse
import math
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


def _best_rollout_control_dir(sweep_dir: Path, fallback: Path, objective_col: int = 5) -> Path:
    summary_path = sweep_dir / "summary.csv"
    run_paths_path = sweep_dir / "run_paths.csv"
    if not summary_path.exists() or not run_paths_path.exists():
        return fallback
    try:
        sweep = _load_csv_2d(summary_path)
        if sweep.shape[1] <= objective_col:
            return fallback
        best = int(np.argmin(sweep[:, objective_col]))
        lines = run_paths_path.read_text(encoding="utf-8").strip().splitlines()[1:]
        if best >= len(lines):
            return fallback
        _, run_path = lines[best].split(",", 1)
        candidate = Path(run_path.strip())
        return candidate if candidate.exists() else fallback
    except Exception:
        return fallback


def plot_overview(control_dir: Path, sweep_dir: Path, out_png: Path) -> None:
    field_u = _load_csv_2d(control_dir / "field_u.csv")
    field_f = _load_csv_2d(control_dir / "control_f.csv")
    loss = _load_csv_2d(control_dir / "loss_history.csv")
    rollout_best_dir = _best_rollout_control_dir(sweep_dir, control_dir)
    term = _load_csv_2d(rollout_best_dir / "terminal_state.csv")
    roll_term = _load_csv_2d(rollout_best_dir / "rollout_terminal_state.csv")
    no_reg_control_dir = control_dir.parent / "no_reg" / "sweep_wj" / "optimal_control_results"
    no_reg_sweep_dir = control_dir.parent / "no_reg" / "sweep_wj" / "sweep_wj_results"
    no_reg_rollout_best_dir = _best_rollout_control_dir(no_reg_sweep_dir, no_reg_control_dir, objective_col=3)
    sweep_summary = _load_csv_2d(sweep_dir / "summary.csv") if (sweep_dir / "summary.csv").exists() else None
    no_reg_term = None
    no_reg_roll_term = None
    if (no_reg_rollout_best_dir / "terminal_state.csv").exists() and (no_reg_rollout_best_dir / "rollout_terminal_state.csv").exists():
        no_reg_term = _load_csv_2d(no_reg_rollout_best_dir / "terminal_state.csv")
        no_reg_roll_term = _load_csv_2d(no_reg_rollout_best_dir / "rollout_terminal_state.csv")

    xx_u, tt_u, u_vals = _reshape_field(field_u, [2, 3, 4])
    u_true, u_pred, u_err = u_vals
    xx_f, tt_f, f_vals = _reshape_field(field_f, [2, 3, 4])
    f_true, f_pred, f_err = f_vals
    f_term_x = xx_f[-1]
    f_term_true = f_true[-1]
    f_term_pred = f_pred[-1]
    f_term_no_reg = None
    if (no_reg_control_dir / "control_f.csv").exists():
        field_f_no_reg = _load_csv_2d(no_reg_control_dir / "control_f.csv")
        xx_f_no_reg, _, f_vals_no_reg = _reshape_field(field_f_no_reg, [2, 3, 4])
        f_term_no_reg = f_vals_no_reg[1][-1]
        f_term_x = xx_f_no_reg[-1]

    # Recompute the rollout state field from the plotted control so panel (e)
    # is consistent with panel (b), which already shows the rollout terminal state.
    from core import HeatForwardSolver

    x_eval = xx_f[0]
    t_eval = tt_f[:, 0]
    solver = HeatForwardSolver(nx=len(x_eval), nt=len(t_eval) - 1, L=math.pi, T=1.0)
    forcing = np.zeros((solver.nt + 1, solver.nx), dtype=np.float64)
    for n in range(solver.nt + 1):
        forcing[n] = np.interp(solver.x, x_eval, f_pred[min(n, len(t_eval) - 1)])
    states_roll = solver.rollout(forcing=forcing)
    u_roll = states_roll
    u_roll_interp = np.empty_like(u_pred)
    for n, t_val in enumerate(t_eval):
        idx = int(np.argmin(np.abs(solver.t - t_val)))
        u_roll_interp[n] = np.interp(x_eval, solver.x, u_roll[idx])
    u_roll_err = np.abs(u_roll_interp - u_pred)

    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(5, 6, hspace=0.55, wspace=0.55, height_ratios=[1.25, 1.25, 0.27, 0.27, 0.27])

    ax_loss = fig.add_subplot(gs[0, :2])
    ax_term = fig.add_subplot(gs[0, 2:4])
    ax_term_no_reg = fig.add_subplot(gs[0, 4:])
    ax_ctrl = fig.add_subplot(gs[1, :2])
    ax_lcurve = fig.add_subplot(gs[1, 2:4])
    ax_wj = fig.add_subplot(gs[1, 4:])
    ax_u_true = fig.add_subplot(gs[2:5, :2])
    ax_u_pred = fig.add_subplot(gs[2:5, 2:4])
    ax_u_err = fig.add_subplot(gs[2:5, 4:])

    ax_loss.semilogy(loss[:, 0], loss[:, 2], label="PDE")
    ax_loss.semilogy(loss[:, 0], loss[:, 3], label="BC")
    ax_loss.semilogy(loss[:, 0], loss[:, 4], label="IC")
    ax_loss.semilogy(loss[:, 0], loss[:, 5], label="J")
    if loss.shape[1] >= 7:
        ax_loss.semilogy(loss[:, 0], loss[:, 6], label="H1")
    ax_loss.set_title("(a) Loss History")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend(fontsize=8, ncol=2)

    ax_term.plot(term[:, 0], term[:, 1], color="black", lw=2.2, label="target")
    ax_term.plot(
        term[:, 0],
        term[:, 2],
        color="tab:orange",
        lw=1.8,
        ls="-.",
        label="PINN",
    )
    ax_term.plot(
        roll_term[:, 0],
        roll_term[:, 2],
        color="tab:blue",
        lw=2.0,
        ls="--",
        label="rollout",
    )
    ax_term.set_title("(b) Terminal State + Rollout")
    ax_term.set_xlabel("x")
    ax_term.legend(fontsize=8)

    if no_reg_term is not None and no_reg_roll_term is not None:
        ax_term_no_reg.plot(no_reg_term[:, 0], no_reg_term[:, 1], color="black", lw=2.2, label="target")
        ax_term_no_reg.plot(
            no_reg_term[:, 0],
            no_reg_term[:, 2],
            color="tab:orange",
            lw=1.8,
            ls="-.",
            label="PINN",
        )
        ax_term_no_reg.plot(
            no_reg_roll_term[:, 0],
            no_reg_roll_term[:, 2],
            color="tab:blue",
            lw=2.0,
            ls="--",
            label="rollout",
        )
        ax_term_no_reg.legend(fontsize=8)
    else:
        ax_term_no_reg.text(0.08, 0.5, "Missing no_reg terminal outputs", fontsize=12)
    ax_term_no_reg.set_title("(c) No-Reg Terminal State + Rollout")
    ax_term_no_reg.set_xlabel("x")

    ax_ctrl.plot(f_term_x, f_term_pred, color="tab:red", lw=1.8, ls="--", label="with Tikhonov")
    if f_term_no_reg is not None:
        ax_ctrl.plot(f_term_x, f_term_no_reg, color="tab:blue", lw=1.6, ls="-.", label="no Tikhonov")
    else:
        ax_ctrl.text(0.04, 0.90, "missing no_reg output", transform=ax_ctrl.transAxes, fontsize=9)
    ax_ctrl.set_title("(d) Control Comparison (With vs. Without Tikhonov)")
    ax_ctrl.set_xlabel("x")
    ax_ctrl.legend(fontsize=8)

    lcurve_path = control_dir / "alpha_lcurve.csv"
    if lcurve_path.exists():
        table = _load_csv_2d(lcurve_path)
        alpha = table[:, 0]
        residual = table[:, 1]
        regularization = table[:, 2]
        summary = _load_csv_2d(control_dir / "summary.csv")
        alpha_star = float(summary[0, 1]) if summary.shape[1] > 1 else alpha[np.argmin(residual)]
        idx = int(np.argmin(np.abs(alpha - alpha_star)))
        ax_lcurve.loglog(residual, regularization, "b.-", lw=1.4, label="L-curve")
        ax_lcurve.loglog(residual[idx], regularization[idx], "ro", ms=8, label=rf"$\alpha^*={alpha_star:.2e}$")
        ax_lcurve.legend(fontsize=8)
    else:
        ax_lcurve.text(0.12, 0.5, "Missing alpha_lcurve.csv", fontsize=12)
    ax_lcurve.set_title("(e) L-curve")
    ax_lcurve.set_xlabel("Residual norm")
    ax_lcurve.set_ylabel("H1 norm")

    if sweep_summary is not None and sweep_summary.shape[1] >= 6:
        best = int(np.argmin(sweep_summary[:, 5]))
        ax_wj.semilogx(sweep_summary[:, 0], sweep_summary[:, 5], marker="o", label="rollout objective")
        ax_wj.scatter([sweep_summary[best, 0]], [sweep_summary[best, 5]], color="red", zorder=3, label="best")
        ax_wj.legend(fontsize=8)
    else:
        ax_wj.text(0.10, 0.5, "Missing sweep_wj summary.csv", fontsize=12)
    ax_wj.set_title(r"(f) Rollout Objective vs $w_J$")
    ax_wj.set_xlabel(r"$w_J$")

    c = ax_u_true.contourf(xx_u, tt_u, u_roll_interp, levels=40, cmap="viridis")
    ax_u_true.set_title(r"(g) Rollout State $u(f)$")
    ax_u_true.set_xlabel("x")
    ax_u_true.set_ylabel("t")
    fig.colorbar(c, ax=ax_u_true)

    c = ax_u_pred.contourf(xx_u, tt_u, u_pred, levels=40, cmap="viridis")
    ax_u_pred.set_title("(h) PINN State")
    ax_u_pred.set_xlabel("x")
    ax_u_pred.set_ylabel("t")
    fig.colorbar(c, ax=ax_u_pred)

    c = ax_u_err.contourf(xx_u, tt_u, u_roll_err, levels=40, cmap="magma")
    ax_u_err.set_title("(i) Rollout vs. PINN Error")
    ax_u_err.set_xlabel("x")
    ax_u_err.set_ylabel("t")
    fig.colorbar(c, ax=ax_u_err)

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


def plot_control_surfaces(control_dir: Path, out_png: Path) -> None:
    field_f = _load_csv_2d(control_dir / "control_f.csv")
    xx_f, tt_f, f_vals = _reshape_field(field_f, [2, 3, 4])
    _, f_pred, _ = f_vals

    no_reg_control_dir = control_dir.parent / "no_reg" / "sweep_wj" / "optimal_control_results"
    has_no_reg = (no_reg_control_dir / "control_f.csv").exists()
    if has_no_reg:
        field_f_no_reg = _load_csv_2d(no_reg_control_dir / "control_f.csv")
        xx_f_no_reg, tt_f_no_reg, f_vals_no_reg = _reshape_field(field_f_no_reg, [2, 3, 4])
        _, f_pred_no_reg, _ = f_vals_no_reg

    fig = plt.figure(figsize=(15, 6.2))
    ax_reg = fig.add_subplot(1, 2, 1, projection="3d")
    surf_reg = ax_reg.plot_surface(xx_f, tt_f, f_pred, cmap="viridis", linewidth=0, antialiased=True)
    ax_reg.set_title("(a) Control Surface with Tikhonov")
    ax_reg.set_xlabel("x")
    ax_reg.set_ylabel("t")
    ax_reg.set_zlabel("f(x,t)")
    ax_reg.view_init(elev=28, azim=-130)
    fig.colorbar(surf_reg, ax=ax_reg, shrink=0.7, pad=0.08)

    ax_no_reg = fig.add_subplot(1, 2, 2, projection="3d")
    if has_no_reg:
        surf_no_reg = ax_no_reg.plot_surface(
            xx_f_no_reg,
            tt_f_no_reg,
            f_pred_no_reg,
            cmap="plasma",
            linewidth=0,
            antialiased=True,
        )
        fig.colorbar(surf_no_reg, ax=ax_no_reg, shrink=0.7, pad=0.08)
    else:
        ax_no_reg.text2D(0.18, 0.5, "Missing no_reg control_f.csv", transform=ax_no_reg.transAxes)
    ax_no_reg.set_title("(b) Control Surface without Tikhonov")
    ax_no_reg.set_xlabel("x")
    ax_no_reg.set_ylabel("t")
    ax_no_reg.set_zlabel("f(x,t)")
    ax_no_reg.view_init(elev=28, azim=-130)

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

    plot_overview(control_dir, sweep_dir, outdir / "fig_heat_optimal_overview.png")
    plot_control_surfaces(control_dir, outdir / "fig_heat_control_surfaces_3d.png")
    plot_lcurve(control_dir, outdir / "fig_heat_alpha_lcurve.png")
    plot_wj_sweep(sweep_dir, outdir / "fig_heat_wj_sweep.png")
    print(f"Using control results from: {control_dir}")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
