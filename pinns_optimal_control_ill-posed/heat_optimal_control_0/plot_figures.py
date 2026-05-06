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


def _best_run_dir(sweep_dir: Path, fallback: Path, objective_col: int) -> Path:
    summary_path = sweep_dir / "summary.csv"
    run_paths_path = sweep_dir / "run_paths.csv"
    if not summary_path.exists() or not run_paths_path.exists():
        return fallback
    try:
        summary = _load_csv_2d(summary_path)
        idx = int(np.argmin(summary[:, objective_col]))
        rows = run_paths_path.read_text(encoding="utf-8").strip().splitlines()[1:]
        if idx >= len(rows):
            return fallback
        _, run_path = rows[idx].split(",", 1)
        candidate = Path(run_path.strip())
        return candidate if candidate.exists() else fallback
    except Exception:
        return fallback


def _legend_if_any(ax: plt.Axes, **kwargs: object) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def plot_overview(control_dir: Path, no_reg_dir: Path, sweep_dir: Path, out_png: Path) -> None:
    field_u_reg = _load_csv_2d(control_dir / "field_u.csv")
    field_f_reg = _load_csv_2d(control_dir / "control_f.csv")
    loss_reg = _load_csv_2d(control_dir / "loss_history.csv")
    term_reg = _load_csv_2d(control_dir / "terminal_state.csv")
    roll_term_reg = _load_csv_2d(control_dir / "rollout_terminal_state.csv")
    summary_reg = _load_csv_2d(control_dir / "summary.csv")

    field_u_noreg = _load_csv_2d(no_reg_dir / "field_u.csv")
    field_f_noreg = _load_csv_2d(no_reg_dir / "control_f.csv")
    loss_noreg = _load_csv_2d(no_reg_dir / "loss_history.csv")
    term_noreg = _load_csv_2d(no_reg_dir / "terminal_state.csv")
    roll_term_noreg = _load_csv_2d(no_reg_dir / "rollout_terminal_state.csv")
    summary_noreg = _load_csv_2d(no_reg_dir / "summary.csv")

    xx_u, tt_u, u_reg_vals = _reshape_field(field_u_reg, [2, 3, 4])
    u_exact, u_reg, u_reg_err = u_reg_vals
    _, _, f_reg_vals = _reshape_field(field_f_reg, [2, 3, 4])
    f_exact, f_reg, _ = f_reg_vals
    _, _, u_noreg_vals = _reshape_field(field_u_noreg, [2, 3, 4])
    _, u_noreg, u_noreg_err = u_noreg_vals
    _, _, f_noreg_vals = _reshape_field(field_f_noreg, [2, 3, 4])
    _, f_noreg, _ = f_noreg_vals

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_term_reg = fig.add_subplot(gs[0, 1])
    ax_term_noreg = fig.add_subplot(gs[0, 2])
    ax_ctrl = fig.add_subplot(gs[1, 0])
    ax_lcurve = fig.add_subplot(gs[1, 1])
    ax_wj = fig.add_subplot(gs[1, 2])
    ax_u_exact = fig.add_subplot(gs[2, 0])
    ax_u_reg_err = fig.add_subplot(gs[2, 1])
    ax_u_noreg_err = fig.add_subplot(gs[2, 2])

    ax_loss.semilogy(loss_reg[:, 0], loss_reg[:, 2], label="reg PDE", color="tab:blue")
    ax_loss.semilogy(loss_reg[:, 0], loss_reg[:, 3], label="reg BC", color="tab:orange")
    ax_loss.semilogy(loss_reg[:, 0], loss_reg[:, 4], label="reg IC", color="tab:green")
    ax_loss.semilogy(loss_reg[:, 0], loss_reg[:, 5], label="reg J", color="tab:red")
    if loss_reg.shape[1] >= 7:
        ax_loss.semilogy(loss_reg[:, 0], loss_reg[:, 6], label="reg H1", color="tab:purple")
    ax_loss.semilogy(loss_noreg[:, 0], loss_noreg[:, 2], "--", label="no-reg PDE", color="tab:blue", alpha=0.6)
    ax_loss.semilogy(loss_noreg[:, 0], loss_noreg[:, 5], "--", label="no-reg J", color="tab:red", alpha=0.6)
    ax_loss.set_title("(a) Loss History")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend(fontsize=8, ncol=2)

    ax_term_reg.plot(term_reg[:, 0], term_reg[:, 1], color="black", lw=2.2, label="exact target")
    ax_term_reg.plot(term_reg[:, 0], term_reg[:, 2], color="tab:orange", lw=1.7, ls="-.", label="reg PINN")
    ax_term_reg.plot(roll_term_reg[:, 0], roll_term_reg[:, 2], color="tab:blue", lw=1.9, ls="--", label="reg rollout")
    ax_term_reg.set_title("(b) Terminal State: Reg vs Exact")
    ax_term_reg.set_xlabel("x")
    ax_term_reg.legend(fontsize=8)

    ax_term_noreg.plot(term_noreg[:, 0], term_noreg[:, 1], color="black", lw=2.2, label="exact target")
    ax_term_noreg.plot(term_noreg[:, 0], term_noreg[:, 2], color="tab:orange", lw=1.7, ls="-.", label="no-reg PINN")
    ax_term_noreg.plot(roll_term_noreg[:, 0], roll_term_noreg[:, 2], color="tab:blue", lw=1.9, ls="--", label="no-reg rollout")
    ax_term_noreg.set_title("(c) Terminal State: No-Reg vs Exact")
    ax_term_noreg.set_xlabel("x")
    ax_term_noreg.legend(fontsize=8)

    x_line = xx_u[-1]
    ax_ctrl.plot(x_line, f_exact[-1], color="black", lw=2.2, label="exact")
    ax_ctrl.plot(x_line, f_reg[-1], color="tab:red", lw=1.8, ls="--", label="reg")
    ax_ctrl.plot(x_line, f_noreg[-1], color="tab:blue", lw=1.8, ls="-.", label="no-reg")
    ax_ctrl.set_title("(d) Control at t = 1")
    ax_ctrl.set_xlabel("x")
    ax_ctrl.legend(fontsize=8)

    lcurve_path = control_dir / "alpha_lcurve.csv"
    if lcurve_path.exists():
        alpha_curve = _load_csv_2d(lcurve_path)
        alpha_vals = alpha_curve[:, 0]
        residual = alpha_curve[:, 1]
        regularization = alpha_curve[:, 2]
        alpha_star = float(summary_reg[0, 1])
        idx = int(np.argmin(np.abs(alpha_vals - alpha_star)))
        ax_lcurve.loglog(residual, regularization, "b.-", lw=1.4, label="L-curve")
        ax_lcurve.loglog(residual[idx], regularization[idx], "ro", ms=8, label=rf"$\alpha^*={alpha_star:.2e}$")
        ax_lcurve.legend(fontsize=8)
    else:
        ax_lcurve.text(0.12, 0.50, "Missing alpha_lcurve.csv", fontsize=11)
        ax_lcurve.axis("off")
    ax_lcurve.set_title("(e) Reg L-Curve")
    ax_lcurve.set_xlabel("Residual norm")
    ax_lcurve.set_ylabel("H1 norm")

    sweep_summary_path = sweep_dir / "summary.csv"
    no_reg_sweep_path = no_reg_dir.parent / "sweep_wj" / "sweep_wj_results" / "summary.csv"
    if sweep_summary_path.exists():
        reg_sweep = _load_csv_2d(sweep_summary_path)
        best_reg = int(np.argmin(reg_sweep[:, 7]))
        ax_wj.semilogx(reg_sweep[:, 0], reg_sweep[:, 7], marker="o", color="tab:red", label="reg rollout J")
        ax_wj.scatter([reg_sweep[best_reg, 0]], [reg_sweep[best_reg, 7]], color="tab:red", zorder=3)
    if no_reg_sweep_path.exists():
        no_reg_sweep = _load_csv_2d(no_reg_sweep_path)
        best_noreg = int(np.argmin(no_reg_sweep[:, 5]))
        ax_wj.semilogx(no_reg_sweep[:, 0], no_reg_sweep[:, 5], marker="s", color="tab:blue", label="no-reg rollout J")
        ax_wj.scatter([no_reg_sweep[best_noreg, 0]], [no_reg_sweep[best_noreg, 5]], color="tab:blue", zorder=3)
    ax_wj.set_title("(f) Rollout Objective vs $w_J$")
    ax_wj.set_xlabel("$w_J$")
    _legend_if_any(ax_wj, fontsize=8)

    c0 = ax_u_exact.contourf(xx_u, tt_u, u_exact, levels=40, cmap="viridis")
    ax_u_exact.set_title("(g) Exact State $u^*$")
    ax_u_exact.set_xlabel("x")
    ax_u_exact.set_ylabel("t")
    fig.colorbar(c0, ax=ax_u_exact)

    c1 = ax_u_reg_err.contourf(xx_u, tt_u, u_reg_err, levels=40, cmap="magma")
    ax_u_reg_err.set_title(r"(h) $|u_{\mathrm{reg}} - u^*|$")
    ax_u_reg_err.set_xlabel("x")
    ax_u_reg_err.set_ylabel("t")
    fig.colorbar(c1, ax=ax_u_reg_err)

    c2 = ax_u_noreg_err.contourf(xx_u, tt_u, u_noreg_err, levels=40, cmap="magma")
    ax_u_noreg_err.set_title(r"(i) $|u_{\mathrm{no-reg}} - u^*|$")
    ax_u_noreg_err.set_xlabel("x")
    ax_u_noreg_err.set_ylabel("t")
    fig.colorbar(c2, ax=ax_u_noreg_err)

    reg_text = (
        f"reg: rel_u={summary_reg[0, 2]:.2e}, rel_f={summary_reg[0, 3]:.2e}, "
        f"J={summary_reg[0, 5]:.2e}, H1={summary_reg[0, 6]:.2e}"
    )
    no_reg_text = (
        f"no-reg: rel_u={summary_noreg[0, 1]:.2e}, rel_f={summary_noreg[0, 2]:.2e}, "
        f"J={summary_noreg[0, 4]:.2e}"
    )
    fig.text(0.08, 0.015, reg_text, fontsize=10)
    fig.text(0.56, 0.015, no_reg_text, fontsize=10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_control_surfaces(control_dir: Path, no_reg_dir: Path, out_png: Path) -> None:
    field_f_reg = _load_csv_2d(control_dir / "control_f.csv")
    field_f_noreg = _load_csv_2d(no_reg_dir / "control_f.csv")
    xx_f, tt_f, f_reg_vals = _reshape_field(field_f_reg, [2, 3, 4])
    f_exact, f_reg, _ = f_reg_vals
    xx_n, tt_n, f_noreg_vals = _reshape_field(field_f_noreg, [2, 3, 4])
    _, f_noreg, _ = f_noreg_vals

    fig = plt.figure(figsize=(18, 6.2))
    ax_exact = fig.add_subplot(1, 3, 1, projection="3d")
    ax_reg = fig.add_subplot(1, 3, 2, projection="3d")
    ax_noreg = fig.add_subplot(1, 3, 3, projection="3d")

    surf0 = ax_exact.plot_surface(xx_f, tt_f, f_exact, cmap="cividis", linewidth=0, antialiased=True)
    surf1 = ax_reg.plot_surface(xx_f, tt_f, f_reg, cmap="viridis", linewidth=0, antialiased=True)
    surf2 = ax_noreg.plot_surface(xx_n, tt_n, f_noreg, cmap="plasma", linewidth=0, antialiased=True)

    for ax, title in [
        (ax_exact, "(a) Exact Control"),
        (ax_reg, "(b) Regularized PINN"),
        (ax_noreg, "(c) No-Reg PINN"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("f(x,t)")
        ax.view_init(elev=28, azim=-130)

    fig.colorbar(surf0, ax=ax_exact, shrink=0.72, pad=0.08)
    fig.colorbar(surf1, ax=ax_reg, shrink=0.72, pad=0.08)
    fig.colorbar(surf2, ax=ax_noreg, shrink=0.72, pad=0.08)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_wj_sweep(reg_sweep_dir: Path, no_reg_sweep_dir: Path, out_png: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

    reg_summary = _load_csv_2d(reg_sweep_dir / "summary.csv") if (reg_sweep_dir / "summary.csv").exists() else None
    no_reg_summary = _load_csv_2d(no_reg_sweep_dir / "summary.csv") if (no_reg_sweep_dir / "summary.csv").exists() else None

    if reg_summary is not None:
        axs[0].loglog(reg_summary[:, 0], reg_summary[:, 2], marker="o", label=r"reg $L_{F/B/I}$")
        axs[0].loglog(reg_summary[:, 0], reg_summary[:, 3], marker="s", label=r"reg $L_J$")
        axs[0].loglog(reg_summary[:, 0], reg_summary[:, 4], marker="^", label=r"reg $L_{H^1}$")
    if no_reg_summary is not None:
        axs[0].loglog(no_reg_summary[:, 0], no_reg_summary[:, 1], "--o", label=r"no-reg $L_{F/B/I}$")
        axs[0].loglog(no_reg_summary[:, 0], no_reg_summary[:, 2], "--s", label=r"no-reg $L_J$")
    axs[0].set_title("(a) Loss Terms vs $w_J$")
    axs[0].set_xlabel("$w_J$")
    _legend_if_any(axs[0], fontsize=8)

    if reg_summary is not None:
        best_reg = int(np.argmin(reg_summary[:, 7]))
        axs[1].semilogx(reg_summary[:, 0], reg_summary[:, 7], marker="o", color="tab:red", label="reg rollout J")
        axs[1].scatter([reg_summary[best_reg, 0]], [reg_summary[best_reg, 7]], color="tab:red", zorder=3)
    if no_reg_summary is not None:
        best_noreg = int(np.argmin(no_reg_summary[:, 5]))
        axs[1].semilogx(no_reg_summary[:, 0], no_reg_summary[:, 5], marker="s", color="tab:blue", label="no-reg rollout J")
        axs[1].scatter([no_reg_summary[best_noreg, 0]], [no_reg_summary[best_noreg, 5]], color="tab:blue", zorder=3)
    axs[1].set_title("(b) Rollout Objective")
    axs[1].set_xlabel("$w_J$")
    _legend_if_any(axs[1], fontsize=8)

    if reg_summary is not None:
        axs[2].semilogx(reg_summary[:, 0], reg_summary[:, 1], marker="o", color="tab:purple", label=r"$\alpha(w_J)$")
        axs[2].legend(fontsize=8)
    else:
        axs[2].text(0.15, 0.50, "Missing reg sweep summary", fontsize=11)
        axs[2].axis("off")
    axs[2].set_title("(c) Selected Alpha")
    axs[2].set_xlabel("$w_J$")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for heat optimal control example 5.1.2.")
    parser.add_argument("--control-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0/optimal_control_results")
    parser.add_argument("--no-reg-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0/no_reg/optimal_control_results")
    parser.add_argument("--sweep-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0/sweep_wj/sweep_wj_results")
    parser.add_argument("--no-reg-sweep-dir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0/no_reg/sweep_wj/sweep_wj_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0/figures")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    control_dir = _best_run_dir(Path(args.sweep_dir), Path(args.control_dir), objective_col=7)
    no_reg_dir = _best_run_dir(Path(args.no_reg_sweep_dir), Path(args.no_reg_dir), objective_col=5)
    outdir = Path(args.outdir)

    plot_overview(control_dir, no_reg_dir, Path(args.sweep_dir), outdir / "fig_heat0_overview.png")
    plot_control_surfaces(control_dir, no_reg_dir, outdir / "fig_heat0_control_surfaces_3d.png")
    plot_wj_sweep(Path(args.sweep_dir), Path(args.no_reg_sweep_dir), outdir / "fig_heat0_wj_sweep.png")
    print(f"Using regularized results from: {control_dir}")
    print(f"Using no-reg results from: {no_reg_dir}")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
