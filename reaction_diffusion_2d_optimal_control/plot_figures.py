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


def _reshape_field(table: np.ndarray, value_cols: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    x = table[:, 0]
    y = table[:, 1]
    t = table[:, 2]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    t_unique = np.unique(t)
    shape = (len(y_unique), len(x_unique), len(t_unique))
    xx = x.reshape(shape)
    yy = y.reshape(shape)
    tt = t.reshape(shape)
    values = [table[:, col].reshape(shape) for col in value_cols]
    return xx, yy, tt, values


def _reshape_terminal(table: np.ndarray, value_cols: list[int]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    x = table[:, 0]
    y = table[:, 1]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    shape = (len(y_unique), len(x_unique))
    xx = x.reshape(shape)
    yy = y.reshape(shape)
    values = [table[:, col].reshape(shape) for col in value_cols]
    return xx, yy, values


def _nearest_time_indices(t_values: np.ndarray, start: float, end: float, n_snapshots: int) -> list[int]:
    requested = np.linspace(start, min(end, float(t_values[-1])), n_snapshots)
    return [int(np.argmin(np.abs(t_values - t))) for t in requested]


def _contour(
    fig: plt.Figure,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    cmap: str,
    levels: np.ndarray | int = 40,
) -> None:
    c = ax.contourf(x, y, z, levels=levels, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.02)


def plot_training_terminal(control_dir: Path, out_png: Path) -> None:
    loss = _load_csv_2d(control_dir / "loss_history.csv")
    summary = _load_csv_2d(control_dir / "summary.csv")
    terminal = _load_csv_2d(control_dir / "terminal_state.csv")
    rollout_terminal = _load_csv_2d(control_dir / "rollout_terminal_state.csv")

    xx_t, yy_t, terminal_vals = _reshape_terminal(terminal, [2, 3])
    target_t, pinn_t = terminal_vals
    xx_r, yy_r, rollout_vals = _reshape_terminal(rollout_terminal, [2, 3])
    target_r, rollout_t = rollout_vals
    terminal_abs = np.abs(pinn_t - target_t)
    rollout_abs = np.abs(rollout_t - target_r)

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    axs[0, 0].semilogy(loss[:, 0], loss[:, 1], label="total")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 5], label="terminal obj")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 6], label="H1 reg")
    axs[0, 0].set_title("(a) Training Loss")
    axs[0, 0].set_xlabel("epoch")
    axs[0, 0].legend(fontsize=7)

    lcurve_path = control_dir / "alpha_lcurve.csv"
    if lcurve_path.exists():
        lcurve = _load_csv_2d(lcurve_path)
        alpha_star = float(summary[0, 2])
        axs[0, 1].loglog(lcurve[:, 1], lcurve[:, 2], marker="o")
        idx = int(np.argmin(np.abs(lcurve[:, 0] - alpha_star)))
        axs[0, 1].scatter([lcurve[idx, 1]], [lcurve[idx, 2]], color="red", zorder=3)
        axs[0, 1].set_title(f"(b) L-curve, alpha={alpha_star:.1e}")
        axs[0, 1].set_xlabel("residual")
        axs[0, 1].set_ylabel("regularization")
    else:
        axs[0, 1].axis("off")
        axs[0, 1].text(0.1, 0.5, "Missing alpha_lcurve.csv")

    axs[0, 2].axis("off")
    axs[0, 2].set_title("(c) Summary")
    axs[0, 2].text(
        0.02,
        0.95,
        "\n".join(
            [
                f"c = {summary[0, 0]:.3g}",
                f"wJ = {summary[0, 1]:.3g}",
                f"alpha = {summary[0, 2]:.3e}",
                f"rel_u = {summary[0, 3]:.3e}",
                f"f L2 RMS = {summary[0, 4]:.3e}",
                f"terminal misfit PINN = {summary[0, 5]:.3e}",
                f"rollout objective = {summary[0, 6]:.3e}",
                f"H1 rollout = {summary[0, 7]:.3e}",
                f"final state rel = {summary[0, 9]:.3e}",
            ]
        ),
        transform=axs[0, 2].transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.4"},
    )

    _contour(fig, axs[1, 0], xx_t, yy_t, target_t, "(d) Target u(T)", "viridis")
    _contour(fig, axs[1, 1], xx_t, yy_t, terminal_abs, "(e) |PINN u(T) - target|", "magma")
    _contour(fig, axs[1, 2], xx_r, yy_r, rollout_abs, "(f) |Rollout u(T) - target|", "magma")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_state_snapshots(control_dir: Path, out_png: Path, n_snapshots: int, snapshot_end_time: float) -> None:
    field = _load_csv_2d(control_dir / "field_u.csv")
    xx, yy, tt, vals = _reshape_field(field, [3, 4, 5])
    u_true, u_pinn, abs_error = vals
    t_values = tt[0, 0, :]
    time_indices = _nearest_time_indices(t_values, float(t_values[0]), snapshot_end_time, n_snapshots)

    fig, axs = plt.subplots(len(time_indices), 3, figsize=(10, 2.7 * len(time_indices)), squeeze=False, constrained_layout=True)
    for col, title in enumerate(["Analytical u", "PINN u", "Absolute Error"]):
        axs[0, col].set_title(title)

    for row, idx in enumerate(time_indices):
        t = float(t_values[idx])
        u_min = float(min(np.min(u_true[:, :, idx]), np.min(u_pinn[:, :, idx])))
        u_max = float(max(np.max(u_true[:, :, idx]), np.max(u_pinn[:, :, idx])))
        if np.isclose(u_min, u_max):
            u_min -= 1e-12
            u_max += 1e-12
        u_levels = np.linspace(u_min, u_max, 41)
        err_max = max(float(np.max(abs_error[:, :, idx])), 1e-12)
        panels = [
            (u_true[:, :, idx], "viridis", u_levels),
            (u_pinn[:, :, idx], "viridis", u_levels),
            (abs_error[:, :, idx], "magma", np.linspace(0.0, err_max, 41)),
        ]
        for col, (z, cmap, levels) in enumerate(panels):
            c = axs[row, col].contourf(xx[:, :, idx], yy[:, :, idx], z, levels=levels, cmap=cmap)
            axs[row, col].set_aspect("equal", adjustable="box")
            axs[row, col].set_xlabel("x")
            axs[row, col].set_ylabel(f"t={t:.2f}\ny" if col == 0 else "")
            fig.colorbar(c, ax=axs[row, col], fraction=0.046, pad=0.02)

    fig.suptitle("Optimal-Control State Snapshots")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_control_snapshots(control_dir: Path, out_png: Path, n_snapshots: int, snapshot_end_time: float) -> None:
    field = _load_csv_2d(control_dir / "control_f.csv")
    xx, yy, tt, vals = _reshape_field(field, [4, 5])
    f_pred, abs_error = vals
    t_values = tt[0, 0, :]
    time_indices = _nearest_time_indices(t_values, float(t_values[0]), snapshot_end_time, n_snapshots)

    fig, axs = plt.subplots(len(time_indices), 2, figsize=(8, 2.7 * len(time_indices)), squeeze=False, constrained_layout=True)
    axs[0, 0].set_title("Predicted Control f")
    axs[0, 1].set_title("|f|")

    for row, idx in enumerate(time_indices):
        t = float(t_values[idx])
        signed_max = max(float(np.max(np.abs(f_pred[:, :, idx]))), 1e-12)
        abs_max = max(float(np.max(abs_error[:, :, idx])), 1e-12)
        panels = [
            (f_pred[:, :, idx], "coolwarm", np.linspace(-signed_max, signed_max, 41)),
            (abs_error[:, :, idx], "magma", np.linspace(0.0, abs_max, 41)),
        ]
        for col, (z, cmap, levels) in enumerate(panels):
            c = axs[row, col].contourf(xx[:, :, idx], yy[:, :, idx], z, levels=levels, cmap=cmap)
            axs[row, col].set_aspect("equal", adjustable="box")
            axs[row, col].set_xlabel("x")
            axs[row, col].set_ylabel(f"t={t:.2f}\ny" if col == 0 else "")
            fig.colorbar(c, ax=axs[row, col], fraction=0.046, pad=0.02)

    fig.suptitle("Optimal-Control Forcing Snapshots")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_heat_style_overview(control_dir: Path, out_png: Path, snapshot_time: float) -> None:
    field_u = _load_csv_2d(control_dir / "field_u.csv")
    field_f = _load_csv_2d(control_dir / "control_f.csv")
    loss = _load_csv_2d(control_dir / "loss_history.csv")
    summary = _load_csv_2d(control_dir / "summary.csv")
    terminal = _load_csv_2d(control_dir / "terminal_state.csv")
    rollout_terminal = _load_csv_2d(control_dir / "rollout_terminal_state.csv")

    xx_u, yy_u, tt_u, u_vals = _reshape_field(field_u, [3, 4, 5])
    u_true, u_pinn, u_err = u_vals
    xx_f, yy_f, tt_f, f_vals = _reshape_field(field_f, [4, 5])
    f_pred, f_abs = f_vals
    t_values = tt_u[0, 0, :]
    snap_idx = int(np.argmin(np.abs(t_values - snapshot_time)))
    snap_t = float(t_values[snap_idx])

    xx_t, yy_t, terminal_vals = _reshape_terminal(terminal, [2, 3])
    target_t, pinn_t = terminal_vals
    xx_r, yy_r, rollout_vals = _reshape_terminal(rollout_terminal, [2, 3])
    target_r, rollout_t = rollout_vals

    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_lcurve = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[0, 2])
    ax_target = fig.add_subplot(gs[1, 0])
    ax_pinn_terminal = fig.add_subplot(gs[1, 1])
    ax_rollout_terminal = fig.add_subplot(gs[1, 2])
    ax_control = fig.add_subplot(gs[2, 0])
    ax_state = fig.add_subplot(gs[2, 1])
    ax_state_err = fig.add_subplot(gs[2, 2])

    ax_loss.semilogy(loss[:, 0], loss[:, 2], label="PDE")
    ax_loss.semilogy(loss[:, 0], loss[:, 3], label="BC")
    ax_loss.semilogy(loss[:, 0], loss[:, 4], label="IC")
    ax_loss.semilogy(loss[:, 0], loss[:, 5], label="J")
    if loss.shape[1] >= 7:
        ax_loss.semilogy(loss[:, 0], loss[:, 6], label="H1")
    ax_loss.set_title("(a) Loss History")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend(fontsize=8, ncol=2)

    lcurve_path = control_dir / "alpha_lcurve.csv"
    if lcurve_path.exists():
        table = _load_csv_2d(lcurve_path)
        alpha = table[:, 0]
        residual = table[:, 1]
        regularization = table[:, 2]
        alpha_star = float(summary[0, 2])
        idx = int(np.argmin(np.abs(alpha - alpha_star)))
        ax_lcurve.loglog(residual, regularization, "b.-", lw=1.4, label="L-curve")
        ax_lcurve.loglog(residual[idx], regularization[idx], "ro", ms=8, label=rf"$\alpha^*={alpha_star:.2e}$")
        ax_lcurve.legend(fontsize=8)
    else:
        ax_lcurve.text(0.12, 0.5, "Missing alpha_lcurve.csv", fontsize=12)
    ax_lcurve.set_title("(b) L-curve")
    ax_lcurve.set_xlabel("Residual norm")
    ax_lcurve.set_ylabel("H1 norm")

    ax_summary.axis("off")
    ax_summary.set_title("(c) Summary")
    ax_summary.text(
        0.02,
        0.95,
        "\n".join(
            [
                f"c = {summary[0, 0]:.3g}",
                f"wJ = {summary[0, 1]:.3g}",
                f"alpha = {summary[0, 2]:.3e}",
                f"rel_u = {summary[0, 3]:.3e}",
                f"f L2 RMS = {summary[0, 4]:.3e}",
                f"terminal misfit PINN = {summary[0, 5]:.3e}",
                f"rollout objective = {summary[0, 6]:.3e}",
                f"H1 rollout = {summary[0, 7]:.3e}",
                f"final state rel = {summary[0, 9]:.3e}",
            ]
        ),
        transform=ax_summary.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.4"},
    )

    _contour(fig, ax_target, xx_t, yy_t, target_t, "(d) Target u(T)", "viridis")
    _contour(fig, ax_pinn_terminal, xx_t, yy_t, np.abs(pinn_t - target_t), "(e) |PINN u(T) - target|", "magma")
    _contour(fig, ax_rollout_terminal, xx_r, yy_r, np.abs(rollout_t - target_r), "(f) |Rollout u(T) - target|", "magma")

    signed_max = max(float(np.max(np.abs(f_pred[:, :, snap_idx]))), 1e-12)
    _contour(
        fig,
        ax_control,
        xx_f[:, :, snap_idx],
        yy_f[:, :, snap_idx],
        f_pred[:, :, snap_idx],
        f"(g) Control f, t={snap_t:.2f}",
        "coolwarm",
        np.linspace(-signed_max, signed_max, 41),
    )
    _contour(fig, ax_state, xx_u[:, :, snap_idx], yy_u[:, :, snap_idx], u_pinn[:, :, snap_idx], f"(h) PINN State, t={snap_t:.2f}", "viridis")
    _contour(fig, ax_state_err, xx_u[:, :, snap_idx], yy_u[:, :, snap_idx], u_err[:, :, snap_idx], f"(i) |u_PINN - u_a|, t={snap_t:.2f}", "magma")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_lcurve(control_dir: Path, out_png: Path) -> None:
    path = control_dir / "alpha_lcurve.csv"
    if not path.exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.1, 0.5, "Missing alpha_lcurve.csv", fontsize=12)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        return

    table = _load_csv_2d(path)
    summary = _load_csv_2d(control_dir / "summary.csv")
    alpha = table[:, 0]
    residual = table[:, 1]
    regularization = table[:, 2]
    alpha_star = float(summary[0, 2])
    idx = int(np.argmin(np.abs(alpha - alpha_star)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(residual, regularization, "b.-", lw=1.4, label="L-curve")
    ax.loglog(residual[idx], regularization[idx], "ro", ms=8, label=rf"$\alpha^* = {alpha_star:.3e}$")
    ax.set_xlabel("Residual norm")
    ax.set_ylabel("H1 regularization norm")
    ax.set_title("L-curve for Alpha Selection")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_control_surfaces(control_dir: Path, out_png: Path, snapshot_end_time: float) -> None:
    field = _load_csv_2d(control_dir / "control_f.csv")
    xx, yy, tt, vals = _reshape_field(field, [4])
    (f_pred,) = vals
    t_values = tt[0, 0, :]
    time_indices = _nearest_time_indices(t_values, float(t_values[0]), snapshot_end_time, 3)

    fig = plt.figure(figsize=(18, 5.8))
    for plot_idx, idx in enumerate(time_indices, start=1):
        ax = fig.add_subplot(1, 3, plot_idx, projection="3d")
        surf = ax.plot_surface(xx[:, :, idx], yy[:, :, idx], f_pred[:, :, idx], cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(f"({chr(96 + plot_idx)}) Control Surface, t={float(t_values[idx]):.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y,t)")
        ax.view_init(elev=28, azim=-130)
        fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for the 2D reaction-diffusion optimal-control example.")
    parser.add_argument("--control-dir", default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d_optimal_control/optimal_control_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d_optimal_control/figures")
    parser.add_argument("--n-snapshots", type=int, default=5)
    parser.add_argument("--snapshot-end-time", type=float, default=0.2)
    parser.add_argument("--snapshot-time", type=float, default=0.1)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    control_dir = Path(args.control_dir)
    outdir = Path(args.outdir)
    plot_heat_style_overview(control_dir, outdir / "fig_reaction_diffusion_2d_oc_optimal_overview.png", args.snapshot_time)
    plot_lcurve(control_dir, outdir / "fig_reaction_diffusion_2d_oc_alpha_lcurve.png")
    plot_control_surfaces(control_dir, outdir / "fig_reaction_diffusion_2d_oc_control_surfaces_3d.png", args.snapshot_end_time)
    plot_training_terminal(control_dir, outdir / "fig_reaction_diffusion_2d_oc_training_terminal.png")
    plot_state_snapshots(control_dir, outdir / "fig_reaction_diffusion_2d_oc_state_snapshots.png", args.n_snapshots, args.snapshot_end_time)
    plot_control_snapshots(control_dir, outdir / "fig_reaction_diffusion_2d_oc_control_snapshots.png", args.n_snapshots, args.snapshot_end_time)
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
