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


def _nearest_time_indices(t_values: np.ndarray, requested_times: np.ndarray) -> list[int]:
    return [int(np.argmin(np.abs(t_values - t))) for t in requested_times]


def _set_xy_labels(ax: plt.Axes) -> None:
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_contour(
    fig: plt.Figure,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    levels: np.ndarray | int = 40,
) -> None:
    c = ax.contourf(x, y, values, levels=levels, cmap=cmap)
    ax.set_title(title)
    _set_xy_labels(ax)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(c, ax=ax)


def _error_metrics_by_time(u_true: np.ndarray, u_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signed_err = u_pred - u_true
    err_flat = signed_err.reshape((-1, signed_err.shape[-1]))
    true_flat = u_true.reshape((-1, u_true.shape[-1]))
    signal_scale = max(float(np.max(np.linalg.norm(true_flat, axis=0))), np.finfo(float).eps)
    scaled_l2_t = np.linalg.norm(err_flat, axis=0) / signal_scale
    linf_t = np.max(np.abs(err_flat), axis=0)
    mae_t = np.mean(np.abs(err_flat), axis=0)
    return scaled_l2_t, linf_t, mae_t


def plot_forward(forward_dir: Path, out_png: Path, snapshot_time: float) -> None:
    field = _load_csv_2d(forward_dir / "field_u.csv")
    loss = _load_csv_2d(forward_dir / "loss_history.csv")
    test = _load_csv_2d(forward_dir / "test_error_history.csv")
    summary = _load_csv_2d(forward_dir / "summary.csv")

    xx, yy, _tt, u_true, u_pred, err = _reshape_field(field)
    signed_err = u_pred - u_true
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

    u_min = float(min(np.min(u_true[:, :, snapshot_idx]), np.min(u_pred[:, :, snapshot_idx])))
    u_max = float(max(np.max(u_true[:, :, snapshot_idx]), np.max(u_pred[:, :, snapshot_idx])))
    u_levels = np.linspace(u_min, u_max, 41)
    signed_max = float(np.max(np.abs(signed_err[:, :, snapshot_idx])))
    signed_levels = np.linspace(-signed_max, signed_max, 41)
    panels = [
        (axs[1, 0], u_true[:, :, snapshot_idx], f"(d) Analytical $u_a$, t={snapshot_t:.2f}", "viridis", u_levels),
        (axs[1, 1], u_pred[:, :, snapshot_idx], f"(e) PINN $u$, t={snapshot_t:.2f}", "viridis", u_levels),
        (axs[1, 2], signed_err[:, :, snapshot_idx], f"(f) Signed Error, t={snapshot_t:.2f}", "coolwarm", signed_levels),
    ]
    for ax, values, title, cmap, levels in panels:
        _plot_contour(fig, ax, xx[:, :, snapshot_idx], yy[:, :, snapshot_idx], values, title, cmap, levels)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_forward_diagnostics(forward_dir: Path, out_png: Path) -> None:
    field = _load_csv_2d(forward_dir / "field_u.csv")
    loss = _load_csv_2d(forward_dir / "loss_history.csv")
    test = _load_csv_2d(forward_dir / "test_error_history.csv")
    summary = _load_csv_2d(forward_dir / "summary.csv")

    _xx, _yy, tt, u_true, u_pred, _err = _reshape_field(field)
    t_values = tt[0, 0, :]
    scaled_l2_t, linf_t, mae_t = _error_metrics_by_time(u_true, u_pred)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
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

    axs[1, 0].plot(t_values, scaled_l2_t, label="scaled L2")
    axs[1, 0].plot(t_values, linf_t, label="Linf")
    axs[1, 0].plot(t_values, mae_t, label="MAE")
    axs[1, 0].set_title("(c) Error Metrics over Time")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].legend(fontsize=8)

    rel_l2 = float(summary[0, 1]) if summary.shape[1] > 1 else float("nan")
    grid_l2 = float(summary[0, 2]) if summary.shape[1] > 2 else float("nan")
    axs[1, 1].axis("off")
    axs[1, 1].text(
        0.05,
        0.75,
        "\n".join(
            [
                f"final relative L2: {rel_l2:.3e}",
                f"grid L2 error: {grid_l2:.3e}",
                f"max scaled L2(t): {np.max(scaled_l2_t):.3e}",
                f"max Linf(t): {np.max(linf_t):.3e}",
                f"mean MAE(t): {np.mean(mae_t):.3e}",
            ]
        ),
        transform=axs[1, 1].transAxes,
        fontsize=11,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.5"},
    )
    axs[1, 1].set_title("(d) Summary")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_grid(forward_dir: Path, out_png: Path, n_snapshots: int, snapshot_end_time: float | None) -> None:
    field = _load_csv_2d(forward_dir / "field_u.csv")
    xx, yy, tt, u_true, u_pred, err = _reshape_field(field)
    t_values = tt[0, 0, :]
    t_start = float(t_values[0])
    t_end = float(t_values[-1])
    if snapshot_end_time is not None:
        t_end = min(float(snapshot_end_time), t_end)
    requested_times = np.linspace(t_start, t_end, n_snapshots)
    time_indices = _nearest_time_indices(t_values, requested_times)

    fig, axs = plt.subplots(
        len(time_indices),
        3,
        figsize=(10, 2.7 * len(time_indices)),
        squeeze=False,
        constrained_layout=True,
    )
    col_titles = ["Analytical", "PINN", "Absolute Error"]
    for col, title in enumerate(col_titles):
        axs[0, col].set_title(title)

    for row, idx in enumerate(time_indices):
        t = float(t_values[idx])
        u_min = float(min(np.min(u_true[:, :, idx]), np.min(u_pred[:, :, idx])))
        u_max = float(max(np.max(u_true[:, :, idx]), np.max(u_pred[:, :, idx])))
        if np.isclose(u_min, u_max):
            u_min -= 1e-12
            u_max += 1e-12
        u_levels = np.linspace(u_min, u_max, 41)

        abs_max = float(np.max(err[:, :, idx]))
        abs_max = abs_max if abs_max > 0 else 1e-12
        abs_levels = np.linspace(0.0, abs_max, 41)

        panels = [
            (u_true[:, :, idx], "viridis", u_levels),
            (u_pred[:, :, idx], "viridis", u_levels),
            (err[:, :, idx], "magma", abs_levels),
        ]
        for col, (values, cmap, levels) in enumerate(panels):
            c = axs[row, col].contourf(xx[:, :, idx], yy[:, :, idx], values, levels=levels, cmap=cmap)
            axs[row, col].set_aspect("equal", adjustable="box")
            axs[row, col].set_xlabel("x")
            if col == 0:
                axs[row, col].set_ylabel(f"t={t:.2f}\ny")
            else:
                axs[row, col].set_ylabel("")
            fig.colorbar(c, ax=axs[row, col], fraction=0.046, pad=0.02)

    fig.suptitle("Forward Solution Snapshots across Time")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_global_error_summary(forward_dir: Path, out_png: Path) -> None:
    field = _load_csv_2d(forward_dir / "field_u.csv")
    xx, yy, tt, u_true, u_pred, err = _reshape_field(field)
    signed_err = u_pred - u_true
    t_values = tt[0, 0, :]
    x_values = xx[0, :, 0]
    y_values = yy[:, 0, 0]

    rms_t = np.sqrt(np.mean(signed_err**2, axis=2))
    max_abs_t = np.max(err, axis=2)
    mean_y_abs = np.mean(err, axis=0)
    mean_x_abs = np.mean(err, axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    _plot_contour(fig, axs[0, 0], xx[:, :, 0], yy[:, :, 0], rms_t, "(a) RMS Error over Time", "magma")
    _plot_contour(fig, axs[0, 1], xx[:, :, 0], yy[:, :, 0], max_abs_t, "(b) Max Absolute Error over Time", "magma")

    tx, xt = np.meshgrid(t_values, x_values)
    c = axs[1, 0].contourf(tx, xt, mean_y_abs, levels=40, cmap="magma")
    axs[1, 0].set_title("(c) Mean-y Absolute Error, x-t")
    axs[1, 0].set_xlabel("t")
    axs[1, 0].set_ylabel("x")
    fig.colorbar(c, ax=axs[1, 0])

    ty, yt = np.meshgrid(t_values, y_values)
    c = axs[1, 1].contourf(ty, yt, mean_x_abs, levels=40, cmap="magma")
    axs[1, 1].set_title("(d) Mean-x Absolute Error, y-t")
    axs[1, 1].set_xlabel("t")
    axs[1, 1].set_ylabel("y")
    fig.colorbar(c, ax=axs[1, 1])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for the 2D reaction-diffusion forward example.")
    parser.add_argument("--forward-dir", default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d/forward/forward_results")
    parser.add_argument("--outdir", default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d/figures")
    parser.add_argument("--snapshot-time", type=float, default=0.25)
    parser.add_argument("--n-snapshots", type=int, default=5)
    parser.add_argument("--snapshot-end-time", type=float, default=0.2)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    forward_dir = Path(args.forward_dir)
    plot_forward(forward_dir, outdir / "fig_forward_reaction_diffusion_2d.png", args.snapshot_time)
    plot_forward_diagnostics(forward_dir, outdir / "fig_forward_reaction_diffusion_2d_diagnostics.png")
    plot_snapshot_grid(
        forward_dir,
        outdir / "fig_forward_reaction_diffusion_2d_snapshots.png",
        args.n_snapshots,
        args.snapshot_end_time,
    )
    plot_global_error_summary(forward_dir, outdir / "fig_forward_reaction_diffusion_2d_global_error.png")
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
