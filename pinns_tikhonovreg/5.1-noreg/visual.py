from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_surface_comparison(
    x,
    y,
    u_exact_grid,
    u_ann_grid,
    noise_label="3%",
    save_path=None,
):
    """Plot 3D exact / ANN / error surfaces in one figure."""
    X, Y = np.meshgrid(x, y)
    err_grid = u_ann_grid - u_exact_grid

    fig = plt.figure(figsize=(18, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.26)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    s1 = ax1.plot_surface(X, Y, u_exact_grid, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title("Exact Solution")
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel("u")
    fig.colorbar(s1, ax=ax1, shrink=0.60, pad=0.06)

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    s2 = ax2.plot_surface(X, Y, u_ann_grid, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_title(f"ANN Solution, noise {noise_label}")
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel("u")
    fig.colorbar(s2, ax=ax2, shrink=0.60, pad=0.06)

    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    s3 = ax3.plot_surface(X, Y, err_grid, cmap="viridis", linewidth=0, antialiased=True)
    ax3.set_title(f"Error, noise {noise_label}")
    ax3.set_xlabel(r"$x_1$")
    ax3.set_ylabel(r"$x_2$")
    ax3.set_zlabel("Error")
    fig.colorbar(s3, ax=ax3, shrink=0.60, pad=0.06)

    for ax in (ax1, ax2, ax3):
        ax.view_init(elev=35, azim=45)
        ax.tick_params(labelsize=9, pad=2)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_top_boundary_comparison(x1, u_exact_top, predictions_by_noise, save_path=None):
    """Plot top-boundary comparison and absolute errors for multiple noise levels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(x1, u_exact_top, color="b", lw=1.8, label="Exact")

    for noise_level, u_pred in sorted(predictions_by_noise.items()):
        label = f"{noise_level * 100:g}%"
        axes[0].plot(x1, u_pred, "--", lw=1.4, label=label)

    axes[0].set_title(r"Exact vs. ANN Solution at $x_2 = 1$")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$u(x_1,1)$")
    axes[0].legend()

    for noise_level, u_pred in sorted(predictions_by_noise.items()):
        label = f"{noise_level * 100:g}%"
        abs_err = np.abs(u_pred - u_exact_top)
        axes[1].plot(x1, abs_err, "--", lw=1.4, label=label)

    axes[1].set_title(r"Absolute Error at $x_2 = 1$")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel("Absolute Error")
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
