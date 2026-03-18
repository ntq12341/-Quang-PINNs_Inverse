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
    """Plot ANN surface only (single 3D panel)."""
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    s = ax.plot_surface(X, Y, u_ann_grid, cmap="viridis", linewidth=0, antialiased=True)

    ax.set_title(f"ANN Solution, noise {noise_label}")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("u")
    ax.view_init(elev=35, azim=45)
    fig.colorbar(s, ax=ax, shrink=0.7, pad=0.08)

    fig.tight_layout()
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
