from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_surface_overview(
    X1,
    X2,
    U_exact,
    U_ann,
    noise_label="5%",
    save_path=None,
):
    """Plot 3D surfaces: exact / ANN / error over the 2D domain Omega.

    X1, X2 : 2D meshgrid arrays (shape nx1 x nx2)
    """
    U_err = U_ann - U_exact

    fig = plt.figure(figsize=(18, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.26)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    s1 = ax1.plot_surface(X1, X2, U_exact, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title("Exact Solution")
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel("u")
    fig.colorbar(s1, ax=ax1, shrink=0.60, pad=0.06)

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    s2 = ax2.plot_surface(X1, X2, U_ann, cmap="viridis", linewidth=0, antialiased=True)
    ax2.set_title(f"ANN Solution, noise {noise_label}")
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel("u")
    fig.colorbar(s2, ax=ax2, shrink=0.60, pad=0.06)

    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    s3 = ax3.plot_surface(X1, X2, U_err, cmap="viridis", linewidth=0, antialiased=True)
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


def plot_line_x2_half(x1, u_exact_line, predictions_by_noise, save_path=None):
    """Plot u(x1, 0.5) and absolute errors for multiple noise levels.

    Corresponds to Figure 8 in the paper (Exact vs ANN at x2=0.5).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(x1, u_exact_line, color="b", lw=1.8, label="Exact")
    for noise_level, u_pred in sorted(predictions_by_noise.items()):
        label = f"{noise_level * 100:g}%"
        axes[0].plot(x1, u_pred, "--", lw=1.4, label=label)

    axes[0].set_title(r"Exact vs. ANN at $x_2 = 0.5$")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$u(x_1, 0.5)$")
    axes[0].legend()

    for noise_level, u_pred in sorted(predictions_by_noise.items()):
        label = f"{noise_level * 100:g}%"
        axes[1].plot(x1, np.abs(u_pred - u_exact_line), "--", lw=1.4, label=label)

    axes[1].set_title(r"Absolute Error at $x_2 = 0.5$")
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel("Absolute Error")
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_loss_history(histories_by_noise, save_path=None):
    """Plot training loss curves for multiple noise levels."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for noise_level, history in sorted(histories_by_noise.items()):
        label = f"Noise {noise_level * 100:g}%"
        ax.semilogy(history, lw=1.4, label=label)

    ax.set_title("Training Loss (L-BFGS)")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_l_curve(alpha_scan, optimal_alpha, save_path=None):
    """Plot the L-curve used for alpha selection."""
    if not alpha_scan:
        return None

    residuals = [r["residual"] for r in alpha_scan]
    regs = [r["regularization"] for r in alpha_scan]
    alphas = [r["alpha"] for r in alpha_scan]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(residuals, regs, "b.-", lw=1.4, label="L-curve")

    # Mark the selected alpha
    for r in alpha_scan:
        if abs(r["alpha"] - optimal_alpha) < 1e-15:
            ax.loglog(r["residual"], r["regularization"], "ro", ms=8,
                      label=rf"$\alpha^* = {optimal_alpha:.4e}$")
            break

    ax.set_xlabel(r"Residual norm $\|Au_\alpha - b\|_2$")
    ax.set_ylabel(r"Solution norm $\|u_\alpha\|_2$")
    ax.set_title("L-curve for Tikhonov regularization")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
