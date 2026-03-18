from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


def _masked(field, mask):
    return np.where(mask, field, np.nan)


def plot_example54_overview(
    X,
    Y,
    mask,
    U_exact,
    U_ann,
    noise_label="5%",
    gamma1_fraction=0.5,
    radius=0.5,
    save_path=None,
):
    """Layout like paper: Data (top) + Exact/ANN/Abs error (bottom)."""
    Ue = _masked(U_exact, mask)
    Ua = _masked(U_ann, mask)
    Uerr = _masked(np.abs(U_ann - U_exact), mask)

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1.0, 1.2],
        hspace=0.10,
        wspace=0.28,
    )

    # Top row: centered data panel
    ax_data = fig.add_subplot(gs[0, 1])
    t = np.linspace(0.0, 2.0 * np.pi, 1200)
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    split = 2.0 * np.pi * float(gamma1_fraction)
    gamma1 = t <= split
    gamma2 = t > split
    ax_data.plot(x[gamma1], y[gamma1], color="crimson", lw=1.6)
    ax_data.plot(x[gamma2], y[gamma2], color="blue", lw=1.6)
    scale = radius / 0.5
    ax_data.text(-0.82 * scale, 0.22 * scale, r"$\Gamma_1$", fontsize=8)
    ax_data.text(0.58 * scale, -0.66 * scale, r"$\Gamma_2$", fontsize=8)
    ax_data.text(-0.04 * scale, -0.04 * scale, r"$\Omega$", fontsize=8)
    ax_data.set_title("Data", fontsize=10)
    ax_data.set_xlabel(r"$x_1$", fontsize=9)
    ax_data.set_ylabel(r"$x_2$", fontsize=9)
    ax_data.set_aspect("equal")
    lim = 1.05 * radius
    ax_data.set_xlim([-lim, lim])
    ax_data.set_ylim([-lim, lim])
    ax_data.tick_params(labelsize=7)

    # Bottom row: three contour panels
    levels = 60

    ax1 = fig.add_subplot(gs[1, 0])
    c1 = ax1.contourf(X, Y, Ue, levels=levels, cmap="viridis")
    ax1.set_title("Exact solution", fontsize=11)
    ax1.set_xlabel(r"$x_1$", fontsize=9)
    ax1.set_ylabel(r"$x_2$", fontsize=9, labelpad=1)
    ax1.set_aspect("equal")
    ax1.tick_params(labelsize=8)
    cb1 = fig.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[1, 1])
    c2 = ax2.contourf(X, Y, Ua, levels=levels, cmap="viridis")
    ax2.set_title("ANN solution", fontsize=11)
    ax2.set_xlabel(r"$x_1$", fontsize=9)
    ax2.set_ylabel(r"$x_2$", fontsize=9, labelpad=1)
    ax2.set_aspect("equal")
    ax2.tick_params(labelsize=8)
    cb2 = fig.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[1, 2])
    c3 = ax3.contourf(X, Y, Uerr, levels=levels, cmap="viridis")
    ax3.set_title("Absolute error", fontsize=11)
    ax3.set_xlabel(r"$x_1$", fontsize=9)
    ax3.set_ylabel(r"$x_2$", fontsize=9, labelpad=1)
    ax3.set_aspect("equal")
    ax3.tick_params(labelsize=8)
    cb3 = fig.colorbar(c3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=7)

    fig.suptitle(f"Noise {noise_label}", y=0.98, fontsize=10)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_gamma2_comparison(theta, u_exact_gamma2, predictions_by_noise, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(theta, u_exact_gamma2, color="b", lw=1.8, label="Exact")
    for noise_level, u_pred in sorted(predictions_by_noise.items()):
        label = f"{noise_level * 100:g}%"
        axes[0].plot(theta, u_pred, "--", lw=1.4, label=label)

    axes[0].set_title(r"Exact vs. ANN on $\Gamma_2$")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"$u|_{\Gamma_2}$")
    axes[0].legend()

    for noise_level, u_pred in sorted(predictions_by_noise.items()):
        label = f"{noise_level * 100:g}%"
        abs_err = np.abs(u_pred - u_exact_gamma2)
        axes[1].plot(theta, abs_err, "--", lw=1.4, label=label)

    axes[1].set_title(r"Absolute Error on $\Gamma_2$")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel("Absolute Error")
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
