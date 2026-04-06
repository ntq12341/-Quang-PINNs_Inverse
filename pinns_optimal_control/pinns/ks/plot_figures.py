from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220


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


def _load_csv_2d(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))


def plot_fig6_forward(forward_dir: Path, out_png: Path) -> None:
    field = _load_csv_2d(forward_dir / "field_u.csv")
    loss = _load_csv_2d(forward_dir / "loss_history.csv")
    test = _load_csv_2d(forward_dir / "test_error_history.csv")
    xx, tt, values = _reshape_field(field, [2, 3, 4])
    u_true, u_pred, err = values

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    axs[0, 0].semilogy(loss[:, 0], loss[:, 1], label="total")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 2], label="PDE")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 3], label="BC")
    axs[0, 0].semilogy(loss[:, 0], loss[:, 4], label="IC")
    axs[0, 0].set_title("(a) Loss Components")
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].semilogy(test[:, 0], test[:, 1], color="tab:red")
    axs[0, 1].set_title("(b) Relative L2 Test Error")

    axs[0, 2].plot(xx[-1], u_true[-1], label="Spectral")
    axs[0, 2].plot(xx[-1], u_pred[-1], "--", label="PINN")
    axs[0, 2].set_title("(c) Final-Time Snapshot")
    axs[0, 2].legend(fontsize=8)

    c = axs[1, 0].contourf(xx, tt, u_true, levels=40, cmap="viridis")
    axs[1, 0].set_title("(d) Spectral $u_s$")
    fig.colorbar(c, ax=axs[1, 0])

    c = axs[1, 1].contourf(xx, tt, u_pred, levels=40, cmap="viridis")
    axs[1, 1].set_title("(e) PINN $u$")
    fig.colorbar(c, ax=axs[1, 1])

    c = axs[1, 2].contourf(xx, tt, err, levels=40, cmap="magma")
    axs[1, 2].set_title("(f) Absolute Error")
    fig.colorbar(c, ax=axs[1, 2])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_fig7_optimal(control_dir: Path, dal_dir: Path, out_png: Path, sweep_dir: Path | None = None) -> None:
    ctl_loss = _load_csv_2d(control_dir / "loss_history.csv")
    ctl_f = _load_csv_2d(control_dir / "control_f.csv")
    ctl_u = _load_csv_2d(control_dir / "field_u.csv")
    dal_hist = _load_csv_2d(dal_dir / "history.csv")
    dal_f = _load_csv_2d(dal_dir / "control_f.csv")
    dal_u = _load_csv_2d(dal_dir / "field_u.csv")
    sweep = _load_csv_2d(sweep_dir / "summary.csv") if sweep_dir and (sweep_dir / "summary.csv").exists() else None

    xx_f, tt_f, ctl_f_vals = _reshape_field(ctl_f, [2])
    ctl_force = ctl_f_vals[0]
    _, _, dal_f_vals = _reshape_field(dal_f, [2])
    dal_force = dal_f_vals[0]

    xx_u, tt_u, ctl_u_vals = _reshape_field(ctl_u, [2, 3])
    ctl_u_pinn, ctl_u_rollout = ctl_u_vals
    _, _, dal_u_vals = _reshape_field(dal_u, [2])
    dal_u_rollout = dal_u_vals[0]

    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    if sweep is not None and sweep.shape[1] >= 4:
        axs[0, 0].loglog(sweep[:, 0], sweep[:, 1], marker="o", label=r"$L_{F/B/I}$")
        axs[0, 0].loglog(sweep[:, 0], sweep[:, 2], marker="s", label=r"$L_J$")
        axs[0, 0].set_title("(a) Step-1 Line Search")
        axs[0, 0].legend(fontsize=8)

        best = int(np.argmin(sweep[:, 3]))
        axs[0, 1].semilogx(sweep[:, 0], sweep[:, 3], marker="o")
        axs[0, 1].scatter([sweep[best, 0]], [sweep[best, 3]], color="red", zorder=3)
        axs[0, 1].set_title("(b) Step-2 Objective Check")
    else:
        axs[0, 0].text(0.1, 0.5, "Missing sweep data\nfor step-1 line search.", fontsize=11)
        axs[0, 0].axis("off")
        axs[0, 1].text(0.1, 0.5, "Missing sweep data\nfor step-2 forward check.", fontsize=11)
        axs[0, 1].axis("off")

    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 1], label="total")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 2], label="PDE")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 3], label="BC")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 4], label="IC")
    axs[0, 2].semilogy(ctl_loss[:, 0], ctl_loss[:, 5], label="J")
    axs[0, 2].set_title("(c) PINN Training")
    axs[0, 2].legend(fontsize=8)

    axs[1, 0].semilogy(dal_hist[:, 0], dal_hist[:, 1], color="tab:green")
    axs[1, 0].set_title("(d) DAL Objective History")

    c = axs[1, 1].contourf(xx_f, tt_f, ctl_force, levels=40, cmap="coolwarm")
    axs[1, 1].set_title("(e) PINN Optimal Forcing")
    fig.colorbar(c, ax=axs[1, 1])

    c = axs[1, 2].contourf(xx_f, tt_f, dal_force, levels=40, cmap="coolwarm")
    axs[1, 2].set_title("(f) DAL Optimal Forcing")
    fig.colorbar(c, ax=axs[1, 2])

    axs[2, 0].plot(xx_u[-1], ctl_u_rollout[-1], label="PINN rollout")
    axs[2, 0].plot(xx_u[-1], dal_u_rollout[-1], label="DAL rollout")
    axs[2, 0].plot(xx_u[-1], np.zeros_like(xx_u[-1]), "--", label="target")
    axs[2, 0].set_title("(g) Final-Time Snapshot")
    axs[2, 0].legend(fontsize=8)

    c = axs[2, 1].contourf(xx_u, tt_u, ctl_u_rollout, levels=40, cmap="viridis")
    axs[2, 1].set_title("(h) State Using PINN Forcing")
    fig.colorbar(c, ax=axs[2, 1])

    c = axs[2, 2].contourf(xx_u, tt_u, dal_u_rollout, levels=40, cmap="viridis")
    axs[2, 2].set_title("(i) State Using DAL Forcing")
    fig.colorbar(c, ax=axs[2, 2])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create figures for Kuramoto-Sivashinsky example.")
    parser.add_argument("--forward-dir", default="pinns_optimal_control/outputs/ks/forward/forward_results")
    parser.add_argument("--control-dir", default="pinns_optimal_control/outputs/ks/optimal_control/optimal_control_results")
    parser.add_argument("--dal-dir", default="pinns_optimal_control/outputs/ks/dal/dal_ks_optimal_control")
    parser.add_argument("--sweep-dir", default="pinns_optimal_control/outputs/ks/optimal_control/sweep_wj/sweep_wj_results")
    parser.add_argument("--outdir", default="pinns_optimal_control/outputs/ks/figures")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    _set_style()
    outdir = Path(args.outdir)
    plot_fig6_forward(Path(args.forward_dir), outdir / "fig6_forward_ks.png")
    plot_fig7_optimal(
        Path(args.control_dir),
        Path(args.dal_dir),
        outdir / "fig7_optimal_ks.png",
        Path(args.sweep_dir),
    )
    print(f"Saved figures in: {outdir}")


if __name__ == "__main__":
    main()
