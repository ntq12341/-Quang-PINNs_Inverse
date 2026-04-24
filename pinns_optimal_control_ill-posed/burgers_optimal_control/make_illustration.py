from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("pinns_optimal_control_ill-posed/outputs/burgers_optimal_control")
SRC_DIR = ROOT / "optimal_control_results"
OUT_DIR = ROOT / "illustration"


def read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, np.asarray(rows, dtype=np.float64)


def write_csv(path: Path, header: list[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow([f"{v:.12g}" for v in row])


def make_curves(x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = np.maximum(target.max(), 1.0)
    bump = np.exp(-((x - 0.5) / 0.18) ** 2)
    wave = np.sin(4.0 * np.pi * x)
    pinn = target + 0.035 * width * wave * bump - 0.01 * width * np.cos(2.0 * np.pi * x)
    rollout = target + 0.055 * width * wave * bump - 0.015 * width * np.cos(2.5 * np.pi * x)
    pinn = np.clip(pinn, -0.02, 0.55)
    rollout = np.clip(rollout, -0.03, 0.57)
    return pinn, rollout


def plot_terminal(x: np.ndarray, target: np.ndarray, pinn: np.ndarray, rollout: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, target, color="black", lw=2.2, label="target")
    ax.plot(x, pinn, color="tab:orange", lw=1.8, ls="-.", label="PINN (illustrative)")
    ax.plot(x, rollout, color="tab:blue", lw=2.0, ls="--", label="rollout (illustrative)")
    ax.set_title("Illustrative Terminal State Shape")
    ax.set_xlabel("x")
    ax.legend(fontsize=9)
    ax.text(
        0.02,
        0.05,
        "Synthetic visualization only; not a training result.",
        transform=ax.transAxes,
        fontsize=9,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    term_header, term_data = read_csv(SRC_DIR / "terminal_state.csv")
    roll_header, roll_data = read_csv(SRC_DIR / "rollout_terminal_state.csv")

    x_term = term_data[:, 0]
    target_term = term_data[:, 1]
    pinn_term, _ = make_curves(x_term, target_term)

    x_roll = roll_data[:, 0]
    target_roll = roll_data[:, 1]
    _, rollout = make_curves(x_roll, target_roll)
    pinn_roll = np.interp(x_roll, x_term, pinn_term)

    term_out = np.column_stack([x_term, target_term, pinn_term])
    roll_out = np.column_stack([x_roll, target_roll, rollout])

    write_csv(OUT_DIR / "terminal_state.csv", term_header, term_out)
    write_csv(OUT_DIR / "rollout_terminal_state.csv", roll_header, roll_out)
    (OUT_DIR / "README.txt").write_text(
        "These files are illustrative only. They do not come from model training.\n",
        encoding="utf-8",
    )
    plot_terminal(x_roll, target_roll, pinn_roll, rollout, OUT_DIR / "fig_burgers_terminal_illustration.png")


if __name__ == "__main__":
    main()
