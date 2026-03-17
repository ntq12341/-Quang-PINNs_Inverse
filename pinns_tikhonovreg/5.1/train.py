from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generate import (
    X1_MAX,
    X2_MAX,
    generate_forward_data,
    generate_inverse_data,
    u_exact,
)
from network import PINN, l_curve_method, train_lbfgs
from perform_evaluate import export_error_tables
from visual import plot_surface_comparison, plot_top_boundary_comparison


def _predict_field(model, device, nx=120, ny=80):
    x = np.linspace(0.0, X1_MAX, nx)
    y = np.linspace(0.0, X2_MAX, ny)
    X, Y = np.meshgrid(x, y)

    with torch.no_grad():
        x_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
        y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, device=device)
        u_ann = model(x_t, y_t).cpu().numpy().reshape(ny, nx)

    u_ref = u_exact(X, Y)
    return x, y, u_ref, u_ann


def run_example_51(
    noise_level=0.01,
    device=None,
    forward_iter=800,
    lcurve_iter=200,
    inverse_iter=1000,
):
    """Run Example 5.1 from Hao-Shishlenin-Cong (2025)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== Example 5.1 | noise={noise_level * 100:.1f}% | device={device} ===\n")

    print("Step 1/3: solve the forward problem to generate bottom boundary data...")
    model_forward = PINN().to(device)
    data_forward = generate_forward_data(N_r=10000, N_b=1000)
    train_lbfgs(model_forward, data_forward, alpha=0.0, max_iter=forward_iter, device=device)

    x_bottom = np.linspace(0.0, X1_MAX, 400)
    y_bottom = np.zeros_like(x_bottom)
    with torch.no_grad():
        x_t = torch.tensor(x_bottom, dtype=torch.float32, device=device).reshape(-1, 1)
        y_t = torch.tensor(y_bottom, dtype=torch.float32, device=device).reshape(-1, 1)
        u_bottom = model_forward(x_t, y_t).cpu().numpy().flatten()

    print("Step 2/3: add noise to Cauchy data on x2=0...")
    u_bottom_func = lambda x: np.interp(x, x_bottom, u_bottom)
    data_inverse = generate_inverse_data(
        u_bottom_func,
        noise_level=noise_level,
        N_r=5000,
        N_b=1000,
        N_cauchy=1000,
        N_reg=500,
    )

    print("Step 3/3: solve inverse problem with L-curve alpha selection...")
    alphas = np.logspace(-5, -1, 10)
    model_for_alpha = PINN().to(device)
    optimal_alpha, l_curve_results = l_curve_method(
        model_for_alpha,
        data_inverse,
        alphas,
        max_iter=lcurve_iter,
        device=device,
    )
    print(f"Selected alpha (L-curve): {optimal_alpha:.6e}")

    model_final = PINN().to(device)
    losses = train_lbfgs(
        model_final,
        data_inverse,
        alpha=optimal_alpha,
        max_iter=inverse_iter,
        device=device,
    )

    x_eval = np.linspace(0.0, X1_MAX, 500)
    y_eval = np.full_like(x_eval, X2_MAX)
    with torch.no_grad():
        x_t = torch.tensor(x_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        y_t = torch.tensor(y_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        u_top_pred = model_final(x_t, y_t).cpu().numpy().flatten()

    u_top_exact = u_exact(x_eval, X2_MAX)
    l2_error = float(np.sqrt(np.mean((u_top_pred - u_top_exact) ** 2)))
    print(f"L2 error at x2=1: {l2_error:.6e}")

    x_field, y_field, u_field_exact, u_field_pred = _predict_field(model_final, device)

    return {
        "noise_level": noise_level,
        "alpha": optimal_alpha,
        "l2_error": l2_error,
        "losses": losses,
        "l_curve": l_curve_results,
        "model": model_final,
        "x_top": x_eval,
        "u_top_exact": u_top_exact,
        "u_top_pred": u_top_pred,
        "x_field": x_field,
        "y_field": y_field,
        "u_field_exact": u_field_exact,
        "u_field_pred": u_field_pred,
    }


def run_example_51_suite(
    noise_levels=(0.01, 0.03, 0.05, 0.075),
    surface_noise=0.03,
    device=None,
    forward_iter=800,
    lcurve_iter=200,
    inverse_iter=1000,
    figs_dir="results/figs",
    tables_dir="results/tables",
):
    """Run multiple noise levels, save figures, and export error tables."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    figs_path = Path(figs_dir)
    tables_path = Path(tables_dir)
    figs_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for noise in noise_levels:
        result = run_example_51(
            noise_level=noise,
            device=device,
            forward_iter=forward_iter,
            lcurve_iter=lcurve_iter,
            inverse_iter=inverse_iter,
        )
        all_results[noise] = result

    if surface_noise not in all_results:
        surface_noise = noise_levels[0]

    surf = all_results[surface_noise]
    plot_surface_comparison(
        surf["x_field"],
        surf["y_field"],
        surf["u_field_exact"],
        surf["u_field_pred"],
        noise_label=f"{surface_noise * 100:g}%",
        save_path=figs_path / f"surface-noise-{surface_noise * 100:g}.png",
    )

    x_top = surf["x_top"]
    u_top_exact = surf["u_top_exact"]
    preds_by_noise = {noise: all_results[noise]["u_top_pred"] for noise in noise_levels}
    plot_top_boundary_comparison(
        x_top,
        u_top_exact,
        preds_by_noise,
        save_path=figs_path / "top-boundary-all-noise.png",
    )

    written_csv = export_error_tables(all_results, output_dir=tables_path)

    plt.show()

    print("\nSummary (L2 at x2=1):")
    for noise in noise_levels:
        r = all_results[noise]
        print(f"noise={noise * 100:>4.1f}% | alpha={r['alpha']:.3e} | L2={r['l2_error']:.6e}")

    print(f"\nSaved figures to: {figs_path}")
    print("Saved CSV files:")
    for file_path in written_csv:
        print(f"  - {file_path}")

    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    run_example_51_suite()
