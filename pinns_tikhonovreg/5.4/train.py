from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generate import (
    evaluation_boundary_gamma2,
    evaluation_grid,
    generate_inverse_data_circle,
    u_exact,
)
from network import (
    PINN,
    discrepancy_principle_method,
    l_curve_method,
    train_lbfgs,
)
from perform_evaluate import export_error_tables
from visual import plot_example54_overview


def _predict_grid(model, device, n=180):
    X, Y, mask = evaluation_grid(n=n)
    with torch.no_grad():
        x_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
        y_t = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, device=device)
        u_ann = model(x_t, y_t).cpu().numpy().reshape(X.shape)
    return X, Y, mask, u_ann


def run_example_54(
    noise_level=0.01,
    gamma1_fraction=0.5,
    device=None,
    lcurve_iter=400,
    inverse_iter=1000,
    alpha_method="discrepancy",
    discrepancy_tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g"),
):
    """Example 5.4/5.8-style circle Cauchy setup with configurable Γ1 length."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"\n=== Circle Cauchy | noise={noise_level * 100:.1f}% "
        f"| gamma1={gamma1_fraction:.2f} | device={device} ===\n"
    )

    data_inverse = generate_inverse_data_circle(
        noise_level=noise_level,
        N_r=5000,
        N_b=1500,
        N_reg=500,
        gamma1_fraction=gamma1_fraction,
    )

    loss_weights = {
        "pde": 1.0,
        "cauchy_u": 8.0,
        "cauchy_g": 8.0,
        "regularization": 1.0,
    }

    alphas = np.logspace(-6, -2, 10)
    model_for_alpha = PINN().to(device)

    if alpha_method.lower() == "lcurve":
        print("Choosing alpha by L-curve...")
        optimal_alpha, alpha_scan = l_curve_method(
            model_for_alpha,
            data_inverse,
            alphas,
            max_iter=lcurve_iter,
            device=device,
            loss_weights=loss_weights,
        )
        discrepancy_target = None
        print(f"Selected alpha (L-curve): {optimal_alpha:.6e}")
    elif alpha_method.lower() == "discrepancy":
        print("Choosing alpha by discrepancy principle...")
        optimal_alpha, alpha_scan, discrepancy_target = discrepancy_principle_method(
            model_for_alpha,
            data_inverse,
            alphas,
            noise_level=noise_level,
            max_iter=lcurve_iter,
            device=device,
            loss_weights=loss_weights,
            tau=discrepancy_tau,
            discrepancy_terms=discrepancy_terms,
        )
        used_terms = alpha_scan[0]["used_terms"] if alpha_scan else tuple(discrepancy_terms)
        print(
            f"Selected alpha (discrepancy, target={discrepancy_target:.3e}, "
            f"terms={used_terms}): {optimal_alpha:.6e}"
        )
    else:
        raise ValueError("alpha_method must be 'lcurve' or 'discrepancy'.")

    model_final = PINN().to(device)
    losses = train_lbfgs(
        model_final,
        data_inverse,
        alpha=optimal_alpha,
        max_iter=inverse_iter,
        device=device,
        loss_weights=loss_weights,
    )

    theta_eval, x_eval, y_eval = evaluation_boundary_gamma2(
        n_points=500, gamma1_fraction=gamma1_fraction
    )
    with torch.no_grad():
        x_t = torch.tensor(x_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        y_t = torch.tensor(y_eval, dtype=torch.float32, device=device).reshape(-1, 1)
        u_eval_pred = model_final(x_t, y_t).cpu().numpy().flatten()

    u_eval_exact = u_exact(x_eval, y_eval)
    l2_error = float(np.sqrt(np.mean((u_eval_pred - u_eval_exact) ** 2)))
    print(f"L2 error on Gamma2: {l2_error:.6e}")

    X, Y, mask, U_ann = _predict_grid(model_final, device, n=180)

    return {
        "noise_level": noise_level,
        "gamma1_fraction": gamma1_fraction,
        "alpha": optimal_alpha,
        "alpha_method": alpha_method,
        "discrepancy_target": discrepancy_target,
        "l2_error": l2_error,
        "losses": losses,
        "alpha_scan": alpha_scan,
        "model": model_final,
        "theta_eval": theta_eval,
        "u_eval_exact": u_eval_exact,
        "u_eval_pred": u_eval_pred,
        "X_grid": X,
        "Y_grid": Y,
        "mask_grid": mask,
        "U_ann_grid": U_ann,
    }


def run_example_54_suite(
    noise_levels=(0.01, 0.03, 0.05),
    gamma1_fraction=0.75,
    surface_noise=0.05,
    device=None,
    lcurve_iter=400,
    inverse_iter=1000,
    figs_dir=None,
    tables_dir=None,
    alpha_method="discrepancy",
    discrepancy_tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g"),
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(__file__).resolve().parent
    figs_path = base_dir / "results" / "figs" if figs_dir is None else Path(figs_dir)
    tables_path = base_dir / "results" / "tables" if tables_dir is None else Path(tables_dir)
    figs_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for noise in noise_levels:
        r = run_example_54(
            noise_level=noise,
            gamma1_fraction=gamma1_fraction,
            device=device,
            lcurve_iter=lcurve_iter,
            inverse_iter=inverse_iter,
            alpha_method=alpha_method,
            discrepancy_tau=discrepancy_tau,
            discrepancy_terms=discrepancy_terms,
        )
        all_results[noise] = r

    if surface_noise not in all_results:
        surface_noise = noise_levels[0]

    # Only overview images for each noise.
    for noise in noise_levels:
        r = all_results[noise]
        U_exact_grid = u_exact(r["X_grid"], r["Y_grid"])
        plot_example54_overview(
            r["X_grid"],
            r["Y_grid"],
            r["mask_grid"],
            U_exact_grid,
            r["U_ann_grid"],
            noise_label=f"{noise * 100:g}%",
            gamma1_fraction=gamma1_fraction,
            radius=float(np.max(np.abs(r["X_grid"]))),
            save_path=figs_path / f"example54-overview-noise-{noise * 100:g}.png",
        )

    plt.show()

    written_csv = export_error_tables(all_results, output_dir=tables_path)

    print("\nSummary (L2 on Gamma2):")
    for noise in noise_levels:
        r = all_results[noise]
        print(
            f"noise={noise * 100:>4.1f}% | alpha={r['alpha']:.3e} "
            f"| method={r['alpha_method']} | L2={r['l2_error']:.6e}"
        )

    print(f"\nSaved figures to: {figs_path}")
    print("Saved CSV files:")
    for fp in written_csv:
        print(f"  - {fp}")

    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    run_example_54_suite(
        noise_levels=(0.01, 0.03, 0.05),
        gamma1_fraction=0.75,
        surface_noise=0.05,
        alpha_method="discrepancy",
        discrepancy_tau=1.1,
        discrepancy_terms=("pde", "cauchy_u", "cauchy_g"),
    )
