from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generate import (
    evaluation_grid,
    evaluation_line_x2_half,
    generate_inverse_data_56,
    u_exact,
)
from network import (
    PINN,
    discrepancy_principle_method,
    l_curve_method,
    train_lbfgs,
)
from perform_evaluate import export_error_tables
from visual import (
    plot_l_curve,
    plot_line_x2_half,
    plot_loss_history,
    plot_surface_overview,
)


def _predict_points(model, device, x1, x2):
    """Run forward pass without gradient tracking."""
    with torch.no_grad():
        x1_t = torch.tensor(x1.reshape(-1, 1), dtype=torch.float32, device=device)
        x2_t = torch.tensor(x2.reshape(-1, 1), dtype=torch.float32, device=device)
        u_pred = model(x1_t, x2_t).cpu().numpy().reshape(x1.shape)
    return u_pred


def run_example_56(
    noise_level=0.01,
    device=None,
    lcurve_iter=600,
    inverse_iter=1000,
    alpha_method="lcurve",
    fixed_alpha=1.714e-3,       # paper value for Example 5.6
    discrepancy_tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g", "gamma3", "gamma4"),
    seed=42,
):
    """Run Example 5.6 (2D nonlinear Cauchy problem in a parallelepiped).

    Domain  : Omega = (0,1)x(0,0.5)
    Equation: -div(q(u)*grad(u)) = h,  q(u)=1+u^2
    Exact   : u_bar = x1^2 - x2^2 + 5x1 - 2x2 - 3x1*x2
    Cauchy  : u, q(u)*du/dn  on Gamma1 (x2=0)
    Dirichlet: u on Gamma3 (x1=0) and Gamma4 (x1=1)
    Regularization: ||du/dn||^2 on Gamma2 (x2=0.5)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== Example 5.6 | noise={noise_level * 100:.1f}% | device={device} ===\n")

    data_inverse = generate_inverse_data_56(
        noise_level=noise_level,
        N_r=6000,
        N_b=2000,
        N_k=600,
    )

    loss_weights = {
        "pde": 1.0,
        "cauchy_u": 1.0,
        "cauchy_g": 1.0,
        "gamma3": 1.0,
        "gamma4": 1.0,
        "regularization": 1.0,
    }

    alphas = np.logspace(-4, -1, 20)

    # -----------------------------------------------------------------------
    # Alpha selection
    # -----------------------------------------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_for_alpha = PINN().to(device)

    if alpha_method.lower() == "fixed":
        optimal_alpha = float(fixed_alpha)
        alpha_scan = []
        discrepancy_target = None
        print(f"Using fixed alpha: {optimal_alpha:.6e}")

    elif alpha_method.lower() == "lcurve":
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
        raise ValueError("alpha_method must be 'fixed', 'lcurve', or 'discrepancy'.")

    # -----------------------------------------------------------------------
    # Final training with optimal alpha
    # -----------------------------------------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_final = PINN().to(device)
    losses = train_lbfgs(
        model_final,
        data_inverse,
        alpha=optimal_alpha,
        max_iter=inverse_iter,
        device=device,
        loss_weights=loss_weights,
    )

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    X1g, X2g = evaluation_grid(nx1=90, nx2=45)
    U_exact_g = u_exact(X1g, X2g)
    U_pred_g = _predict_points(model_final, device, X1g, X2g)
    l2_global = float(np.sqrt(np.mean((U_pred_g - U_exact_g) ** 2)))

    x1_line, x2_line = evaluation_line_x2_half(n=600)
    u_line_exact = u_exact(x1_line, x2_line)
    u_line_pred = _predict_points(model_final, device, x1_line, x2_line).reshape(-1)
    l2_line = float(np.sqrt(np.mean((u_line_pred - u_line_exact) ** 2)))

    print(f"L2 global in Omega    : {l2_global:.6e}")
    print(f"L2 at x2=0.5 (Gamma2) : {l2_line:.6e}")

    return {
        "noise_level": noise_level,
        "alpha": optimal_alpha,
        "alpha_method": alpha_method,
        "discrepancy_target": discrepancy_target,
        "l2_global": l2_global,
        "l2_line": l2_line,
        "losses": losses,
        "alpha_scan": alpha_scan,
        "model": model_final,
        # Grid results for surface plots
        "X1_grid": X1g,
        "X2_grid": X2g,
        "U_grid_exact": U_exact_g,
        "U_grid_pred": U_pred_g,
        # Line results for 1D comparison (paper Fig 8)
        "x1_line": x1_line,
        "u_line_exact": u_line_exact,
        "u_line_pred": u_line_pred,
    }


def run_example_56_suite(
    noise_levels=(0.01, 0.03, 0.05),
    surface_noise=0.03,
    device=None,
    lcurve_iter=600,
    inverse_iter=1000,
    figs_dir=None,
    tables_dir=None,
    alpha_method="lcurve",
    fixed_alpha=1.714e-3,
    discrepancy_tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g", "gamma3", "gamma4"),
    seed=42,
):
    """Run Example 5.6 for multiple noise levels, generate figures and CSV tables."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(__file__).resolve().parent
    figs_path = base_dir / "results56" / "figs" if figs_dir is None else Path(figs_dir)
    tables_path = base_dir / "results56" / "tables" if tables_dir is None else Path(tables_dir)
    figs_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    histories_by_noise = {}

    for noise in noise_levels:
        r = run_example_56(
            noise_level=noise,
            device=device,
            lcurve_iter=lcurve_iter,
            inverse_iter=inverse_iter,
            alpha_method=alpha_method,
            fixed_alpha=fixed_alpha,
            discrepancy_tau=discrepancy_tau,
            discrepancy_terms=discrepancy_terms,
            seed=seed,
        )
        all_results[noise] = r
        histories_by_noise[noise] = r["losses"]

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------

    # Surface plots (exact / ANN / error) for each noise level
    if surface_noise not in all_results:
        surface_noise = noise_levels[0]

    for noise in noise_levels:
        r = all_results[noise]
        plot_surface_overview(
            r["X1_grid"],
            r["X2_grid"],
            r["U_grid_exact"],
            r["U_grid_pred"],
            noise_label=f"{noise * 100:g}%",
            save_path=figs_path / f"example56-surface-noise-{noise * 100:g}.png",
        )

    # Line comparison at x2=0.5 for all noise levels (paper Fig 8)
    x1_line = all_results[surface_noise]["x1_line"]
    u_exact_line = all_results[surface_noise]["u_line_exact"]
    preds_by_noise = {noise: all_results[noise]["u_line_pred"] for noise in noise_levels}
    plot_line_x2_half(
        x1_line,
        u_exact_line,
        preds_by_noise,
        save_path=figs_path / "example56-line-x2-0.5-all-noise.png",
    )

    # Training loss curves
    plot_loss_history(
        histories_by_noise,
        save_path=figs_path / "example56-loss-history.png",
    )

    # L-curve for the surface_noise run
    r_surf = all_results[surface_noise]
    if r_surf.get("alpha_scan"):
        plot_l_curve(
            r_surf["alpha_scan"],
            r_surf["alpha"],
            save_path=figs_path / f"example56-lcurve-noise-{surface_noise * 100:g}.png",
        )

    # -----------------------------------------------------------------------
    # CSV tables
    # -----------------------------------------------------------------------
    written_csv = export_error_tables(all_results, output_dir=tables_path)

    plt.show()

    print("\nSummary:")
    for noise in noise_levels:
        r = all_results[noise]
        print(
            f"noise={noise * 100:>4.1f}% | alpha={r['alpha']:.3e} | method={r['alpha_method']} "
            f"| L2_global={r['l2_global']:.6e} | L2_line(x2=0.5)={r['l2_line']:.6e}"
        )

    print(f"\nSaved figures to : {figs_path}")
    print("Saved CSV files:")
    for fp in written_csv:
        print(f"  - {fp}")

    return all_results


if __name__ == "__main__":
    run_example_56_suite(
        noise_levels=(0.01, 0.03, 0.05),
        surface_noise=0.03,
        alpha_method="lcurve",
        seed=42,
    )