from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generate import (
    evaluation_grid,
    evaluation_line_x1_half_x3_one,
    evaluation_slice_x3_eq_1,
    generate_inverse_data_55,
    u_exact,
)
from network import (
    PINN,
    discrepancy_principle_method,
    l_curve_method,
    train_lbfgs,
)
from perform_evaluate import export_error_tables
from visual import plot_line_x1_half_x3_one, plot_slice_x3_one_overview


def _predict_points(model, device, x1, x2, x3):
    with torch.no_grad():
        x1_t = torch.tensor(x1.reshape(-1, 1), dtype=torch.float32, device=device)
        x2_t = torch.tensor(x2.reshape(-1, 1), dtype=torch.float32, device=device)
        x3_t = torch.tensor(x3.reshape(-1, 1), dtype=torch.float32, device=device)
        u_pred = model(x1_t, x2_t, x3_t).cpu().numpy().reshape(x1.shape)
    return u_pred


def run_example_55(
    noise_level=0.01,
    device=None,
    lcurve_iter=600,          # FIX: tăng từ 400 → 600 để L-curve ổn định hơn
    inverse_iter=1000,
    alpha_method="lcurve",
    fixed_alpha=4.822e-3,
    discrepancy_tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g", "gamma3", "gamma4", "gamma5", "gamma6"),
    seed=42,                  # FIX: thêm tham số seed
):
    """Run Example 5.5 (3D Poisson Cauchy problem)."""

    # FIX: reset seed mỗi lần gọi để mỗi noise level bắt đầu từ cùng trạng thái RNG,
    # tránh việc kết quả noise 3% phụ thuộc vào trạng thái RNG còn lại từ noise 1%.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== Example 5.5 | noise={noise_level * 100:.1f}% | device={device} ===\n")

    data_inverse = generate_inverse_data_55(
        noise_level=noise_level,
        N_r=6000,
        N_b=2000,
        N_k=500,
    )

    loss_weights = {
        "pde": 1.0,
        "cauchy_u": 1.0,
        "cauchy_g": 1.0,
        "gamma3": 1.0,
        "gamma4": 1.0,
        "gamma5": 1.0,
        "gamma6": 1.0,
        "regularization": 1.0,
    }

    alphas = np.logspace(-4, -2, 10)

    # FIX: reset seed lần nữa trước khi khởi tạo model để weights khởi đầu như nhau
    # cho mọi noise level khi dùng L-curve
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
        raise ValueError("alpha_method must be 'fixed', 'lcurve' or 'discrepancy'.")

    # FIX: reset seed trước khi train model final để weights khởi đầu nhất quán
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

    X1g, X2g, X3g = evaluation_grid(n=56)
    U_exact_g = u_exact(X1g, X2g, X3g)
    U_pred_g = _predict_points(model_final, device, X1g, X2g, X3g)
    l2_global = float(np.sqrt(np.mean((U_pred_g - U_exact_g) ** 2)))

    x1_line, x2_line, x3_line = evaluation_line_x1_half_x3_one(n=600)
    u_line_exact = u_exact(x1_line, x2_line, x3_line)
    u_line_pred = _predict_points(model_final, device, x1_line, x2_line, x3_line).reshape(-1)
    l2_line = float(np.sqrt(np.mean((u_line_pred - u_line_exact) ** 2)))

    X1s, X2s, X3s = evaluation_slice_x3_eq_1(nx1=90, nx2=90)
    U_slice_exact = u_exact(X1s, X2s, X3s)
    U_slice_pred = _predict_points(model_final, device, X1s, X2s, X3s)

    print(f"L2 global in Omega: {l2_global:.6e}")
    print(f"L2 at x1=0.5, x3=1: {l2_line:.6e}")

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
        "x2_line": x2_line,
        "u_line_exact": u_line_exact,
        "u_line_pred": u_line_pred,
        "X1_slice": X1s,
        "X2_slice": X2s,
        "U_slice_exact": U_slice_exact,
        "U_slice_pred": U_slice_pred,
    }


def run_example_55_suite(
    noise_levels=(0.01, 0.03, 0.05),
    surface_noise=0.05,
    device=None,
    lcurve_iter=600,          # FIX: tăng từ 400 → 600
    inverse_iter=1000,
    figs_dir=None,
    tables_dir=None,
    alpha_method="lcurve",
    fixed_alpha=4.822e-3,
    discrepancy_tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g", "gamma3", "gamma4", "gamma5", "gamma6"),
    seed=42,                  # FIX: thêm tham số seed
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
        # FIX: truyền seed vào từng run → mỗi noise level bắt đầu từ cùng RNG state
        r = run_example_55(
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

    if surface_noise not in all_results:
        surface_noise = noise_levels[0]

    for noise in noise_levels:
        r = all_results[noise]
        plot_slice_x3_one_overview(
            r["X1_slice"],
            r["X2_slice"],
            r["U_slice_exact"],
            r["U_slice_pred"],
            noise_label=f"{noise * 100:g}%",
            save_path=figs_path / f"example55-slice-x3-1-noise-{noise * 100:g}.png",
        )

    x2_line = all_results[surface_noise]["x2_line"]
    u_exact_line = all_results[surface_noise]["u_line_exact"]
    preds_by_noise = {noise: all_results[noise]["u_line_pred"] for noise in noise_levels}
    plot_line_x1_half_x3_one(
        x2_line,
        u_exact_line,
        preds_by_noise,
        save_path=figs_path / "example55-line-x1-0.5-x3-1-all-noise.png",
    )

    written_csv = export_error_tables(all_results, output_dir=tables_path)

    plt.show()

    print("\nSummary:")
    for noise in noise_levels:
        r = all_results[noise]
        print(
            f"noise={noise * 100:>4.1f}% | alpha={r['alpha']:.3e} | method={r['alpha_method']} "
            f"| L2_global={r['l2_global']:.6e} | L2_line={r['l2_line']:.6e}"
        )

    print(f"\nSaved figures to: {figs_path}")
    print("Saved CSV files:")
    for fp in written_csv:
        print(f"  - {fp}")

    return all_results


if __name__ == "__main__":
    run_example_55_suite(
        noise_levels=(0.01, 0.03, 0.05),
        surface_noise=0.05,
        alpha_method="lcurve",
        seed=42,
    )