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
    v_smooth,
)
from network import PINN, train_lbfgs
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
    return x, y, u_exact(X, Y), u_ann


def run_example_51(
    noise_level=0.01,
    device=None,
    forward_iter=800,
    inverse_iter=1000,
    v_top_func=None,
):
    """
    Run Example 5.1 — không dùng Tikhonov regularization.

    Bài toán inverse dùng:
      - PDE loss (Laplace=0)
      - Dirichlet tại x=0, x=pi  (u=0, biết trước)
      - Neumann tại y=0           (du/dy=0, BC biết trước, KHÔNG noise)
      - Cauchy Dirichlet tại y=0  (u=u_delta, dữ liệu đo DUY NHẤT, CÓ noise)

    Vì chỉ có một loại Cauchy data nên bài toán ill-posed — thiếu regularization
    thì kết quả tại y=1 sẽ kém như paper chứng minh.
    """
    if v_top_func is None:
        v_top_func = v_smooth

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== Example 5.1 | noise={noise_level * 100:.1f}% | device={device} ===\n")

    # ------------------------------------------------------------------ #
    # Step 1: Giải bài toán forward
    # ------------------------------------------------------------------ #
    print("Step 1/2: Cauchy data = exact u(x,0) = sin(x)  [no forward model]")
    u_bottom_func = lambda x: np.sin(x)

    # ------------------------------------------------------------------ #
    # Step 2: Giải bài toán inverse
    # ------------------------------------------------------------------ #
    print("Step 2/2: solve inverse problem (no regularization)...")

    data_inverse = generate_inverse_data(
        u_bottom_func,
        noise_level=noise_level,
        N_r=5000,
        N_b=1000,
        N_cauchy=1000,
    )

    model_final = PINN().to(device)
    losses = train_lbfgs(
        model_final,
        data_inverse,
        max_iter=inverse_iter,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # Đánh giá tại y=1
    # ------------------------------------------------------------------ #
    x_eval = np.linspace(0.0, X1_MAX, 500)
    y_eval = np.full_like(x_eval, X2_MAX)
    with torch.no_grad():
        x_t = torch.tensor(x_eval.reshape(-1, 1), dtype=torch.float32, device=device)
        y_t = torch.tensor(y_eval.reshape(-1, 1), dtype=torch.float32, device=device)
        u_top_pred = model_final(x_t, y_t).cpu().numpy().flatten()

    u_top_exact = v_top_func(x_eval)
    l2_error = float(np.sqrt(np.mean((u_top_pred - u_top_exact) ** 2)))
    print(f"L2 error at x2=1: {l2_error:.6e}")

    x_field, y_field, u_field_exact, u_field_pred = _predict_field(model_final, device)

    return {
        "noise_level": noise_level,
        "alpha": None,
        "l2_error": l2_error,
        "losses": losses,
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
    inverse_iter=1000,
    v_top_func=None,
    seed=42,
    figs_dir=None,
    tables_dir=None,
):
    if v_top_func is None:
        v_top_func = v_smooth

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(__file__).resolve().parent
    figs_path  = base_dir / "results" / "figs"   if figs_dir   is None else Path(figs_dir)
    tables_path = base_dir / "results" / "tables" if tables_dir is None else Path(tables_dir)
    figs_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for noise in noise_levels:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        result = run_example_51(
            noise_level=noise,
            device=device,
            forward_iter=forward_iter,
            inverse_iter=inverse_iter,
            v_top_func=v_top_func,
        )
        all_results[noise] = result

    if surface_noise not in all_results:
        surface_noise = noise_levels[0]

    for noise in noise_levels:
        s = all_results[noise]
        plot_surface_comparison(
            s["x_field"], s["y_field"],
            s["u_field_exact"], s["u_field_pred"],
            noise_label=f"{noise * 100:g}%",
            save_path=figs_path / f"surface-noise-{noise * 100:g}.png",
        )

    surf = all_results[surface_noise]
    plot_surface_comparison(
        surf["x_field"], surf["y_field"],
        surf["u_field_exact"], surf["u_field_pred"],
        noise_label=f"{surface_noise * 100:g}%",
    )

    preds_by_noise = {n: all_results[n]["u_top_pred"] for n in noise_levels}
    plot_top_boundary_comparison(
        surf["x_top"], surf["u_top_exact"], preds_by_noise,
        save_path=figs_path / "top-boundary-all-noise.png",
    )

    written_csv = export_error_tables(all_results, output_dir=tables_path)
    plt.show()

    print("\nSummary (L2 at x2=1) — no regularization:")
    for noise in noise_levels:
        r = all_results[noise]
        print(f"noise={noise * 100:>4.1f}% | L2={r['l2_error']:.6e}")

    print(f"\nSaved figures to: {figs_path}")
    for fp in written_csv:
        print(f"  - {fp}")

    return all_results


if __name__ == "__main__":
    run_example_51_suite(
        noise_levels=(0.01, 0.03, 0.05, 0.075),
        v_top_func=v_smooth,
    )