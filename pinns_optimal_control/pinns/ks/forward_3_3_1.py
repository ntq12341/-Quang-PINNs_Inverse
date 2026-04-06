from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.common import set_seed
from pinns_optimal_control.pinns.ks.core import (
    KSSpectralSolver,
    NormalizedMLP,
    TrainHistory,
    flatten_field_to_rows,
    gradients,
    ks_initial_condition_torch,
    ks_residual,
    lhs_2d,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN forward solver for Kuramoto-Sivashinsky example (paper section 3.3.1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--L", type=float, default=50.0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epochs", type=int, nargs="*", default=[3000, 5000])
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=40000)
    parser.add_argument("--batch-residual", type=int, default=2000)
    parser.add_argument("--n-boundary-time", type=int, default=82)
    parser.add_argument("--n-initial", type=int, default=41)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--hidden-width", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--ref-nx", type=int, default=128)
    parser.add_argument("--ref-dt", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/ks/forward")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def periodic_bc_loss(model: NormalizedMLP, left_t: torch.Tensor, right_t: torch.Tensor) -> torch.Tensor:
    u_left = model(left_t)
    u_right = model(right_t)
    ux_left = gradients(u_left, left_t, order=1)[:, :1]
    ux_right = gradients(u_right, right_t, order=1)[:, :1]
    uxx_left = gradients(ux_left, left_t, order=1)[:, :1]
    uxx_right = gradients(ux_right, right_t, order=1)[:, :1]
    uxxx_left = gradients(uxx_left, left_t, order=1)[:, :1]
    uxxx_right = gradients(uxx_right, right_t, order=1)[:, :1]
    return (
        torch.mean((u_left - u_right) ** 2)
        + torch.mean((ux_left - ux_right) ** 2)
        + torch.mean((uxx_left - uxx_right) ** 2)
        + torch.mean((uxxx_left - uxxx_right) ** 2)
    )


def build_reference(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    solver = KSSpectralSolver(nx=args.ref_nx, L=args.L, T=args.T, dt=args.ref_dt, sigma=1.0)
    states = solver.rollout()
    xx, tt = np.meshgrid(solver.x, solver.t, indexing="xy")
    xt = torch.tensor(np.column_stack([xx.reshape(-1), tt.reshape(-1)]), dtype=torch.float32, device=device)
    u_ref = torch.tensor(states, dtype=torch.float32, device=device)
    return xt, u_ref


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, TrainHistory, list[int], list[float], KSSpectralSolver, np.ndarray]:
    set_seed(args.seed)
    device = torch.device(args.device)

    xt_r = lhs_2d(args.n_residual, low=(0.0, 0.0), high=(args.L, args.T), seed=args.seed)
    xt_r_t = as_tensor(xt_r, device=device, requires_grad=True)

    t_b = np.linspace(0.0, args.T, args.n_boundary_time, dtype=np.float32).reshape(-1, 1)
    xt_left = np.concatenate([np.zeros_like(t_b), t_b], axis=1)
    xt_right = np.concatenate([args.L * np.ones_like(t_b), t_b], axis=1)
    x0 = np.linspace(0.0, args.L, args.n_initial, endpoint=False, dtype=np.float32).reshape(-1, 1)
    xt0 = np.concatenate([x0, np.zeros_like(x0)], axis=1)

    left_t = as_tensor(xt_left, device=device, requires_grad=True)
    right_t = as_tensor(xt_right, device=device, requires_grad=True)
    init_t = as_tensor(xt0, device=device, requires_grad=False)

    mean = xt_r_t.detach().mean(dim=0)
    std = xt_r_t.detach().std(dim=0) + 1e-6
    model = NormalizedMLP(2, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = TrainHistory(total=[], pde=[], bc=[], ic=[], obj=[])

    xt_eval, u_eval = build_reference(args, device)
    rel_epochs: list[int] = []
    rel_values: list[float] = []
    n_batches = max(1, args.n_residual // args.batch_residual)
    lr_drop_epochs = set(args.lr_drop_epochs)

    for epoch in range(1, args.epochs + 1):
        if epoch in lr_drop_epochs:
            for group in opt.param_groups:
                group["lr"] *= args.lr_drop_factor

        perm = torch.randperm(args.n_residual, device=device)
        pde_accum = 0.0
        for k in range(n_batches):
            ids = perm[k * args.batch_residual : (k + 1) * args.batch_residual]
            xt_batch = xt_r_t[ids]
            xt_batch.requires_grad_(True)
            u_r = model(xt_batch)
            pde_loss = torch.mean(ks_residual(u_r, xt_batch, torch.zeros_like(u_r)) ** 2)
            bc_loss = periodic_bc_loss(model, left_t, right_t)
            ic_loss = torch.mean((model(init_t) - ks_initial_condition_torch(init_t[:, :1], L=args.L)) ** 2)

            loss = pde_loss + bc_loss + ic_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pde_accum += pde_loss.item()

        pde_mean = pde_accum / n_batches
        bc_val = periodic_bc_loss(model, left_t, right_t)
        with torch.no_grad():
            ic_val = torch.mean((model(init_t) - ks_initial_condition_torch(init_t[:, :1], L=args.L)) ** 2)
        total = pde_mean + bc_val.item() + ic_val.item()
        history.total.append(total)
        history.pde.append(pde_mean)
        history.bc.append(bc_val.item())
        history.ic.append(ic_val.item())
        history.obj.append(0.0)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                u_pred = model(xt_eval).reshape(u_eval.shape)
            rel_epochs.append(epoch)
            rel_values.append(relative_l2(u_pred, u_eval))

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[ks forward] epoch={epoch:5d} total={total:.3e} "
                f"pde={pde_mean:.3e} bc={bc_val.item():.3e} ic={ic_val.item():.3e}"
            )

    solver = KSSpectralSolver(nx=args.ref_nx, L=args.L, T=args.T, dt=args.ref_dt, sigma=1.0)
    states = solver.rollout()
    return model, history, rel_epochs, rel_values, solver, states


def evaluate_and_save(
    model: NormalizedMLP,
    history: TrainHistory,
    rel_epochs: list[int],
    rel_values: list[float],
    solver: KSSpectralSolver,
    states_ref: np.ndarray,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    outdir = Path(args.outdir) / "forward_results"
    outdir.mkdir(parents=True, exist_ok=True)

    xx, tt = np.meshgrid(solver.x, solver.t, indexing="xy")
    xt = torch.tensor(np.column_stack([xx.reshape(-1), tt.reshape(-1)]), dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = model(xt).reshape(states_ref.shape)
    u_true = torch.tensor(states_ref, dtype=torch.float32, device=device)
    rel_err = relative_l2(u_pred, u_true)
    print(f"[ks forward] relative L2 error: {rel_err:.3e}")

    u_pred_np = u_pred.detach().cpu().numpy()
    save_named_columns_csv(
        outdir / "field_u.csv",
        flatten_field_to_rows(
            xx,
            tt,
            {
                "u_true": states_ref,
                "u_pred": u_pred_np,
                "abs_error": np.abs(u_pred_np - states_ref),
            },
        ),
    )
    save_named_columns_csv(
        outdir / "loss_history.csv",
        {
            "epoch": np.arange(1, len(history.total) + 1, dtype=np.int32),
            "loss_total": np.array(history.total, dtype=np.float64),
            "loss_pde": np.array(history.pde, dtype=np.float64),
            "loss_bc": np.array(history.bc, dtype=np.float64),
            "loss_ic": np.array(history.ic, dtype=np.float64),
        },
    )
    save_named_columns_csv(
        outdir / "test_error_history.csv",
        {
            "epoch": np.array(rel_epochs, dtype=np.int32),
            "rel_l2": np.array(rel_values, dtype=np.float64),
        },
    )
    save_named_columns_csv(outdir / "summary.csv", {"rel_l2_final": np.array([rel_err], dtype=np.float64)})


def main() -> None:
    args = make_parser().parse_args()
    model, history, rel_epochs, rel_values, solver, states_ref = train(args)
    evaluate_and_save(model, history, rel_epochs, rel_values, solver, states_ref, args)


if __name__ == "__main__":
    main()
