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
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN optimal control for Kuramoto-Sivashinsky example (paper section 3.3.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--L", type=float, default=50.0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epochs", type=int, nargs="*", default=[4000, 6500])
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=40000)
    parser.add_argument("--batch-residual", type=int, default=2000)
    parser.add_argument("--n-boundary-time", type=int, default=82)
    parser.add_argument("--n-initial", type=int, default=41)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--hidden-width", type=int, default=50)
    parser.add_argument("--wJ", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-nx", type=int, default=128)
    parser.add_argument("--eval-dt", type=float, default=0.01)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/ks/optimal_control")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def periodic_bc_loss(u_net: NormalizedMLP, left_t: torch.Tensor, right_t: torch.Tensor) -> torch.Tensor:
    u_left = u_net(left_t)
    u_right = u_net(right_t)
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


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, NormalizedMLP, TrainHistory]:
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
    u_net = NormalizedMLP(2, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    f_net = NormalizedMLP(2, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    opt = torch.optim.Adam(list(u_net.parameters()) + list(f_net.parameters()), lr=args.lr)
    history = TrainHistory(total=[], pde=[], bc=[], ic=[], obj=[])

    n_batches = max(1, args.n_residual // args.batch_residual)
    lr_drop_epochs = set(args.lr_drop_epochs)
    domain_scale = args.L * args.T

    for epoch in range(1, args.epochs + 1):
        if epoch in lr_drop_epochs:
            for group in opt.param_groups:
                group["lr"] *= args.lr_drop_factor

        perm = torch.randperm(args.n_residual, device=device)
        pde_accum = 0.0
        obj_accum = 0.0
        for k in range(n_batches):
            ids = perm[k * args.batch_residual : (k + 1) * args.batch_residual]
            xt_batch = xt_r_t[ids]
            xt_batch.requires_grad_(True)
            f_r = f_net(xt_batch)
            u_r = u_net(xt_batch)
            pde_loss = torch.mean(ks_residual(u_r, xt_batch, f_r) ** 2)
            bc_loss = periodic_bc_loss(u_net, left_t, right_t)
            ic_loss = torch.mean((u_net(init_t) - ks_initial_condition_torch(init_t[:, :1], L=args.L)) ** 2)
            obj_loss = 0.5 * domain_scale * torch.mean(u_r**2 + args.sigma * f_r**2)

            loss = pde_loss + bc_loss + ic_loss + args.wJ * obj_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pde_accum += pde_loss.item()
            obj_accum += obj_loss.item()

        pde_mean = pde_accum / n_batches
        obj_mean = obj_accum / n_batches
        bc_val = periodic_bc_loss(u_net, left_t, right_t)
        with torch.no_grad():
            ic_val = torch.mean((u_net(init_t) - ks_initial_condition_torch(init_t[:, :1], L=args.L)) ** 2)
        total = pde_mean + bc_val.item() + ic_val.item() + args.wJ * obj_mean
        history.total.append(total)
        history.pde.append(pde_mean)
        history.bc.append(bc_val.item())
        history.ic.append(ic_val.item())
        history.obj.append(obj_mean)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[ks control] epoch={epoch:5d} total={total:.3e} "
                f"pde={pde_mean:.3e} bc={bc_val.item():.3e} ic={ic_val.item():.3e} obj={obj_mean:.3e}"
            )

    return u_net, f_net, history


def evaluate_and_save(
    u_net: NormalizedMLP,
    f_net: NormalizedMLP,
    history: TrainHistory,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    outdir = Path(args.outdir) / "optimal_control_results"
    outdir.mkdir(parents=True, exist_ok=True)

    solver = KSSpectralSolver(nx=args.eval_nx, L=args.L, T=args.T, dt=args.eval_dt, sigma=args.sigma)
    xx, tt = np.meshgrid(solver.x, solver.t, indexing="xy")
    xt_np = np.column_stack([xx.reshape(-1), tt.reshape(-1)])
    xt_t = torch.tensor(xt_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pinn = u_net(xt_t).reshape(xx.shape).detach().cpu().numpy()
        f_all = f_net(xt_t).reshape(xx.shape).detach().cpu().numpy()

    forcing = f_all[:-1]
    states_rollout = solver.rollout(forcing=forcing)
    j_rollout = solver.objective(states_rollout, forcing)
    j_pinn = solver.objective(u_pinn, forcing)
    final_norm = float(np.sqrt(solver.dx * np.sum(states_rollout[-1] ** 2)))
    mismatch = float(np.linalg.norm(u_pinn - states_rollout) / (np.linalg.norm(states_rollout) + 1e-12))

    print(f"[ks control] spectral objective J(u,f): {j_rollout:.3e}")
    print(f"[ks control] rollout final-state L2 norm: {final_norm:.3e}")
    print(f"[ks control] relative mismatch PINN-vs-rollout: {mismatch:.3e}")

    save_named_columns_csv(
        outdir / "field_u.csv",
        flatten_field_to_rows(
            xx,
            tt,
            {
                "u_pinn": u_pinn,
                "u_rollout": states_rollout,
                "abs_error": np.abs(u_pinn - states_rollout),
            },
        ),
    )
    save_named_columns_csv(
        outdir / "control_f.csv",
        flatten_field_to_rows(
            xx[:-1],
            tt[:-1],
            {
                "f": forcing,
            },
        ),
    )
    save_named_columns_csv(
        outdir / "terminal_state.csv",
        {
            "x": solver.x,
            "uT_pinn": u_pinn[-1],
            "uT_rollout": states_rollout[-1],
            "abs_error": np.abs(u_pinn[-1] - states_rollout[-1]),
        },
    )
    save_named_columns_csv(
        outdir / "loss_history.csv",
        {
            "epoch": np.arange(1, len(history.total) + 1, dtype=np.int32),
            "loss_total": np.array(history.total, dtype=np.float64),
            "loss_pde": np.array(history.pde, dtype=np.float64),
            "loss_bc": np.array(history.bc, dtype=np.float64),
            "loss_ic": np.array(history.ic, dtype=np.float64),
            "loss_obj": np.array(history.obj, dtype=np.float64),
        },
    )
    save_named_columns_csv(
        outdir / "summary.csv",
        {
            "wJ": np.array([args.wJ], dtype=np.float64),
            "objective_pinn_grid": np.array([j_pinn], dtype=np.float64),
            "objective_rollout": np.array([j_rollout], dtype=np.float64),
            "final_state_l2": np.array([final_norm], dtype=np.float64),
            "rel_state_mismatch": np.array([mismatch], dtype=np.float64),
        },
    )


def main() -> None:
    args = make_parser().parse_args()
    u_net, f_net, history = train(args)
    evaluate_and_save(u_net, f_net, history, args)


if __name__ == "__main__":
    main()
