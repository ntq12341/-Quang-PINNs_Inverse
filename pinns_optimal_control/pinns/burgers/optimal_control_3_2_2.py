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
from pinns_optimal_control.pinns.burgers.core import (
    NormalizedMLP,
    TrainHistory,
    analytic_burgers,
    analytic_initial_condition,
    analytic_terminal_target,
    burgers_residual,
    flatten_field_to_rows,
    gradients,
    lhs_2d,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN optimal control for Burgers example (paper section 3.2.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--L", type=float, default=4.0)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epochs", type=int, nargs="*", default=[20000, 25000])
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=20000)
    parser.add_argument("--batch-residual", type=int, default=2000)
    parser.add_argument("--n-boundary-time", type=int, default=41)
    parser.add_argument("--n-initial", type=int, default=41)
    parser.add_argument("--n-objective", type=int, default=41)
    parser.add_argument("--wJ", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/burgers/optimal_control")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def midpoint_points(n: int, L: float) -> np.ndarray:
    dx = L / n
    return ((np.arange(n, dtype=np.float32) + 0.5) * dx).reshape(-1, 1)


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, NormalizedMLP, TrainHistory]:
    set_seed(args.seed)
    device = torch.device(args.device)

    xt_r = lhs_2d(args.n_residual, low=(0.0, 0.0), high=(args.L, args.T), seed=args.seed)
    xt_r_t = as_tensor(xt_r, device=device, requires_grad=True)
    x_r_t = xt_r_t[:, :1].detach()

    t_b = np.linspace(0.0, args.T, args.n_boundary_time, dtype=np.float32).reshape(-1, 1)
    xt_left = np.concatenate([np.zeros_like(t_b), t_b], axis=1)
    xt_right = np.concatenate([args.L * np.ones_like(t_b), t_b], axis=1)
    x0 = np.linspace(0.0, args.L, args.n_initial, dtype=np.float32).reshape(-1, 1)
    xt0 = np.concatenate([x0, np.zeros_like(x0)], axis=1)
    xj = midpoint_points(args.n_objective, args.L)
    xtT = np.concatenate([xj, args.T * np.ones_like(xj)], axis=1)

    left_t = as_tensor(xt_left, device=device, requires_grad=True)
    right_t = as_tensor(xt_right, device=device, requires_grad=True)
    init_t = as_tensor(xt0, device=device, requires_grad=False)
    x0_t = init_t[:, :1]
    term_t = as_tensor(xtT, device=device, requires_grad=False)
    xj_t = term_t[:, :1]

    mean_u = xt_r_t.detach().mean(dim=0)
    std_u = xt_r_t.detach().std(dim=0) + 1e-6
    mean_c = x_r_t.mean(dim=0)
    std_c = x_r_t.std(dim=0) + 1e-6
    u_net = NormalizedMLP(2, 1, mean_u, std_u, hidden_layers=4, hidden_width=50).to(device)
    u0_net = NormalizedMLP(1, 1, mean_c, std_c, hidden_layers=3, hidden_width=30).to(device)

    opt = torch.optim.Adam(list(u_net.parameters()) + list(u0_net.parameters()), lr=args.lr)
    history = TrainHistory(total=[], pde=[], bc=[], ic=[], obj=[])
    lr_drop_epochs = set(args.lr_drop_epochs)

    n_batches = max(1, args.n_residual // args.batch_residual)
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
            u_r = u_net(xt_batch)
            pde_loss = torch.mean(burgers_residual(u_r, xt_batch, args.nu) ** 2)

            u_left = u_net(left_t)
            u_right = u_net(right_t)
            ux_left = gradients(u_left, left_t, order=1)[:, :1]
            ux_right = gradients(u_right, right_t, order=1)[:, :1]
            bc_loss = torch.mean((u_left - u_right) ** 2) + torch.mean((ux_left - ux_right) ** 2)

            u_init = u_net(init_t)
            ic_loss = torch.mean((u_init - u0_net(x0_t)) ** 2)

            u_term = u_net(term_t)
            obj_loss = 0.5 * torch.mean((u_term - analytic_terminal_target(xj_t, T=args.T, nu=args.nu)) ** 2)

            loss = pde_loss + bc_loss + ic_loss + args.wJ * obj_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pde_accum += pde_loss.item()

        pde_mean = pde_accum / n_batches
        u_left = u_net(left_t)
        u_right = u_net(right_t)
        ux_left = gradients(u_left, left_t, order=1)[:, :1]
        ux_right = gradients(u_right, right_t, order=1)[:, :1]
        bc_loss = torch.mean((u_left - u_right) ** 2) + torch.mean((ux_left - ux_right) ** 2)
        with torch.no_grad():
            u_init = u_net(init_t)
            ic_loss = torch.mean((u_init - u0_net(x0_t)) ** 2)
            u_term = u_net(term_t)
            obj_loss = 0.5 * torch.mean((u_term - analytic_terminal_target(xj_t, T=args.T, nu=args.nu)) ** 2)
        total = pde_mean + bc_loss.item() + ic_loss.item() + args.wJ * obj_loss.item()

        history.total.append(total)
        history.pde.append(pde_mean)
        history.bc.append(bc_loss.item())
        history.ic.append(ic_loss.item())
        history.obj.append(obj_loss.item())

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[burgers control] epoch={epoch:5d} total={total:.3e} "
                f"pde={pde_mean:.3e} bc={bc_loss.item():.3e} ic={ic_loss.item():.3e} obj={obj_loss.item():.3e}"
            )

    return u_net, u0_net, history


def evaluate_and_save(u_net: NormalizedMLP, u0_net: NormalizedMLP, history: TrainHistory, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    outdir = Path(args.outdir) / "optimal_control_results"
    outdir.mkdir(parents=True, exist_ok=True)

    nx = 161
    nt = 101
    x = torch.linspace(0.0, args.L, nx, device=device)
    t = torch.linspace(0.0, args.T, nt, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="xy")
    xt = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
    with torch.no_grad():
        u_pred = u_net(xt).reshape(xx.shape)
        u_true = analytic_burgers(xx, tt, nu=args.nu)
        u0_pred = u0_net(x.reshape(-1, 1))
        u0_true = analytic_initial_condition(x.reshape(-1, 1), nu=args.nu)
        u_term = u_net(torch.stack([x, args.T * torch.ones_like(x)], dim=1))
        target_term = analytic_terminal_target(x.reshape(-1, 1), T=args.T, nu=args.nu)
    rel_u = relative_l2(u_pred, u_true)
    rel_u0 = relative_l2(u0_pred, u0_true)
    j_val = 0.5 * torch.mean((u_term - target_term) ** 2).item()
    print(f"[burgers control] relative L2 error for u(x,t): {rel_u:.3e}")
    print(f"[burgers control] relative L2 error for u0(x): {rel_u0:.3e}")
    print(f"[burgers control] objective J(u): {j_val:.3e}")

    xx_np = xx.detach().cpu().numpy()
    tt_np = tt.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    u0_pred_np = u0_pred.detach().cpu().numpy().reshape(-1)
    u0_true_np = u0_true.detach().cpu().numpy().reshape(-1)
    uT_pred_np = u_term.detach().cpu().numpy().reshape(-1)
    uT_target_np = target_term.detach().cpu().numpy().reshape(-1)

    save_named_columns_csv(
        outdir / "field_u.csv",
        flatten_field_to_rows(
            xx_np,
            tt_np,
            {
                "u_true": u_true_np,
                "u_pred": u_pred_np,
                "abs_error": np.abs(u_pred_np - u_true_np),
            },
        ),
    )
    save_named_columns_csv(
        outdir / "control_u0.csv",
        {
            "x": x_np,
            "u0_true": u0_true_np,
            "u0_pred": u0_pred_np,
            "abs_error": np.abs(u0_pred_np - u0_true_np),
        },
    )
    save_named_columns_csv(
        outdir / "terminal_state.csv",
        {
            "x": x_np,
            "uT_target": uT_target_np,
            "uT_pred": uT_pred_np,
            "abs_error": np.abs(uT_pred_np - uT_target_np),
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
            "rel_u": np.array([rel_u], dtype=np.float64),
            "rel_u0": np.array([rel_u0], dtype=np.float64),
            "objective": np.array([j_val], dtype=np.float64),
        },
    )


def main() -> None:
    args = make_parser().parse_args()
    u_net, u0_net, history = train(args)
    evaluate_and_save(u_net, u0_net, history, args)


if __name__ == "__main__":
    main()
