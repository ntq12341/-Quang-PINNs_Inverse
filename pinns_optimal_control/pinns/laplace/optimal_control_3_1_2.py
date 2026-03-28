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
from pinns_optimal_control.pinns.laplace.core import (
    NormalizedMLP,
    TrainHistory,
    analytic_optimal_f,
    analytic_optimal_u,
    control_bottom_bc,
    desired_flux,
    flatten_field_to_rows,
    gradients,
    laplace_residual,
    lhs_2d,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN optimal control solver for Laplace example (paper section 3.1.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epoch", type=int, default=5000)
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--batch-residual", type=int, default=1000)
    parser.add_argument("--n-boundary", type=int, default=40)
    parser.add_argument("--n-objective", type=int, default=41)
    parser.add_argument("--wJ", type=float, default=100.0, help="Objective weight in the PINN loss.")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/laplace/optimal_control")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def midpoint_points(n: int) -> np.ndarray:
    dx = 1.0 / n
    x = (np.arange(n, dtype=np.float32) + 0.5) * dx
    return x.reshape(-1, 1)


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, NormalizedMLP, TrainHistory]:
    set_seed(args.seed)
    device = torch.device(args.device)

    xy_r = lhs_2d(args.n_residual, seed=args.seed)
    xy_r_t = as_tensor(xy_r, device=device, requires_grad=True)
    x_r_t = xy_r_t[:, :1].detach()

    x_b = np.linspace(0.0, 1.0, args.n_boundary, dtype=np.float32).reshape(-1, 1)
    y_b = np.linspace(0.0, 1.0, args.n_boundary, dtype=np.float32).reshape(-1, 1)
    xy_top = np.concatenate([x_b, np.ones_like(x_b)], axis=1)
    xy_bottom = np.concatenate([x_b, np.zeros_like(x_b)], axis=1)
    xy_left = np.concatenate([np.zeros_like(y_b), y_b], axis=1)
    xy_right = np.concatenate([np.ones_like(y_b), y_b], axis=1)
    x_obj = midpoint_points(args.n_objective)
    xy_obj = np.concatenate([x_obj, np.ones_like(x_obj)], axis=1)

    top_t = as_tensor(xy_top, device=device, requires_grad=False)
    bottom_t = as_tensor(xy_bottom, device=device, requires_grad=False)
    left_t = as_tensor(xy_left, device=device, requires_grad=True)
    right_t = as_tensor(xy_right, device=device, requires_grad=True)
    x_top_t = top_t[:, :1]
    x_obj_t = as_tensor(x_obj, device=device, requires_grad=False)
    obj_t = as_tensor(xy_obj, device=device, requires_grad=True)

    mean_u = xy_r_t.detach().mean(dim=0)
    std_u = xy_r_t.detach().std(dim=0) + 1e-6
    mean_f = x_r_t.mean(dim=0)
    std_f = x_r_t.std(dim=0) + 1e-6

    u_net = NormalizedMLP(
        in_dim=2,
        out_dim=1,
        mean=mean_u,
        std=std_u,
        hidden_layers=4,
        hidden_width=50,
    ).to(device)
    f_net = NormalizedMLP(
        in_dim=1,
        out_dim=1,
        mean=mean_f,
        std=std_f,
        hidden_layers=3,
        hidden_width=30,
    ).to(device)

    opt = torch.optim.Adam(list(u_net.parameters()) + list(f_net.parameters()), lr=args.lr)
    history = TrainHistory(total=[], pde=[], bc=[], obj=[])

    n_batches = max(1, args.n_residual // args.batch_residual)
    for epoch in range(1, args.epochs + 1):
        if epoch == args.lr_drop_epoch:
            for g in opt.param_groups:
                g["lr"] = g["lr"] * args.lr_drop_factor

        perm = torch.randperm(args.n_residual, device=device)
        pde_accum = 0.0
        for k in range(n_batches):
            ids = perm[k * args.batch_residual : (k + 1) * args.batch_residual]
            xy_batch = xy_r_t[ids]
            xy_batch.requires_grad_(True)
            u_r = u_net(xy_batch)
            r = laplace_residual(u_r, xy_batch)
            pde_loss = torch.mean(r**2)

            u_top = u_net(top_t)
            u_bottom = u_net(bottom_t)
            u_left = u_net(left_t)
            u_right = u_net(right_t)
            ux_left = gradients(u_left, left_t, order=1)[:, :1]
            ux_right = gradients(u_right, right_t, order=1)[:, :1]
            bc_loss = (
                torch.mean((u_bottom - control_bottom_bc(bottom_t[:, :1])) ** 2)
                + torch.mean((u_top - f_net(x_top_t)) ** 2)
                + torch.mean((u_left - u_right) ** 2)
                + torch.mean((ux_left - ux_right) ** 2)
            )

            u_obj = u_net(obj_t)
            uy_obj = gradients(u_obj, obj_t, order=1)[:, 1:2]
            j_loss = torch.mean((uy_obj - desired_flux(x_obj_t)) ** 2)

            loss = pde_loss + bc_loss + args.wJ * j_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pde_accum += pde_loss.item()

        pde_mean = pde_accum / n_batches
        with torch.no_grad():
            u_top = u_net(top_t)
            u_bottom = u_net(bottom_t)
            u_left = u_net(left_t)
            u_right = u_net(right_t)
            bc_loss = (
                torch.mean((u_bottom - control_bottom_bc(bottom_t[:, :1])) ** 2)
                + torch.mean((u_top - f_net(x_top_t)) ** 2)
                + torch.mean((u_left - u_right) ** 2)
            )

        u_obj = u_net(obj_t)
        uy_obj = gradients(u_obj, obj_t, order=1)[:, 1:2]
        j_loss = torch.mean((uy_obj - desired_flux(x_obj_t)) ** 2)
        total = pde_mean + bc_loss.item() + args.wJ * j_loss.item()

        history.pde.append(pde_mean)
        history.bc.append(bc_loss.item())
        history.obj.append(j_loss.item())
        history.total.append(total)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[control] epoch={epoch:5d} total={total:.3e} "
                f"pde={pde_mean:.3e} bc={bc_loss.item():.3e} obj={j_loss.item():.3e}"
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

    n_eval = 100
    x = torch.linspace(0.0, 1.0, n_eval, device=device)
    y = torch.linspace(0.0, 1.0, n_eval, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    with torch.no_grad():
        u_pred = u_net(xy).reshape(n_eval, n_eval)
        u_true = analytic_optimal_u(xx, yy)
    rel_u = relative_l2(u_pred, u_true)

    x_line = x.reshape(-1, 1)
    with torch.no_grad():
        f_pred = f_net(x_line)
        f_true = analytic_optimal_f(x_line)
        f_rel = relative_l2(f_pred, f_true)

    x_obj = midpoint_points(args.n_objective)
    obj_t = as_tensor(np.concatenate([x_obj, np.ones_like(x_obj)], axis=1), device=device, requires_grad=True)
    x_obj_t = as_tensor(x_obj, device=device, requires_grad=False)
    u_obj = u_net(obj_t)
    uy_obj = gradients(u_obj, obj_t, order=1)[:, 1:2]
    j_est = torch.mean((uy_obj - desired_flux(x_obj_t)) ** 2).item()

    print(f"[control] relative L2 error for u*: {rel_u:.3e}")
    print(f"[control] relative L2 error for f*: {f_rel:.3e}")
    print(f"[control] objective J(u): {j_est:.3e}")

    x_np = xx.detach().cpu().numpy()
    y_np = yy.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()
    x_f_np = x_line.detach().cpu().numpy().reshape(-1)
    f_pred_np = f_pred.detach().cpu().numpy().reshape(-1)
    f_true_np = f_true.detach().cpu().numpy().reshape(-1)

    save_named_columns_csv(
        outdir / "field_u.csv",
        flatten_field_to_rows(
            x_np,
            y_np,
            {
                "u_true": u_true_np,
                "u_pred": u_pred_np,
                "abs_error": np.abs(u_pred_np - u_true_np),
            },
        ),
    )
    save_named_columns_csv(
        outdir / "control_f.csv",
        {
            "x": x_f_np,
            "f_true": f_true_np,
            "f_pred": f_pred_np,
            "abs_error": np.abs(f_pred_np - f_true_np),
        },
    )
    save_named_columns_csv(
        outdir / "loss_history.csv",
        {
            "epoch": np.arange(1, len(history.total) + 1, dtype=np.int32),
            "loss_total": np.array(history.total, dtype=np.float64),
            "loss_pde": np.array(history.pde, dtype=np.float64),
            "loss_bc": np.array(history.bc, dtype=np.float64),
            "loss_obj": np.array(history.obj, dtype=np.float64),
        },
    )
    save_named_columns_csv(
        outdir / "summary.csv",
        {
            "wJ": np.array([args.wJ], dtype=np.float64),
            "rel_u": np.array([rel_u], dtype=np.float64),
            "rel_f": np.array([f_rel], dtype=np.float64),
            "objective": np.array([j_est], dtype=np.float64),
        },
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    u_net, f_net, history = train(args)
    evaluate_and_save(u_net, f_net, history, args)


if __name__ == "__main__":
    main()
