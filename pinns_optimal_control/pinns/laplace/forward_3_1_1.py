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
    analytic_forward_u,
    flatten_field_to_rows,
    laplace_residual,
    lhs_2d,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN forward solver for Laplace example (paper section 3.1.1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epoch", type=int, default=3000)
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--batch-residual", type=int, default=1000)
    parser.add_argument("--n-boundary-edge", type=int, default=40)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20, help="Compute relative L2 test error every N epochs.")
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control/outputs/laplace/forward")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, TrainHistory, list[int], list[float]]:
    set_seed(args.seed)
    device = torch.device(args.device)

    xy_r = lhs_2d(args.n_residual, seed=args.seed)
    xy_r_t = as_tensor(xy_r, device=device, requires_grad=True)

    x_edge = np.linspace(0.0, 1.0, args.n_boundary_edge, dtype=np.float32).reshape(-1, 1)
    y_edge = np.linspace(0.0, 1.0, args.n_boundary_edge, dtype=np.float32).reshape(-1, 1)

    xy_top = np.concatenate([x_edge, np.ones_like(x_edge)], axis=1)
    xy_bottom = np.concatenate([x_edge, np.zeros_like(x_edge)], axis=1)
    xy_left = np.concatenate([np.zeros_like(y_edge), y_edge], axis=1)
    xy_right = np.concatenate([np.ones_like(y_edge), y_edge], axis=1)

    top_t = as_tensor(xy_top, device=device, requires_grad=False)
    bottom_t = as_tensor(xy_bottom, device=device, requires_grad=False)
    left_t = as_tensor(xy_left, device=device, requires_grad=False)
    right_t = as_tensor(xy_right, device=device, requires_grad=False)

    mean = xy_r_t.detach().mean(dim=0)
    std = xy_r_t.detach().std(dim=0) + 1e-6
    model = NormalizedMLP(
        in_dim=2,
        out_dim=1,
        mean=mean,
        std=std,
        hidden_layers=4,
        hidden_width=50,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = TrainHistory(total=[], pde=[], bc=[], obj=[])
    rel_epochs: list[int] = []
    rel_values: list[float] = []
    n_eval = 100
    x_eval = torch.linspace(0.0, 1.0, n_eval, device=device)
    y_eval = torch.linspace(0.0, 1.0, n_eval, device=device)
    xx_eval, yy_eval = torch.meshgrid(x_eval, y_eval, indexing="xy")
    xy_eval = torch.stack([xx_eval.reshape(-1), yy_eval.reshape(-1)], dim=1)

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
            u_r = model(xy_batch)
            r = laplace_residual(u_r, xy_batch)
            pde_loss = torch.mean(r**2)

            u_top = model(top_t)
            u_bottom = model(bottom_t)
            u_left = model(left_t)
            u_right = model(right_t)
            bc_loss = (
                torch.mean((u_top - torch.sin(torch.pi * top_t[:, :1])) ** 2)
                + torch.mean(u_bottom**2)
                + torch.mean(u_left**2)
                + torch.mean(u_right**2)
            )

            loss = pde_loss + bc_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pde_accum += pde_loss.item()

        pde_mean = pde_accum / n_batches
        with torch.no_grad():
            u_top = model(top_t)
            u_bottom = model(bottom_t)
            u_left = model(left_t)
            u_right = model(right_t)
            bc_loss = (
                torch.mean((u_top - torch.sin(torch.pi * top_t[:, :1])) ** 2)
                + torch.mean(u_bottom**2)
                + torch.mean(u_left**2)
                + torch.mean(u_right**2)
            )
            total = pde_mean + bc_loss.item()

        history.pde.append(pde_mean)
        history.bc.append(bc_loss.item())
        history.obj.append(0.0)
        history.total.append(total)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                u_pred_eval = model(xy_eval).reshape(n_eval, n_eval)
                u_true_eval = analytic_forward_u(xx_eval, yy_eval)
            rel_values.append(relative_l2(u_pred_eval, u_true_eval))
            rel_epochs.append(epoch)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[forward] epoch={epoch:5d} total={total:.3e} pde={pde_mean:.3e} bc={bc_loss.item():.3e}")

    return model, history, rel_epochs, rel_values


def evaluate_and_save(
    model: NormalizedMLP,
    history: TrainHistory,
    rel_epochs: list[int],
    rel_values: list[float],
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    outdir = Path(args.outdir) / "forward_results"
    outdir.mkdir(parents=True, exist_ok=True)

    n_eval = 100
    x = torch.linspace(0.0, 1.0, n_eval, device=device)
    y = torch.linspace(0.0, 1.0, n_eval, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        u_pred = model(xy).reshape(n_eval, n_eval)
        u_true = analytic_forward_u(xx, yy)
    rel_err = relative_l2(u_pred, u_true)
    print(f"[forward] relative L2 error (100x100 grid): {rel_err:.3e}")

    x_np = xx.detach().cpu().numpy()
    y_np = yy.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()
    abs_err_np = np.abs(u_pred_np - u_true_np)

    save_named_columns_csv(
        outdir / "field_u.csv",
        flatten_field_to_rows(
            x_np,
            y_np,
            {
                "u_true": u_true_np,
                "u_pred": u_pred_np,
                "abs_error": abs_err_np,
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
        },
    )
    save_named_columns_csv(
        outdir / "test_error_history.csv",
        {
            "epoch": np.array(rel_epochs, dtype=np.int32),
            "rel_l2": np.array(rel_values, dtype=np.float64),
        },
    )
    save_named_columns_csv(
        outdir / "summary.csv",
        {
            "rel_l2_final": np.array([rel_err], dtype=np.float64),
        },
    )


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    model, history, rel_epochs, rel_values = train(args)
    evaluate_and_save(model, history, rel_epochs, rel_values, args)


if __name__ == "__main__":
    main()
