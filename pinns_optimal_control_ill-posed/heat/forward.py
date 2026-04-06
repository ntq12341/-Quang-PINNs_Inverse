from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinns_optimal_control.common import set_seed

from core import (
    NormalizedMLP,
    TrainHistory,
    analytic_heat,
    analytic_initial_condition,
    flatten_field_to_rows,
    grid_l2_error,
    heat_residual,
    lhs_2d,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN forward solver for the heat equation example.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--L", type=float, default=math.pi)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epochs", type=int, nargs="*", default=[3000])
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--batch-residual", type=int, default=1000)
    parser.add_argument("--n-boundary-time", type=int, default=40)
    parser.add_argument("--n-initial", type=int, default=40)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-nx", type=int, default=100)
    parser.add_argument("--eval-nt", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control_ill-posed/outputs/heat/forward")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, TrainHistory, list[int], list[float]]:
    set_seed(args.seed)
    device = torch.device(args.device)

    xt_r = lhs_2d(args.n_residual, low=(0.0, 0.0), high=(args.L, args.T), seed=args.seed)
    xt_r_t = as_tensor(xt_r, device=device, requires_grad=True)

    t_b = np.linspace(0.0, args.T, args.n_boundary_time, dtype=np.float32).reshape(-1, 1)
    xt_left = np.concatenate([np.zeros_like(t_b), t_b], axis=1)
    xt_right = np.concatenate([args.L * np.ones_like(t_b), t_b], axis=1)

    x0 = np.linspace(0.0, args.L, args.n_initial + 2, dtype=np.float32)[1:-1].reshape(-1, 1)
    xt0 = np.concatenate([x0, np.zeros_like(x0)], axis=1)

    left_t = as_tensor(xt_left, device=device, requires_grad=False)
    right_t = as_tensor(xt_right, device=device, requires_grad=False)
    init_t = as_tensor(xt0, device=device, requires_grad=False)

    mean = xt_r_t.detach().mean(dim=0)
    std = xt_r_t.detach().std(dim=0) + 1e-6
    model = NormalizedMLP(2, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = TrainHistory(total=[], pde=[], bc=[], ic=[], obj=[])

    x_eval = torch.linspace(0.0, args.L, args.eval_nx, device=device)
    t_eval = torch.linspace(0.0, args.T, args.eval_nt, device=device)
    xx_eval, tt_eval = torch.meshgrid(x_eval, t_eval, indexing="xy")
    xt_eval = torch.stack([xx_eval.reshape(-1), tt_eval.reshape(-1)], dim=1)
    rel_epochs: list[int] = []
    rel_values: list[float] = []

    n_batches = max(1, args.n_residual // args.batch_residual)
    lr_drop_epochs = set(args.lr_drop_epochs)
    zero_forcing = None
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
            if zero_forcing is None or zero_forcing.shape != u_r.shape:
                zero_forcing = torch.zeros_like(u_r)
            pde_loss = torch.mean(heat_residual(u_r, xt_batch, zero_forcing) ** 2)

            u_left = model(left_t)
            u_right = model(right_t)
            bc_loss = torch.mean(u_left**2) + torch.mean(u_right**2)

            u_init = model(init_t)
            ic_loss = torch.mean((u_init - analytic_initial_condition(init_t[:, :1])) ** 2)

            loss = pde_loss + bc_loss + ic_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pde_accum += pde_loss.item()

        pde_mean = pde_accum / n_batches
        with torch.no_grad():
            bc_loss = torch.mean(model(left_t) ** 2) + torch.mean(model(right_t) ** 2)
            ic_loss = torch.mean((model(init_t) - analytic_initial_condition(init_t[:, :1])) ** 2)
        total = pde_mean + bc_loss.item() + ic_loss.item()

        history.total.append(total)
        history.pde.append(pde_mean)
        history.bc.append(bc_loss.item())
        history.ic.append(ic_loss.item())
        history.obj.append(0.0)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                u_pred_eval = model(xt_eval).reshape(xx_eval.shape)
                u_true_eval = analytic_heat(xx_eval, tt_eval)
            rel_epochs.append(epoch)
            rel_values.append(relative_l2(u_pred_eval, u_true_eval))

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[heat forward] epoch={epoch:5d} total={total:.3e} "
                f"pde={pde_mean:.3e} bc={bc_loss.item():.3e} ic={ic_loss.item():.3e}"
            )

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

    x = torch.linspace(0.0, args.L, args.eval_nx, device=device)
    t = torch.linspace(0.0, args.T, args.eval_nt, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="xy")
    xt = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)

    with torch.no_grad():
        u_pred = model(xt).reshape(xx.shape)
        u_true = analytic_heat(xx, tt)
    rel_err = relative_l2(u_pred, u_true)

    xx_np = xx.detach().cpu().numpy()
    tt_np = tt.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()
    l2_grid = grid_l2_error(u_pred_np, u_true_np, x_np, t_np)
    print(f"[heat forward] relative L2 error: {rel_err:.3e}")
    print(f"[heat forward] grid L2 error: {l2_grid:.3e}")

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
    save_named_columns_csv(
        outdir / "summary.csv",
        {
            "rel_l2_final": np.array([rel_err], dtype=np.float64),
            "grid_l2_error": np.array([l2_grid], dtype=np.float64),
        },
    )


def main() -> None:
    args = make_parser().parse_args()
    model, history, rel_epochs, rel_values = train(args)
    evaluate_and_save(model, history, rel_epochs, rel_values, args)


if __name__ == "__main__":
    main()
