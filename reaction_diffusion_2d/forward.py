from __future__ import annotations

import argparse
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
    analytic_initial_condition,
    analytic_reaction_diffusion,
    flatten_field_to_rows,
    grid_l2_error_3d,
    lhs_nd,
    reaction_diffusion_residual,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN forward solver for the 2D reaction-diffusion example 5.3.1.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epochs", type=int, nargs="*", default=[5000, 10000])
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=20000)
    parser.add_argument("--batch-residual", type=int, default=2000)
    parser.add_argument("--n-boundary", type=int, default=400)
    parser.add_argument("--n-initial", type=int, default=200)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-nx", type=int, default=50)
    parser.add_argument("--eval-ny", type=int, default=50)
    parser.add_argument("--eval-nt", type=int, default=40)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d/forward")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def _make_boundary_points(args: argparse.Namespace) -> np.ndarray:
    n_edge = max(1, args.n_boundary // 4)
    s = np.linspace(0.0, 1.0, n_edge, dtype=np.float32).reshape(-1, 1)
    t = np.linspace(0.0, args.T, n_edge, dtype=np.float32).reshape(-1, 1)
    bottom = np.concatenate([s, np.zeros_like(s), t], axis=1)
    top = np.concatenate([s, np.ones_like(s), t], axis=1)
    left = np.concatenate([np.zeros_like(s), s, t], axis=1)
    right = np.concatenate([np.ones_like(s), s, t], axis=1)
    return np.vstack([bottom, top, left, right])


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, TrainHistory, list[int], list[float]]:
    set_seed(args.seed)
    device = torch.device(args.device)

    xyt_r = lhs_nd(args.n_residual, low=(0.0, 0.0, 0.0), high=(1.0, 1.0, args.T), seed=args.seed)
    xyt_r_t = as_tensor(xyt_r, device=device, requires_grad=True)
    xyt_b_t = as_tensor(_make_boundary_points(args), device=device)
    xy0 = lhs_nd(args.n_initial, low=(0.0, 0.0), high=(1.0, 1.0), seed=args.seed + 1)
    xyt0 = np.column_stack([xy0, np.zeros(args.n_initial, dtype=np.float64)])
    xyt0_t = as_tensor(xyt0, device=device)

    mean = xyt_r_t.detach().mean(dim=0)
    std = xyt_r_t.detach().std(dim=0) + 1e-6
    model = NormalizedMLP(3, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = TrainHistory(total=[], pde=[], bc=[], ic=[], obj=[])

    x_eval = torch.linspace(0.0, 1.0, args.eval_nx, device=device)
    y_eval = torch.linspace(0.0, 1.0, args.eval_ny, device=device)
    t_eval = torch.linspace(0.0, args.T, args.eval_nt, device=device)
    yy_eval, xx_eval, tt_eval = torch.meshgrid(y_eval, x_eval, t_eval, indexing="ij")
    xyt_eval = torch.stack([xx_eval.reshape(-1), yy_eval.reshape(-1), tt_eval.reshape(-1)], dim=1)
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
            xyt_batch = xyt_r_t[ids]
            xyt_batch.requires_grad_(True)
            u_r = model(xyt_batch)
            pde_loss = torch.mean(reaction_diffusion_residual(u_r, xyt_batch, torch.zeros_like(u_r), args.c) ** 2)
            bc_loss = torch.mean(model(xyt_b_t) ** 2)
            u_init = model(xyt0_t)
            ic_target = analytic_initial_condition(xyt0_t[:, :1], xyt0_t[:, 1:2])
            ic_loss = torch.mean((u_init - ic_target) ** 2)
            loss = pde_loss + bc_loss + ic_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            pde_accum += pde_loss.item()

        pde_mean = pde_accum / n_batches
        with torch.no_grad():
            bc_loss = torch.mean(model(xyt_b_t) ** 2)
            ic_loss = torch.mean((model(xyt0_t) - analytic_initial_condition(xyt0_t[:, :1], xyt0_t[:, 1:2])) ** 2)
        total = pde_mean + bc_loss.item() + ic_loss.item()
        history.total.append(total)
        history.pde.append(pde_mean)
        history.bc.append(bc_loss.item())
        history.ic.append(ic_loss.item())
        history.obj.append(0.0)

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            with torch.no_grad():
                u_pred_eval = model(xyt_eval).reshape(xx_eval.shape)
                u_true_eval = analytic_reaction_diffusion(xx_eval, yy_eval, tt_eval, c=args.c)
            rel_epochs.append(epoch)
            rel_values.append(relative_l2(u_pred_eval, u_true_eval))

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[rd2d forward] epoch={epoch:5d} total={total:.3e} "
                f"pde={pde_mean:.3e} bc={bc_loss.item():.3e} ic={ic_loss.item():.3e}"
            )

    return model, history, rel_epochs, rel_values


def evaluate_and_save(model: NormalizedMLP, history: TrainHistory, rel_epochs: list[int], rel_values: list[float], args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    outdir = Path(args.outdir) / "forward_results"
    outdir.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(0.0, 1.0, args.eval_nx, device=device)
    y = torch.linspace(0.0, 1.0, args.eval_ny, device=device)
    t = torch.linspace(0.0, args.T, args.eval_nt, device=device)
    yy, xx, tt = torch.meshgrid(y, x, t, indexing="ij")
    xyt = torch.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], dim=1)
    with torch.no_grad():
        u_pred = model(xyt).reshape(xx.shape)
        u_true = analytic_reaction_diffusion(xx, yy, tt, c=args.c)
    rel_err = relative_l2(u_pred, u_true)
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    u_true_np = u_true.detach().cpu().numpy()
    l2_grid = grid_l2_error_3d(u_pred_np, u_true_np, x_np, y_np, t_np)
    print(f"[rd2d forward] relative L2 error: {rel_err:.3e}")
    print(f"[rd2d forward] grid L2 error: {l2_grid:.3e}")

    save_named_columns_csv(outdir / "field_u.csv", flatten_field_to_rows(xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), tt.detach().cpu().numpy(), {"u_true": u_true_np, "u_pred": u_pred_np, "abs_error": np.abs(u_pred_np - u_true_np)}))
    save_named_columns_csv(outdir / "loss_history.csv", {"epoch": np.arange(1, len(history.total) + 1, dtype=np.int32), "loss_total": np.array(history.total), "loss_pde": np.array(history.pde), "loss_bc": np.array(history.bc), "loss_ic": np.array(history.ic)})
    save_named_columns_csv(outdir / "test_error_history.csv", {"epoch": np.array(rel_epochs, dtype=np.int32), "rel_l2": np.array(rel_values)})
    save_named_columns_csv(outdir / "summary.csv", {"c": np.array([args.c]), "rel_l2_final": np.array([rel_err]), "grid_l2_error": np.array([l2_grid])})


def main() -> None:
    args = make_parser().parse_args()
    model, history, rel_epochs, rel_values = train(args)
    evaluate_and_save(model, history, rel_epochs, rel_values, args)


if __name__ == "__main__":
    main()
