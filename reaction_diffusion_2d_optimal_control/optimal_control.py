from __future__ import annotations

import argparse
import copy
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
    OptimalControlHistory,
    ReactionDiffusion2DForwardSolver,
    analytic_initial_condition,
    analytic_min_energy_control,
    analytic_reaction_diffusion,
    analytic_terminal_target,
    flatten_field_to_rows,
    gradients,
    lhs_nd,
    reaction_diffusion_residual,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN optimal-control solver for the 2D reaction-diffusion example 5.3.2.")
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
    parser.add_argument("--n-terminal", type=int, default=400)
    parser.add_argument("--n-tikhonov", type=int, default=400)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--wJ", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--alpha-method", choices=["fixed", "lcurve"], default="lcurve")
    parser.add_argument("--alpha-list", type=float, nargs="*", default=None)
    parser.add_argument("--alpha-scan-epochs", type=int, default=2000)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-nx", type=int, default=45)
    parser.add_argument("--eval-ny", type=int, default=45)
    parser.add_argument("--eval-nt", type=int, default=35)
    parser.add_argument("--rollout-nx", type=int, default=31)
    parser.add_argument("--rollout-ny", type=int, default=31)
    parser.add_argument("--rollout-nt", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d_optimal_control")
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


def _build_training_sets(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    xyt_r = lhs_nd(args.n_residual, low=(0.0, 0.0, 0.0), high=(1.0, 1.0, args.T), seed=args.seed)
    tik_seed = None if args.seed is None else args.seed + 1
    term_seed = None if args.seed is None else args.seed + 2
    init_seed = None if args.seed is None else args.seed + 3
    xy_tik = lhs_nd(args.n_tikhonov, low=(0.0, 0.0, 0.0), high=(1.0, 1.0, args.T), seed=tik_seed)
    xy_term = lhs_nd(args.n_terminal, low=(0.0, 0.0), high=(1.0, 1.0), seed=term_seed)
    xy_init = lhs_nd(args.n_initial, low=(0.0, 0.0), high=(1.0, 1.0), seed=init_seed)
    xyt0 = np.column_stack([xy_init, np.zeros(args.n_initial, dtype=np.float64)])
    xyt_terminal = np.column_stack([xy_term, args.T * np.ones(args.n_terminal, dtype=np.float64)])
    return {
        "xyt_r": as_tensor(xyt_r, device=device, requires_grad=True),
        "xyt_tik": as_tensor(xy_tik, device=device, requires_grad=True),
        "boundary": as_tensor(_make_boundary_points(args), device=device),
        "init": as_tensor(xyt0, device=device),
        "terminal": as_tensor(xyt_terminal, device=device),
    }


def _build_networks(args: argparse.Namespace, xyt_r: torch.Tensor, device: torch.device) -> tuple[NormalizedMLP, NormalizedMLP]:
    mean = xyt_r.detach().mean(dim=0)
    std = xyt_r.detach().std(dim=0) + 1e-6
    u_net = NormalizedMLP(3, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    f_net = NormalizedMLP(3, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    return u_net, f_net


def _compute_batch_losses(
    u_net: NormalizedMLP,
    f_net: NormalizedMLP,
    xyt_batch: torch.Tensor,
    xyt_tik: torch.Tensor,
    boundary_t: torch.Tensor,
    init_t: torch.Tensor,
    terminal_t: torch.Tensor,
    args: argparse.Namespace,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    u_r = u_net(xyt_batch)
    f_r = f_net(xyt_batch)
    pde_loss = torch.mean(reaction_diffusion_residual(u_r, xyt_batch, f_r, args.c) ** 2)
    bc_loss = torch.mean(u_net(boundary_t) ** 2)
    ic_target = analytic_initial_condition(init_t[:, :1], init_t[:, 1:2])
    ic_loss = torch.mean((u_net(init_t) - ic_target) ** 2)
    u_terminal = u_net(terminal_t)
    target = analytic_terminal_target(terminal_t[:, :1], terminal_t[:, 1:2], c=args.c, T=args.T)
    objective_loss = 0.5 * torch.mean((u_terminal - target) ** 2)
    f_tik = f_net(xyt_tik)
    grad_f = gradients(f_tik, xyt_tik, order=1)
    f_x = grad_f[:, :1]
    f_y = grad_f[:, 1:2]
    f_t = grad_f[:, 2:3]
    reg_loss = 0.5 * torch.mean(f_tik**2 + f_x**2 + f_y**2 + f_t**2)
    total = pde_loss + bc_loss + ic_loss + args.wJ * objective_loss + alpha * reg_loss
    return total, {
        "pde": pde_loss,
        "bc": bc_loss,
        "ic": ic_loss,
        "objective": objective_loss,
        "regularization": reg_loss,
    }


def _train_once(args: argparse.Namespace, alpha: float, epochs: int, print_every: int) -> tuple[NormalizedMLP, NormalizedMLP, OptimalControlHistory]:
    set_seed(args.seed)
    device = torch.device(args.device)
    data = _build_training_sets(args, device)
    u_net, f_net = _build_networks(args, data["xyt_r"], device)
    opt = torch.optim.Adam(list(u_net.parameters()) + list(f_net.parameters()), lr=args.lr)
    history = OptimalControlHistory(total=[], pde=[], bc=[], ic=[], objective=[], regularization=[], alpha_scan=[])
    n_batches = max(1, args.n_residual // args.batch_residual)
    lr_drop_epochs = set(epoch for epoch in args.lr_drop_epochs if epoch <= epochs)
    for epoch in range(1, epochs + 1):
        if epoch in lr_drop_epochs:
            for group in opt.param_groups:
                group["lr"] *= args.lr_drop_factor
        perm = torch.randperm(args.n_residual, device=device)
        accum = {"pde": 0.0, "bc": 0.0, "ic": 0.0, "objective": 0.0, "regularization": 0.0}
        for k in range(n_batches):
            ids = perm[k * args.batch_residual : (k + 1) * args.batch_residual]
            xyt_batch = data["xyt_r"][ids]
            xyt_batch.requires_grad_(True)
            xyt_tik = data["xyt_tik"]
            xyt_tik.requires_grad_(True)
            total_loss, losses = _compute_batch_losses(u_net, f_net, xyt_batch, xyt_tik, data["boundary"], data["init"], data["terminal"], args, alpha)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            for key in accum:
                accum[key] += float(losses[key].detach().cpu().item())
        mean_losses = {key: accum[key] / n_batches for key in accum}
        total = mean_losses["pde"] + mean_losses["bc"] + mean_losses["ic"] + args.wJ * mean_losses["objective"] + alpha * mean_losses["regularization"]
        history.total.append(total)
        history.pde.append(mean_losses["pde"])
        history.bc.append(mean_losses["bc"])
        history.ic.append(mean_losses["ic"])
        history.objective.append(mean_losses["objective"])
        history.regularization.append(mean_losses["regularization"])
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[rd2d control] epoch={epoch:5d} total={total:.3e} "
                f"pde={mean_losses['pde']:.3e} bc={mean_losses['bc']:.3e} ic={mean_losses['ic']:.3e} "
                f"obj={mean_losses['objective']:.3e} reg={mean_losses['regularization']:.3e}"
            )
    return u_net, f_net, history


def l_curve_method(args: argparse.Namespace) -> tuple[float, list[dict[str, float]]]:
    alpha_values = args.alpha_list if args.alpha_list else list(np.logspace(-7, 0, 15))
    results: list[dict[str, float]] = []
    for alpha in alpha_values:
        run_args = copy.deepcopy(args)
        _, _, history = _train_once(run_args, float(alpha), args.alpha_scan_epochs, print_every=max(args.alpha_scan_epochs + 1, 10))
        residual_sq = history.pde[-1] + history.bc[-1] + history.ic[-1]
        residual = math.sqrt(max(residual_sq, 1e-16))
        reg = math.sqrt(max(history.regularization[-1], 1e-16))
        results.append({"alpha": float(alpha), "residual": residual, "regularization": reg})
        print(f"[rd2d alpha] alpha={alpha:.3e} residual={residual:.3e} reg={reg:.3e}")
    log_res = np.log(np.array([row["residual"] for row in results]))
    log_reg = np.log(np.array([row["regularization"] for row in results]))
    dx = np.gradient(log_res)
    dy = np.gradient(log_reg)
    d1 = dy / (dx + 1e-12)
    d2 = np.gradient(d1) / (dx + 1e-12)
    curvature = d2 / np.power(1.0 + d1**2, 1.5)
    score = np.abs(curvature)
    if len(score) >= 3:
        score[0] = -np.inf
        score[-1] = -np.inf
    idx = int(np.argmax(score))
    return results[idx]["alpha"], results


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, NormalizedMLP, OptimalControlHistory, float]:
    if args.alpha_method == "lcurve":
        alpha, alpha_scan = l_curve_method(args)
        print(f"[rd2d control] selected alpha by L-curve: {alpha:.3e}")
    else:
        alpha = float(args.alpha)
        alpha_scan = []
    u_net, f_net, history = _train_once(args, alpha, args.epochs, args.print_every)
    history.alpha_scan = alpha_scan
    return u_net, f_net, history, alpha


def evaluate_and_save(u_net: NormalizedMLP, f_net: NormalizedMLP, history: OptimalControlHistory, alpha: float, args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    outdir = Path(args.outdir) / "optimal_control_results"
    outdir.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(0.0, 1.0, args.eval_nx, device=device)
    y = torch.linspace(0.0, 1.0, args.eval_ny, device=device)
    t = torch.linspace(0.0, args.T, args.eval_nt, device=device)
    yy, xx, tt = torch.meshgrid(y, x, t, indexing="ij")
    xyt = torch.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], dim=1)
    with torch.no_grad():
        u_pinn = u_net(xyt).reshape(xx.shape)
        f_pinn = f_net(xyt).reshape(xx.shape)
        u_true = analytic_reaction_diffusion(xx, yy, tt, c=args.c)
        f_true = analytic_min_energy_control(xx, yy, tt)
    rel_u = relative_l2(u_pinn, u_true)
    f_l2 = torch.sqrt(torch.mean(f_pinn**2)).item()
    print(f"[rd2d control] relative L2 error for u*: {rel_u:.3e}")
    print(f"[rd2d control] L2 RMS error for f* = 0: {f_l2:.3e}")

    solver = ReactionDiffusion2DForwardSolver(nx=args.rollout_nx, ny=args.rollout_ny, nt=args.rollout_nt, c=args.c, T=args.T)
    xx_roll, yy_roll, tt_roll = np.meshgrid(solver.x, solver.y, solver.t, indexing="xy")
    xyt_roll = torch.tensor(np.column_stack([xx_roll.reshape(-1), yy_roll.reshape(-1), tt_roll.reshape(-1)]), dtype=torch.float32, device=device)
    with torch.no_grad():
        f_roll = f_net(xyt_roll).reshape(tt_roll.shape).detach().cpu().numpy()
    f_roll = np.moveaxis(f_roll, 2, 0)
    states_roll = solver.rollout(forcing=f_roll)
    xx2, yy2 = np.meshgrid(solver.x, solver.y, indexing="xy")
    target_roll = np.exp(-(args.c + 2.0 * math.pi**2) * args.T) * np.sin(math.pi * xx2) * np.sin(math.pi * yy2)
    objective_roll = solver.terminal_objective(states_roll[-1], target_roll)
    h1_roll = solver.h1_control_norm(f_roll)
    tikhonov_roll = objective_roll + alpha * h1_roll
    final_state_rel = float(np.linalg.norm(states_roll[-1] - target_roll) / (np.linalg.norm(target_roll) + 1e-12))
    print(f"[rd2d control] terminal objective J_roll: {objective_roll:.3e}")
    print(f"[rd2d control] H1 regularization term: {h1_roll:.3e}")

    x_term = torch.linspace(0.0, 1.0, args.eval_nx, device=device)
    y_term = torch.linspace(0.0, 1.0, args.eval_ny, device=device)
    yy_term, xx_term = torch.meshgrid(y_term, x_term, indexing="ij")
    xyt_term = torch.stack([xx_term.reshape(-1), yy_term.reshape(-1), args.T * torch.ones_like(xx_term).reshape(-1)], dim=1)
    with torch.no_grad():
        u_term_pinn = u_net(xyt_term).reshape(xx_term.shape)
        target_term = analytic_terminal_target(xx_term, yy_term, c=args.c, T=args.T)
    terminal_misfit = 0.5 * torch.mean((u_term_pinn - target_term) ** 2).item()

    save_named_columns_csv(outdir / "field_u.csv", flatten_field_to_rows(xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), tt.detach().cpu().numpy(), {"u_true": u_true.detach().cpu().numpy(), "u_pinn": u_pinn.detach().cpu().numpy(), "abs_error": np.abs(u_pinn.detach().cpu().numpy() - u_true.detach().cpu().numpy())}))
    save_named_columns_csv(outdir / "control_f.csv", flatten_field_to_rows(xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), tt.detach().cpu().numpy(), {"f_true": f_true.detach().cpu().numpy(), "f_pred": f_pinn.detach().cpu().numpy(), "abs_error": np.abs(f_pinn.detach().cpu().numpy())}))
    save_named_columns_csv(outdir / "terminal_state.csv", {"x": xx_term.detach().cpu().numpy().reshape(-1), "y": yy_term.detach().cpu().numpy().reshape(-1), "uT_target": target_term.detach().cpu().numpy().reshape(-1), "uT_pinn": u_term_pinn.detach().cpu().numpy().reshape(-1)})
    save_named_columns_csv(outdir / "rollout_terminal_state.csv", {"x": xx2.reshape(-1), "y": yy2.reshape(-1), "uT_target": target_roll.reshape(-1), "uT_rollout": states_roll[-1].reshape(-1)})
    save_named_columns_csv(outdir / "loss_history.csv", {"epoch": np.arange(1, len(history.total) + 1, dtype=np.int32), "loss_total": np.array(history.total), "loss_pde": np.array(history.pde), "loss_bc": np.array(history.bc), "loss_ic": np.array(history.ic), "loss_obj": np.array(history.objective), "loss_reg_h1": np.array(history.regularization)})
    if history.alpha_scan:
        save_named_columns_csv(outdir / "alpha_lcurve.csv", {"alpha": np.array([row["alpha"] for row in history.alpha_scan]), "residual": np.array([row["residual"] for row in history.alpha_scan]), "regularization": np.array([row["regularization"] for row in history.alpha_scan])})
    save_named_columns_csv(outdir / "summary.csv", {"c": np.array([args.c]), "wJ": np.array([args.wJ]), "alpha": np.array([alpha]), "rel_u": np.array([rel_u]), "f_l2_rms": np.array([f_l2]), "terminal_misfit_pinn": np.array([terminal_misfit]), "objective_rollout": np.array([objective_roll]), "h1_rollout": np.array([h1_roll]), "tikhonov_rollout": np.array([tikhonov_roll]), "final_state_rel": np.array([final_state_rel])})


def main() -> None:
    args = make_parser().parse_args()
    u_net, f_net, history, alpha = train(args)
    evaluate_and_save(u_net, f_net, history, alpha, args)


if __name__ == "__main__":
    main()
