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
    HeatForwardSolver,
    NormalizedMLP,
    OptimalControlHistory,
    analytic_initial_condition,
    analytic_optimal_f,
    analytic_optimal_u,
    analytic_terminal_target,
    flatten_field_to_rows,
    gradients,
    heat_residual,
    lhs_2d,
    relative_l2,
    save_named_columns_csv,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN optimal control solver for the heat equation example.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--L", type=float, default=math.pi)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epochs", type=int, nargs="*", default=[5000])
    parser.add_argument("--lr-drop-factor", type=float, default=0.1)
    parser.add_argument("--n-residual", type=int, default=10000)
    parser.add_argument("--batch-residual", type=int, default=1000)
    parser.add_argument("--n-boundary-time", type=int, default=40)
    parser.add_argument("--n-initial", type=int, default=40)
    parser.add_argument("--n-terminal", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=50)
    parser.add_argument("--wJ", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--alpha-method", choices=["fixed", "lcurve"], default="lcurve")
    parser.add_argument("--alpha-list", type=float, nargs="*", default=None)
    parser.add_argument("--alpha-scan-epochs", type=int, default=1500)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--eval-nx", type=int, default=100)
    parser.add_argument("--eval-nt", type=int, default=100)
    parser.add_argument("--rollout-nx", type=int, default=101)
    parser.add_argument("--rollout-nt", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="pinns_optimal_control_ill-posed/outputs/heat_optimal_control")
    return parser


def as_tensor(arr: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def _build_training_sets(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    xt_r = lhs_2d(args.n_residual, low=(0.0, 0.0), high=(args.L, args.T), seed=args.seed)
    xt_r_t = as_tensor(xt_r, device=device, requires_grad=True)
    t_b = np.linspace(0.0, args.T, args.n_boundary_time, dtype=np.float32).reshape(-1, 1)
    xt_left = np.concatenate([np.zeros_like(t_b), t_b], axis=1)
    xt_right = np.concatenate([args.L * np.ones_like(t_b), t_b], axis=1)
    x0 = np.linspace(0.0, args.L, args.n_initial + 2, dtype=np.float32)[1:-1].reshape(-1, 1)
    xt0 = np.concatenate([x0, np.zeros_like(x0)], axis=1)
    x_terminal = np.linspace(0.0, args.L, args.n_terminal, dtype=np.float32).reshape(-1, 1)
    xt_terminal = np.concatenate([x_terminal, args.T * np.ones_like(x_terminal)], axis=1)
    return {
        "xt_r": xt_r_t,
        "left": as_tensor(xt_left, device=device, requires_grad=False),
        "right": as_tensor(xt_right, device=device, requires_grad=False),
        "init": as_tensor(xt0, device=device, requires_grad=False),
        "terminal": as_tensor(xt_terminal, device=device, requires_grad=False),
    }


def _build_networks(args: argparse.Namespace, xt_r_t: torch.Tensor, device: torch.device) -> tuple[NormalizedMLP, NormalizedMLP]:
    mean = xt_r_t.detach().mean(dim=0)
    std = xt_r_t.detach().std(dim=0) + 1e-6
    u_net = NormalizedMLP(2, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    f_net = NormalizedMLP(2, 1, mean, std, hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)
    return u_net, f_net


def _compute_batch_losses(
    u_net: NormalizedMLP,
    f_net: NormalizedMLP,
    xt_batch: torch.Tensor,
    left_t: torch.Tensor,
    right_t: torch.Tensor,
    init_t: torch.Tensor,
    terminal_t: torch.Tensor,
    args: argparse.Namespace,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    u_r = u_net(xt_batch)
    f_r = f_net(xt_batch)
    pde_loss = torch.mean(heat_residual(u_r, xt_batch, f_r) ** 2)
    bc_loss = torch.mean(u_net(left_t) ** 2) + torch.mean(u_net(right_t) ** 2)
    ic_loss = torch.mean((u_net(init_t) - analytic_initial_condition(init_t[:, :1])) ** 2)
    u_terminal = u_net(terminal_t)
    terminal_target = analytic_terminal_target(terminal_t[:, :1])
    objective_loss = 0.5 * args.L * torch.mean((u_terminal - terminal_target) ** 2)
    grad_f = gradients(f_r, xt_batch, order=1)
    f_x = grad_f[:, :1]
    f_t = grad_f[:, 1:2]
    reg_loss = 0.5 * args.L * args.T * torch.mean(f_r**2 + f_x**2 + f_t**2)
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
    u_net, f_net = _build_networks(args, data["xt_r"], device)
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
            xt_batch = data["xt_r"][ids]
            xt_batch.requires_grad_(True)
            total_loss, losses = _compute_batch_losses(u_net, f_net, xt_batch, data["left"], data["right"], data["init"], data["terminal"], args, alpha)
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
                f"[heat control] epoch={epoch:5d} total={total:.3e} "
                f"pde={mean_losses['pde']:.3e} bc={mean_losses['bc']:.3e} ic={mean_losses['ic']:.3e} "
                f"obj={mean_losses['objective']:.3e} reg={mean_losses['regularization']:.3e}"
            )
    return u_net, f_net, history


def l_curve_method(args: argparse.Namespace) -> tuple[float, list[dict[str, float]]]:
    alpha_values = args.alpha_list if args.alpha_list else list(np.logspace(-6, 0, 9))
    results: list[dict[str, float]] = []
    for alpha in alpha_values:
        run_args = copy.deepcopy(args)
        _, _, history = _train_once(run_args, float(alpha), args.alpha_scan_epochs, print_every=max(args.alpha_scan_epochs + 1, 10))
        residual_sq = history.pde[-1] + history.bc[-1] + history.ic[-1] + history.objective[-1]
        residual = math.sqrt(max(residual_sq, 1e-16))
        reg = math.sqrt(max(history.regularization[-1], 1e-16))
        results.append({"alpha": float(alpha), "residual": residual, "regularization": reg})
        print(f"[heat alpha] alpha={alpha:.3e} residual={residual:.3e} reg={reg:.3e}")
    log_res = np.log(np.array([row["residual"] for row in results]))
    log_reg = np.log(np.array([row["regularization"] for row in results]))
    dx = np.gradient(log_res)
    dy = np.gradient(log_reg)
    d1 = dy / (dx + 1e-12)
    d2 = np.gradient(d1) / (dx + 1e-12)
    curvature = d2 / np.power(1.0 + d1**2, 1.5)
    idx = int(np.argmax(np.abs(curvature)))
    return results[idx]["alpha"], results


def train(args: argparse.Namespace) -> tuple[NormalizedMLP, NormalizedMLP, OptimalControlHistory, float]:
    if args.alpha_method == "lcurve":
        alpha, alpha_scan = l_curve_method(args)
        print(f"[heat control] selected alpha by L-curve: {alpha:.3e}")
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
    x = torch.linspace(0.0, args.L, args.eval_nx, device=device)
    t = torch.linspace(0.0, args.T, args.eval_nt, device=device)
    xx, tt = torch.meshgrid(x, t, indexing="xy")
    xt = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
    with torch.no_grad():
        u_pinn = u_net(xt).reshape(xx.shape)
        f_pinn = f_net(xt).reshape(xx.shape)
        u_true = analytic_optimal_u(xx, tt)
        f_true = analytic_optimal_f(xx, tt)
    solver = HeatForwardSolver(nx=args.rollout_nx, nt=args.rollout_nt, L=args.L, T=args.T)
    xx_roll, tt_roll = np.meshgrid(solver.x, solver.t, indexing="xy")
    xt_roll = torch.tensor(np.column_stack([xx_roll.reshape(-1), tt_roll.reshape(-1)]), dtype=torch.float32, device=device)
    with torch.no_grad():
        f_roll = f_net(xt_roll).reshape(xx_roll.shape).detach().cpu().numpy()
    states_roll = solver.rollout(forcing=f_roll)
    terminal_target_np = 2.0 * np.sin(solver.x)
    objective_roll = solver.terminal_objective(states_roll[-1], terminal_target_np)
    h1_roll = solver.h1_control_norm(f_roll)
    tikhonov_roll = objective_roll + alpha * h1_roll
    rel_u = relative_l2(u_pinn, u_true)
    rel_f = relative_l2(f_pinn, f_true)
    x_term = torch.linspace(0.0, args.L, args.eval_nx, device=device).reshape(-1, 1)
    xt_term = torch.cat([x_term, args.T * torch.ones_like(x_term)], dim=1)
    with torch.no_grad():
        u_term_pinn = u_net(xt_term).reshape(-1)
        target_term = analytic_terminal_target(x_term).reshape(-1)
    terminal_misfit = 0.5 * args.L * torch.mean((u_term_pinn - target_term) ** 2).item()
    u_roll_term_interp = np.interp(x_term.detach().cpu().numpy().reshape(-1), solver.x, states_roll[-1])
    final_state_rel = float(np.linalg.norm(u_roll_term_interp - target_term.detach().cpu().numpy()) / (np.linalg.norm(target_term.detach().cpu().numpy()) + 1e-12))
    print(f"[heat control] relative L2 error for u*: {rel_u:.3e}")
    print(f"[heat control] relative L2 error for f*: {rel_f:.3e}")
    print(f"[heat control] terminal objective J_roll: {objective_roll:.3e}")
    print(f"[heat control] H1 regularization term: {h1_roll:.3e}")
    print(f"[heat control] Tikhonov functional: {tikhonov_roll:.3e}")
    save_named_columns_csv(outdir / "field_u.csv", flatten_field_to_rows(xx.detach().cpu().numpy(), tt.detach().cpu().numpy(), {"u_true": u_true.detach().cpu().numpy(), "u_pinn": u_pinn.detach().cpu().numpy(), "abs_error": np.abs(u_pinn.detach().cpu().numpy() - u_true.detach().cpu().numpy())}))
    save_named_columns_csv(outdir / "control_f.csv", flatten_field_to_rows(xx.detach().cpu().numpy(), tt.detach().cpu().numpy(), {"f_true": f_true.detach().cpu().numpy(), "f_pred": f_pinn.detach().cpu().numpy(), "abs_error": np.abs(f_pinn.detach().cpu().numpy() - f_true.detach().cpu().numpy())}))
    save_named_columns_csv(outdir / "terminal_state.csv", {"x": x_term.detach().cpu().numpy().reshape(-1), "uT_target": target_term.detach().cpu().numpy(), "uT_pinn": u_term_pinn.detach().cpu().numpy()})
    save_named_columns_csv(outdir / "rollout_terminal_state.csv", {"x": solver.x, "uT_target": terminal_target_np, "uT_rollout": states_roll[-1]})
    save_named_columns_csv(outdir / "loss_history.csv", {"epoch": np.arange(1, len(history.total) + 1, dtype=np.int32), "loss_total": np.array(history.total, dtype=np.float64), "loss_pde": np.array(history.pde, dtype=np.float64), "loss_bc": np.array(history.bc, dtype=np.float64), "loss_ic": np.array(history.ic, dtype=np.float64), "loss_obj": np.array(history.objective, dtype=np.float64), "loss_reg_h1": np.array(history.regularization, dtype=np.float64)})
    if history.alpha_scan:
        save_named_columns_csv(outdir / "alpha_lcurve.csv", {"alpha": np.array([row["alpha"] for row in history.alpha_scan], dtype=np.float64), "residual": np.array([row["residual"] for row in history.alpha_scan], dtype=np.float64), "regularization": np.array([row["regularization"] for row in history.alpha_scan], dtype=np.float64)})
    save_named_columns_csv(outdir / "summary.csv", {"wJ": np.array([args.wJ], dtype=np.float64), "alpha": np.array([alpha], dtype=np.float64), "rel_u": np.array([rel_u], dtype=np.float64), "rel_f": np.array([rel_f], dtype=np.float64), "terminal_misfit_pinn": np.array([terminal_misfit], dtype=np.float64), "objective_rollout": np.array([objective_roll], dtype=np.float64), "h1_rollout": np.array([h1_roll], dtype=np.float64), "tikhonov_rollout": np.array([tikhonov_roll], dtype=np.float64), "final_state_rel": np.array([final_state_rel], dtype=np.float64)})


def main() -> None:
    args = make_parser().parse_args()
    u_net, f_net, history, alpha = train(args)
    evaluate_and_save(u_net, f_net, history, alpha, args)


if __name__ == "__main__":
    main()
