import copy
import numpy as np
import torch
import torch.nn as nn

from data_generate import (
    X1_MAX,
    X1_MIN,
    X2_MAX,
    X2_MIN,
)


class PINN(nn.Module):
    """PINN for 2D nonlinear Cauchy problem (Example 5.6).

    Equation: -div(q(u)*grad(u)) = h,  q(u) = 1 + u^2
    Architecture: 4 hidden layers x 10 neurons, tanh activation (paper setup).
    """

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            # Paper: 4 hidden layers, 10 neurons each
            layers = [2, 10, 10, 10, 10, 1]

        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.activation = nn.Tanh()

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x1, x2):
        # Normalize to [-1, 1]
        x1n = 2.0 * (x1 - X1_MIN) / (X1_MAX - X1_MIN) - 1.0
        x2n = 2.0 * (x2 - X2_MIN) / (X2_MAX - X2_MIN) - 1.0

        u = torch.cat([x1n, x2n], dim=1)
        for layer in self.layers[:-1]:
            u = self.activation(layer(u))
        return self.layers[-1](u)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_tensor(arr, device, requires_grad=False):
    tensor = torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _add_dirichlet_loss(model, data, key, losses, total_loss, w, device):
    if key not in data:
        return total_loss

    x1, x2, u_ref = data[key]
    u_pred = model(_to_tensor(x1, device), _to_tensor(x2, device))
    u_obs = _to_tensor(u_ref, device)
    losses[key] = torch.mean((u_pred - u_obs) ** 2)
    return total_loss + w(key) * losses[key]


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(model, data, alpha=0.0, device="cpu", loss_weights=None):
    """Compute plain least-squares loss (no Tikhonov term) for Example 5.6.

    PDE term: -div(q(u)*grad(u)) = h
      => L[u] = -(q'(u)*|grad u|^2 + q(u)*Delta u) - h  = 0
      where q(u)=1+u^2, q'(u)=2u.
    """
    if loss_weights is None:
        loss_weights = {}

    def w(name):
        return float(loss_weights.get(name, 1.0))

    losses = {}
    total_loss = 0.0

    # -----------------------------------------------------------------------
    # 1. PDE residual: -div(q(u)*grad(u)) - h = 0
    # -----------------------------------------------------------------------
    x1_r, x2_r, h_r = data["residual"]
    x1_r_t = _to_tensor(x1_r, device, requires_grad=True)
    x2_r_t = _to_tensor(x2_r, device, requires_grad=True)
    h_r_t = _to_tensor(h_r, device)

    u_r = model(x1_r_t, x2_r_t)

    u_x1 = torch.autograd.grad(
        u_r, x1_r_t, grad_outputs=torch.ones_like(u_r), create_graph=True
    )[0]
    u_x2 = torch.autograd.grad(
        u_r, x2_r_t, grad_outputs=torch.ones_like(u_r), create_graph=True
    )[0]

    u_x1x1 = torch.autograd.grad(
        u_x1, x1_r_t, grad_outputs=torch.ones_like(u_x1), create_graph=True
    )[0]
    u_x2x2 = torch.autograd.grad(
        u_x2, x2_r_t, grad_outputs=torch.ones_like(u_x2), create_graph=True
    )[0]

    # q(u) = 1 + u^2,  q'(u) = 2u
    q = 1.0 + u_r**2
    dq = 2.0 * u_r

    # -div(q*grad(u)) = -(dq*(u_x1^2+u_x2^2) + q*(u_x1x1+u_x2x2))
    pde_res = -(dq * (u_x1**2 + u_x2**2) + q * (u_x1x1 + u_x2x2)) - h_r_t

    # No per-term scaling in the no-regularization variant:
    # keep the plain least-squares structure from the paper.
    losses["pde"] = torch.mean(pde_res**2)
    total_loss = total_loss + w("pde") * losses["pde"]

    # -----------------------------------------------------------------------
    # 2. Cauchy data on Gamma1 (x2=0)
    #    Dirichlet: u = f
    #    Neumann: q(u) * du/dn = g with n=(0,-1) on Gamma1
    # -----------------------------------------------------------------------
    x1_c, x2_c, u_c, g_c = data["cauchy"]
    x1_c_t = _to_tensor(x1_c, device, requires_grad=True)
    x2_c_t = _to_tensor(x2_c, device, requires_grad=True)
    u_c_t = _to_tensor(u_c, device)
    g_c_t = _to_tensor(g_c, device)

    u_c_pred = model(x1_c_t, x2_c_t)

    u_c_x2 = torch.autograd.grad(
        u_c_pred,
        x2_c_t,
        grad_outputs=torch.ones_like(u_c_pred),
        create_graph=True,
    )[0]

    q_c = 1.0 + u_c_pred**2
    g_c_pred = -q_c * u_c_x2

    losses["cauchy_u"] = torch.mean((u_c_pred - u_c_t) ** 2)
    losses["cauchy_g"] = torch.mean((g_c_pred - g_c_t) ** 2)
    total_loss = total_loss + w("cauchy_u") * losses["cauchy_u"]
    total_loss = total_loss + w("cauchy_g") * losses["cauchy_g"]

    # -----------------------------------------------------------------------
    # 3. Dirichlet on Gamma3 (x1=0) and Gamma4 (x1=1)
    # -----------------------------------------------------------------------
    total_loss = _add_dirichlet_loss(model, data, "gamma3", losses, total_loss, w, device)
    total_loss = _add_dirichlet_loss(model, data, "gamma4", losses, total_loss, w, device)

    losses_float = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    return total_loss, losses_float


# ---------------------------------------------------------------------------
# L-BFGS trainer
# ---------------------------------------------------------------------------

def train_lbfgs(
    model,
    data,
    alpha=0.0,
    max_iter=1000,
    device="cpu",
    print_every=100,
    loss_weights=None,
):
    model.to(device)
    model.train()

    inner_steps = 20
    n_steps = max(1, int(np.ceil(max_iter / inner_steps)))

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=inner_steps,
        max_eval=inner_steps * 2,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    history = []
    step_index = 0

    for _ in range(n_steps):
        closure_state = {"loss": None}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_loss(
                model,
                data,
                alpha=alpha,
                device=device,
                loss_weights=loss_weights,
            )
            loss.backward()
            closure_state["loss"] = float(loss.detach().cpu().item())
            return loss

        optimizer.step(closure)

        if closure_state["loss"] is not None:
            history.append(closure_state["loss"])

        step_index += inner_steps
        if print_every > 0 and step_index % print_every == 0 and history:
            print(f"Iter ~{step_index:4d} | loss = {history[-1]:.6e}")

    return history


# ---------------------------------------------------------------------------
# Regularization parameter selection methods
# ---------------------------------------------------------------------------

def l_curve_method(model, data, alphas, max_iter=600, device="cpu", loss_weights=None):
    """Select alpha by the L-curve method."""
    results = []

    for alpha in alphas:
        model_copy = copy.deepcopy(model).to(device)
        train_lbfgs(
            model_copy,
            data,
            alpha=alpha,
            max_iter=max_iter,
            device=device,
            print_every=0,
            loss_weights=loss_weights,
        )

        _, comp = compute_loss(
            model_copy,
            data,
            alpha=alpha,
            device=device,
            loss_weights=loss_weights,
        )

        residual_sq = (
            comp.get("pde", 0.0)
            + comp.get("cauchy_u", 0.0)
            + comp.get("cauchy_g", 0.0)
            + comp.get("gamma3", 0.0)
            + comp.get("gamma4", 0.0)
        )
        residual = np.sqrt(max(residual_sq, 1e-16))
        reg = np.sqrt(max(comp.get("regularization", 0.0), 1e-16))
        results.append(
            {"alpha": float(alpha), "residual": float(residual), "regularization": float(reg)}
        )

        print(f"alpha={alpha:.3e} | residual={residual:.6e} | reg={reg:.6e}")

    log_res = np.log(np.array([r["residual"] for r in results]))
    log_reg = np.log(np.array([r["regularization"] for r in results]))
    dx = np.gradient(log_res)
    dy = np.gradient(log_reg)
    d1 = dy / (dx + 1e-12)
    d2 = np.gradient(d1) / (dx + 1e-12)
    curvature = d2 / np.power(1.0 + d1**2, 1.5)
    idx = int(np.argmax(np.abs(curvature)))
    return results[idx]["alpha"], results


def discrepancy_principle_method(
    model,
    data,
    alphas,
    noise_level,
    max_iter=600,
    device="cpu",
    loss_weights=None,
    tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g", "gamma3", "gamma4"),
):
    """Select alpha by the discrepancy principle."""
    results = []
    target_rms = max(
        float(tau) * float(noise_level) * np.sqrt(max(len(discrepancy_terms), 1)), 1e-12
    )

    for alpha in alphas:
        model_copy = copy.deepcopy(model).to(device)
        train_lbfgs(
            model_copy,
            data,
            alpha=alpha,
            max_iter=max_iter,
            device=device,
            print_every=0,
            loss_weights=loss_weights,
        )

        _, comp = compute_loss(
            model_copy,
            data,
            alpha=alpha,
            device=device,
            loss_weights=loss_weights,
        )

        fidelity_sq = sum(float(comp[k]) for k in discrepancy_terms if k in comp)
        residual = np.sqrt(max(fidelity_sq, 1e-16))
        reg = np.sqrt(max(comp.get("regularization", 0.0), 1e-16))
        gap = abs(residual - target_rms)

        results.append(
            {
                "alpha": float(alpha),
                "residual": float(residual),
                "regularization": float(reg),
                "discrepancy_gap": float(gap),
                "used_terms": tuple(k for k in discrepancy_terms if k in comp),
            }
        )

        print(
            f"alpha={alpha:.3e} | fidelity_res={residual:.6e} | "
            f"target={target_rms:.6e} | gap={gap:.6e}"
        )

    idx = int(np.argmin([r["discrepancy_gap"] for r in results]))
    return results[idx]["alpha"], results, target_rms
