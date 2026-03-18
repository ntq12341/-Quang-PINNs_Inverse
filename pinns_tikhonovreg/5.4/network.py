import copy
import numpy as np
import torch
import torch.nn as nn


class PINN(nn.Module):
    """PINN for 2D Poisson Cauchy problem in a circle."""

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            # Example 5.4 in the paper: 5 hidden layers, 7 neurons each.
            layers = [2, 7, 7, 7, 7, 7, 1]

        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.activation = nn.Tanh()

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, y):
        u = torch.cat([x, y], dim=1)
        for layer in self.layers[:-1]:
            u = self.activation(layer(u))
        return self.layers[-1](u)


def _to_tensor(arr, device, requires_grad=False):
    tensor = torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def compute_loss(model, data, alpha=0.0, device="cpu", loss_weights=None):
    if loss_weights is None:
        loss_weights = {}

    def w(name):
        return float(loss_weights.get(name, 1.0))

    losses = {}
    total_loss = 0.0

    x_r, y_r, h_r = data["residual"]
    x_r_t = _to_tensor(x_r, device, requires_grad=True)
    y_r_t = _to_tensor(y_r, device, requires_grad=True)
    h_r_t = _to_tensor(h_r, device)

    u_r = model(x_r_t, y_r_t)
    u_x = torch.autograd.grad(u_r, x_r_t, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
    u_y = torch.autograd.grad(u_r, y_r_t, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_r_t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_r_t, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    pde_res = u_xx + u_yy - h_r_t
    losses["pde"] = torch.mean(pde_res**2)
    total_loss = total_loss + w("pde") * losses["pde"]

    x_b, y_b, nx_b, ny_b, u_b, g_b = data["cauchy"]
    x_b_t = _to_tensor(x_b, device, requires_grad=True)
    y_b_t = _to_tensor(y_b, device, requires_grad=True)
    nx_b_t = _to_tensor(nx_b, device)
    ny_b_t = _to_tensor(ny_b, device)
    u_b_t = _to_tensor(u_b, device)
    g_b_t = _to_tensor(g_b, device)

    u_pred_b = model(x_b_t, y_b_t)
    ux_b = torch.autograd.grad(u_pred_b, x_b_t, grad_outputs=torch.ones_like(u_pred_b), create_graph=True)[0]
    uy_b = torch.autograd.grad(u_pred_b, y_b_t, grad_outputs=torch.ones_like(u_pred_b), create_graph=True)[0]
    g_pred_b = ux_b * nx_b_t + uy_b * ny_b_t

    losses["cauchy_u"] = torch.mean((u_pred_b - u_b_t) ** 2)
    losses["cauchy_g"] = torch.mean((g_pred_b - g_b_t) ** 2)
    total_loss = total_loss + w("cauchy_u") * losses["cauchy_u"]
    total_loss = total_loss + w("cauchy_g") * losses["cauchy_g"]

    x_k, y_k, nx_k, ny_k = data["regularization"]
    x_k_t = _to_tensor(x_k, device, requires_grad=True)
    y_k_t = _to_tensor(y_k, device, requires_grad=True)
    nx_k_t = _to_tensor(nx_k, device)
    ny_k_t = _to_tensor(ny_k, device)

    u_k = model(x_k_t, y_k_t)
    ux_k = torch.autograd.grad(u_k, x_k_t, grad_outputs=torch.ones_like(u_k), create_graph=True)[0]
    uy_k = torch.autograd.grad(u_k, y_k_t, grad_outputs=torch.ones_like(u_k), create_graph=True)[0]
    g_k = ux_k * nx_k_t + uy_k * ny_k_t

    losses["regularization"] = torch.mean(g_k**2)
    total_loss = total_loss + w("regularization") * alpha * losses["regularization"]

    losses_float = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    return total_loss, losses_float


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
                model, data, alpha=alpha, device=device, loss_weights=loss_weights
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


def l_curve_method(model, data, alphas, max_iter=200, device="cpu", loss_weights=None):
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

        _, comp = compute_loss(model_copy, data, alpha=alpha, device=device, loss_weights=loss_weights)
        residual = np.sqrt(max(comp.get("cauchy_u", 0.0) + comp.get("cauchy_g", 0.0), 1e-16))
        reg = np.sqrt(max(comp.get("regularization", 0.0), 1e-16))
        results.append({"alpha": float(alpha), "residual": float(residual), "regularization": float(reg)})
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
    max_iter=200,
    device="cpu",
    loss_weights=None,
    tau=1.1,
    discrepancy_terms=("pde", "cauchy_u", "cauchy_g"),
):
    results = []

    # Three noisy channels in this setup: source h, Dirichlet u on Γ1, Neumann g on Γ1
    n_noisy_terms = max(len(discrepancy_terms), 1)
    target_rms = max(float(tau) * float(noise_level) * np.sqrt(n_noisy_terms), 1e-12)

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

        _, comp = compute_loss(model_copy, data, alpha=alpha, device=device, loss_weights=loss_weights)
        fidelity_sq = 0.0
        used_terms = []
        for k in discrepancy_terms:
            if k in comp:
                fidelity_sq += float(comp[k])
                used_terms.append(k)

        residual = np.sqrt(max(fidelity_sq, 1e-16))
        reg = np.sqrt(max(comp.get("regularization", 0.0), 1e-16))
        gap = abs(residual - target_rms)
        results.append(
            {
                "alpha": float(alpha),
                "residual": float(residual),
                "regularization": float(reg),
                "discrepancy_gap": float(gap),
                "used_terms": tuple(used_terms),
            }
        )
        print(
            f"alpha={alpha:.3e} | fidelity_res={residual:.6e} | "
            f"target={target_rms:.6e} | gap={gap:.6e}"
        )

    idx = int(np.argmin([r["discrepancy_gap"] for r in results]))
    return results[idx]["alpha"], results, target_rms
