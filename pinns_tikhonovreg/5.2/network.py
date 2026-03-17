import copy
import numpy as np
import torch
import torch.nn as nn

from data_generate import X1_MIN, X1_MAX, X2_MIN, X2_MAX


class PINN(nn.Module):
    """Physics-Informed Neural Network for Laplace Cauchy problem."""

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = [2, 7, 7, 7, 1]
        self.layer_sizes = list(layers)

        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.activation = nn.Tanh()

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, y):
        x_norm = 2.0 * (x - X1_MIN) / (X1_MAX - X1_MIN) - 1.0
        y_norm = 2.0 * (y - X2_MIN) / (X2_MAX - X2_MIN) - 1.0
        u = torch.cat([x_norm, y_norm], dim=1)

        for layer in self.layers[:-1]:
            u = self.activation(layer(u))
        return self.layers[-1](u)

    def pde_residual(self, x, y):
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        u = self.forward(x, y)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0]
        return u_xx + u_yy


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

    x_r, y_r = data["residual"]
    x_r_t = _to_tensor(x_r, device)
    y_r_t = _to_tensor(y_r, device)
    residual = model.pde_residual(x_r_t, y_r_t)
    losses["pde"] = torch.mean(residual ** 2)
    total_loss = total_loss + w("pde") * losses["pde"]

    if "left" in data:
        x_l, y_l = data["left"]
        u_l = model(_to_tensor(x_l, device), _to_tensor(y_l, device))
        losses["left"] = torch.mean(u_l ** 2)
        total_loss = total_loss + w("left") * losses["left"]

    if "right" in data:
        x_rr, y_rr = data["right"]
        u_rr = model(_to_tensor(x_rr, device), _to_tensor(y_rr, device))
        losses["right"] = torch.mean(u_rr ** 2)
        total_loss = total_loss + w("right") * losses["right"]

    if "top" in data:
        x_t, y_t, u_t = data["top"]
        u_pred = model(_to_tensor(x_t, device), _to_tensor(y_t, device))
        u_ref = _to_tensor(u_t, device)
        losses["top"] = torch.mean((u_pred - u_ref) ** 2)
        total_loss = total_loss + w("top") * losses["top"]

    if "bottom" in data:
        x_b, y_b = data["bottom"]
        x_b_t = _to_tensor(x_b, device, requires_grad=True)
        y_b_t = _to_tensor(y_b, device, requires_grad=True)
        u_b = model(x_b_t, y_b_t)
        u_b_y = torch.autograd.grad(
            u_b, y_b_t, grad_outputs=torch.ones_like(u_b), create_graph=True
        )[0]
        losses["bottom_neumann"] = torch.mean(u_b_y ** 2)
        total_loss = total_loss + w("bottom_neumann") * losses["bottom_neumann"]

    if "cauchy" in data:
        x_c, y_c, u_c_noisy = data["cauchy"]
        u_pred = model(_to_tensor(x_c, device), _to_tensor(y_c, device))
        u_obs = _to_tensor(u_c_noisy, device)
        losses["cauchy"] = torch.mean((u_pred - u_obs) ** 2)
        total_loss = total_loss + w("cauchy") * losses["cauchy"]

    if "regularization" in data:
        x_k, y_k = data["regularization"]
        x_k_t = _to_tensor(x_k, device, requires_grad=True)
        y_k_t = _to_tensor(y_k, device, requires_grad=True)
        u_k = model(x_k_t, y_k_t)
        u_k_n = torch.autograd.grad(
            u_k, y_k_t, grad_outputs=torch.ones_like(u_k), create_graph=True
        )[0]
        losses["regularization"] = torch.mean(u_k_n ** 2)
        total_loss = total_loss + w("regularization") * alpha * losses["regularization"]
    else:
        losses["regularization"] = torch.tensor(0.0, device=device)

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

    # PyTorch LBFGS performs several line-search evaluations per step.
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


def l_curve_method(
    model,
    data,
    alphas,
    max_iter=200,
    device="cpu",
    loss_weights=None,
):
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

        _, components = compute_loss(
            model_copy, data, alpha=alpha, device=device, loss_weights=loss_weights
        )
        residual_norm = np.sqrt(max(components.get("cauchy", 0.0), 1e-16))
        reg_norm = np.sqrt(max(components.get("regularization", 0.0), 1e-16))

        results.append(
            {
                "alpha": float(alpha),
                "residual": float(residual_norm),
                "regularization": float(reg_norm),
            }
        )

        print(
            f"alpha={alpha:.3e} | residual={residual_norm:.6e} | reg={reg_norm:.6e}"
        )

    log_res = np.log(np.array([r["residual"] for r in results]))
    log_reg = np.log(np.array([r["regularization"] for r in results]))

    dx = np.gradient(log_res)
    dy = np.gradient(log_reg)
    d1 = dy / (dx + 1e-12)
    d2 = np.gradient(d1) / (dx + 1e-12)
    curvature = d2 / np.power(1.0 + d1**2, 1.5)

    optimal_idx = int(np.argmax(np.abs(curvature)))
    return results[optimal_idx]["alpha"], results
