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
        u    = self.forward(x, y)
        u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_y  = torch.autograd.grad(u,   y, grad_outputs=torch.ones_like(u),   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        return u_xx + u_yy


def _to_tensor(arr, device, requires_grad=False):
    t = torch.tensor(arr, dtype=torch.float32, device=device).reshape(-1, 1)
    if requires_grad:
        t.requires_grad_(True)
    return t


def compute_loss(model, data, device="cpu"):
    """
    Keys được xử lý:
      "residual" — PDE loss
      "left"     — Dirichlet u=0 tại x=0
      "right"    — Dirichlet u=0 tại x=pi
      "top"      — Dirichlet u=v(x) tại y=1   (forward only)
      "bottom"   — Neumann du/dy=0 tại y=0    (cả forward và inverse, không noise)
      "cauchy"   — Dirichlet u=u_delta tại y=0 (inverse only, có noise)
    """
    losses = {}
    total_loss = 0.0

    # PDE
    x_r, y_r = data["residual"]
    res = model.pde_residual(_to_tensor(x_r, device), _to_tensor(y_r, device))
    losses["pde"] = torch.mean(res ** 2)
    total_loss = total_loss + losses["pde"]

    # Dirichlet x=0
    if "left" in data:
        x_l, y_l = data["left"]
        losses["left"] = torch.mean(model(_to_tensor(x_l, device), _to_tensor(y_l, device)) ** 2)
        total_loss = total_loss + losses["left"]

    # Dirichlet x=pi
    if "right" in data:
        x_r2, y_r2 = data["right"]
        losses["right"] = torch.mean(model(_to_tensor(x_r2, device), _to_tensor(y_r2, device)) ** 2)
        total_loss = total_loss + losses["right"]

    # Dirichlet y=1 (forward)
    if "top" in data:
        x_t, y_t, u_t = data["top"]
        u_pred = model(_to_tensor(x_t, device), _to_tensor(y_t, device))
        losses["top"] = torch.mean((u_pred - _to_tensor(u_t, device)) ** 2)
        total_loss = total_loss + losses["top"]

    # Neumann du/dy=0 tại y=0 — BC biết trước, không noise, dùng cho cả hai bài toán
    if "bottom" in data:
        x_b, y_b = data["bottom"]
        x_b_t = _to_tensor(x_b, device, requires_grad=True)
        y_b_t = _to_tensor(y_b, device, requires_grad=True)
        u_b   = model(x_b_t, y_b_t)
        u_b_y = torch.autograd.grad(
            u_b, y_b_t, grad_outputs=torch.ones_like(u_b), create_graph=True
        )[0]
        losses["bottom_neumann"] = torch.mean(u_b_y ** 2)
        total_loss = total_loss + losses["bottom_neumann"]

    # Cauchy Dirichlet tại y=0 (inverse only, có noise)
    if "cauchy" in data:
        x_c, y_c, u_c = data["cauchy"]
        u_pred = model(_to_tensor(x_c, device), _to_tensor(y_c, device))
        losses["cauchy"] = torch.mean((u_pred - _to_tensor(u_c, device)) ** 2)
        total_loss = total_loss + losses["cauchy"]

    losses_float = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    return total_loss, losses_float


def train_lbfgs(model, data, max_iter=1000, device="cpu", print_every=100):
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
            loss, _ = compute_loss(model, data, device=device)
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