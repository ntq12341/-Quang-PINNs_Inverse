import math
import numpy as np

RADIUS = 0.5


def u_exact(x, y):
    """Exact solution for Example 5.4: u = sin(x1)cos(x2) + exp(x1)sin(x2)."""
    return np.sin(x) * np.cos(y) + np.exp(x) * np.sin(y)


def source_exact(x, y):
    """Right-hand side h = Δu for u = sin(x) cos(y)."""
    return -2.0 * np.sin(x) * np.cos(y)


def _gaussian_noise(size, sigma):
    return np.random.normal(0.0, sigma, size=size)


def sample_interior_disk(n_points, radius=RADIUS):
    """Uniform sampling in a disk."""
    r = radius * np.sqrt(np.random.uniform(0.0, 1.0, n_points))
    theta = np.random.uniform(0.0, 2.0 * math.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def sample_boundary_arc(n_points, theta_min, theta_max, radius=RADIUS):
    theta = np.random.uniform(theta_min, theta_max, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    nx = np.cos(theta)
    ny = np.sin(theta)
    return x, y, nx, ny, theta


def normal_derivative_exact(x, y, nx, ny):
    """∂u/∂n = grad(u)·n  for  u = sin(x)cos(y) + exp(x)sin(y).
    ∂u/∂x = cos(x)cos(y) + exp(x)sin(y)
    ∂u/∂y = -sin(x)sin(y) + exp(x)cos(y)
    """
    ux = np.cos(x) * np.cos(y) + np.exp(x) * np.sin(y)
    uy = -np.sin(x) * np.sin(y) + np.exp(x) * np.cos(y)
    return ux * nx + uy * ny


def get_boundary_split(gamma1_fraction=0.5):
    """Return contiguous arc split for Γ1 and Γ2 on [0, 2π]."""
    gamma1_fraction = float(gamma1_fraction)
    if not (0.0 < gamma1_fraction < 1.0):
        raise ValueError("gamma1_fraction must be in (0,1).")

    theta_start = 0.0
    theta_split = 2.0 * math.pi * gamma1_fraction
    return theta_start, theta_split


def generate_inverse_data_circle(
    noise_level=0.01,
    N_r=5000,
    N_b=1500,
    N_reg=500,
    gamma1_fraction=0.75,
):
    """
    Generate data for Example 5.4 circular Cauchy setup.

    Γ1 is a contiguous arc of length gamma1_fraction * |∂Ω|,
    Γ2 is the remaining arc.
    """
    theta_start, theta_split = get_boundary_split(gamma1_fraction)

    x_r, y_r = sample_interior_disk(N_r)
    h_exact = source_exact(x_r, y_r)
    h_noisy = h_exact + _gaussian_noise(h_exact.shape, noise_level)

    # Γ1: known Cauchy arc
    x_b, y_b, nx_b, ny_b, _ = sample_boundary_arc(N_b, theta_start, theta_split)
    u_b_exact = u_exact(x_b, y_b)
    g_b_exact = normal_derivative_exact(x_b, y_b, nx_b, ny_b)

    u_b_noisy = u_b_exact + _gaussian_noise(u_b_exact.shape, noise_level)
    g_b_noisy = g_b_exact + _gaussian_noise(g_b_exact.shape, noise_level)

    # Γ2: unknown arc for regularization/evaluation
    x_k, y_k, nx_k, ny_k, _ = sample_boundary_arc(N_reg, theta_split, 2.0 * math.pi)

    return {
        "residual": (x_r, y_r, h_noisy),
        "cauchy": (x_b, y_b, nx_b, ny_b, u_b_noisy, g_b_noisy),
        "regularization": (x_k, y_k, nx_k, ny_k),
    }


def evaluation_boundary_gamma2(n_points=500, radius=RADIUS, gamma1_fraction=0.5):
    _, theta_split = get_boundary_split(gamma1_fraction)
    theta = np.linspace(theta_split, 2.0 * math.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return theta, x, y


def evaluation_grid(n=180, radius=RADIUS):
    xs = np.linspace(-radius, radius, n)
    ys = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(xs, ys)
    mask = (X**2 + Y**2) <= radius**2
    return X, Y, mask