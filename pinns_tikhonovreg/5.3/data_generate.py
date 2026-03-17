import math
from functools import lru_cache

import numpy as np

X1_MIN, X1_MAX = 0.0, 5.0 * math.pi
X2_MIN, X2_MAX = 0.0, 1.0


def top_boundary_example_53_step(x):
    """Example 5.3 top boundary v(x1): step function on [0, 5pi]."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)

    left = X1_MAX / 3.0
    right = 2.0 * X1_MAX / 3.0
    out[(x >= left) & (x <= right)] = 1.0
    return out


@lru_cache(maxsize=8)
def _step_coefficients(n_terms=400, n_quad=6000):
    """Fourier sine coefficients of Example 5.3 step boundary."""
    n = np.arange(1, n_terms + 1, dtype=np.float64)
    xq = np.linspace(X1_MIN, X1_MAX, n_quad, dtype=np.float64)
    vq = top_boundary_example_53_step(xq)

    sin_nxq = np.sin(np.outer(n * math.pi / X1_MAX, xq))
    coeff = (2.0 / X1_MAX) * np.trapezoid(vq * sin_nxq, xq, axis=1)
    return coeff


def exact_solution_example_53(x, y, n_terms=400):
    """Approximate Example 5.3 exact field via Laplace sine-series solution."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    original_shape = x_arr.shape

    x_flat = x_arr.reshape(-1)
    y_flat = y_arr.reshape(-1)

    n = np.arange(1, n_terms + 1, dtype=np.float64)
    b_n = _step_coefficients(n_terms=n_terms)
    lam = n * math.pi / X1_MAX

    sin_lx = np.sin(np.outer(lam, x_flat))

    ly = np.outer(lam, y_flat)
    l1 = lam[:, None]
    ratio = np.exp(l1 * (y_flat[None, :] - 1.0)) * (1.0 - np.exp(-2.0 * ly)) / (
        1.0 - np.exp(-2.0 * l1)
    )

    u_flat = np.sum((b_n[:, None] * ratio * sin_lx), axis=0)
    return u_flat.reshape(original_shape)


def _gaussian_noise(signal, noise_level):
    """Add Gaussian noise as in the paper: signal + sigma * xi, xi~N(0,1)."""
    return np.random.normal(0.0, noise_level, size=signal.shape)


def generate_forward_data(N_r=10000, N_b=2000, top_boundary_fn=top_boundary_example_53_step):
    """Generate training points for the forward problem."""
    x_r = np.random.uniform(X1_MIN, X1_MAX, N_r)
    y_r = np.random.uniform(X2_MIN, X2_MAX, N_r)

    x_b_left = np.zeros(N_b)
    y_b_left = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_right = np.full(N_b, X1_MAX)
    y_b_right = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_top = np.random.uniform(X1_MIN, X1_MAX, N_b)
    y_b_top = np.full(N_b, X2_MAX)
    u_b_top = np.asarray(top_boundary_fn(x_b_top))

    x_b_bottom = np.random.uniform(X1_MIN, X1_MAX, N_b)
    y_b_bottom = np.zeros(N_b)

    return {
        "residual": (x_r, y_r),
        "left": (x_b_left, y_b_left),
        "right": (x_b_right, y_b_right),
        "top": (x_b_top, y_b_top, u_b_top),
        "bottom": (x_b_bottom, y_b_bottom),
    }


def generate_inverse_data(
    u_bottom_exact,
    noise_level=0.01,
    N_r=6000,
    N_b=2000,
    N_cauchy=2000,
    N_reg=500,
):
    """Generate training points for the inverse Cauchy problem."""
    x_r = np.random.uniform(X1_MIN, X1_MAX, N_r)
    y_r = np.random.uniform(X2_MIN, X2_MAX, N_r)

    x_b_left = np.zeros(N_b)
    y_b_left = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_right = np.full(N_b, X1_MAX)
    y_b_right = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_cauchy = np.random.uniform(X1_MIN, X1_MAX, N_cauchy)
    y_cauchy = np.zeros(N_cauchy)
    u_cauchy_exact = np.asarray(u_bottom_exact(x_cauchy))
    u_cauchy_noisy = u_cauchy_exact + _gaussian_noise(u_cauchy_exact, noise_level)

    x_b_bottom = np.random.uniform(X1_MIN, X1_MAX, N_b)
    y_b_bottom = np.zeros(N_b)

    x_reg = np.random.uniform(X1_MIN, X1_MAX, N_reg)
    y_reg = np.full(N_reg, X2_MAX)

    return {
        "residual": (x_r, y_r),
        "left": (x_b_left, y_b_left),
        "right": (x_b_right, y_b_right),
        "cauchy": (x_cauchy, y_cauchy, u_cauchy_noisy),
        "bottom": (x_b_bottom, y_b_bottom),
        "regularization": (x_reg, y_reg),
    }
