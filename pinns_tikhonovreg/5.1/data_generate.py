import math
import numpy as np

X1_MIN, X1_MAX = 0.0, math.pi
X2_MIN, X2_MAX = 0.0, 1.0


def u_exact(x, y):
    """Exact solution in Example 5.1: u(x1, x2) = sin(x1) cosh(x2)."""
    return np.sin(x) * np.cosh(y)


def _gaussian_noise(signal, noise_level):
    """Add Gaussian noise as in the paper: signal + sigma * xi, xi~N(0,1)."""
    return np.random.normal(0.0, noise_level, size=signal.shape)


def generate_forward_data(N_r=10000, N_b=1000):
    """Generate training points for the forward problem (Step 1 in the paper)."""
    x_r = np.random.uniform(X1_MIN, X1_MAX, N_r)
    y_r = np.random.uniform(X2_MIN, X2_MAX, N_r)

    x_b_left = np.zeros(N_b)
    y_b_left = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_right = np.full(N_b, X1_MAX)
    y_b_right = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_top = np.random.uniform(X1_MIN, X1_MAX, N_b)
    y_b_top = np.full(N_b, X2_MAX)
    u_b_top = u_exact(x_b_top, y_b_top)

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
    N_r=5000,
    N_b=1000,
    N_cauchy=1000,
    N_reg=500,
):
    """Generate training points for the inverse Cauchy problem (Step 3 in the paper)."""
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
