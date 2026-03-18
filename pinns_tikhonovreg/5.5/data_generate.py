import numpy as np

X1_MIN, X1_MAX = 0.0, 1.0
X2_MIN, X2_MAX = 0.0, 1.0
X3_MIN, X3_MAX = 0.0, 1.0


def u_exact(x1, x2, x3):
    """Exact solution in Example 5.5.

    u(x1, x2, x3) = ((x2 - x2^2) * (x3^3 + 3*x3 - 2)) / (x1 + 2)
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    x3 = np.asarray(x3, dtype=np.float64)
    return ((x2 - x2**2) * (x3**3 + 3.0 * x3 - 2.0)) / (x1 + 2.0)


def source_exact(x1, x2, x3):
    """Right-hand side h = ?u for the exact solution in Example 5.5."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    x3 = np.asarray(x3, dtype=np.float64)

    a = x2 - x2**2
    b = x3**3 + 3.0 * x3 - 2.0

    term_x1 = 2.0 * a * b / (x1 + 2.0) ** 3
    term_x2 = -2.0 * b / (x1 + 2.0)
    term_x3 = 6.0 * x3 * a / (x1 + 2.0)
    return term_x1 + term_x2 + term_x3


def normal_derivative_gamma1_exact(x1, x2):
    """Outward normal derivative on G1: x3=0, n=(0,0,-1)."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    du_dx3_at_gamma1 = 3.0 * (x2 - x2**2) / (x1 + 2.0)
    return -du_dx3_at_gamma1


def _gaussian_noise(signal, sigma):
    return np.random.normal(0.0, sigma, size=np.asarray(signal).shape)


def _sample_interior(n):
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    x3 = np.random.uniform(X3_MIN, X3_MAX, n)
    return x1, x2, x3


def _sample_gamma1_x3_eq_0(n):
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    x3 = np.zeros(n, dtype=np.float64)
    return x1, x2, x3


def _sample_gamma2_x3_eq_1(n):
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    x3 = np.ones(n, dtype=np.float64)
    return x1, x2, x3


def _sample_gamma3_x1_eq_0(n):
    x1 = np.zeros(n, dtype=np.float64)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    x3 = np.random.uniform(X3_MIN, X3_MAX, n)
    return x1, x2, x3


def _sample_gamma4_x1_eq_1(n):
    x1 = np.ones(n, dtype=np.float64)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    x3 = np.random.uniform(X3_MIN, X3_MAX, n)
    return x1, x2, x3


def _sample_gamma5_x2_eq_0(n):
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.zeros(n, dtype=np.float64)
    x3 = np.random.uniform(X3_MIN, X3_MAX, n)
    return x1, x2, x3


def _sample_gamma6_x2_eq_1(n):
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.ones(n, dtype=np.float64)
    x3 = np.random.uniform(X3_MIN, X3_MAX, n)
    return x1, x2, x3


def generate_inverse_data_55(
    noise_level=0.01,
    N_r=6000,
    N_b=2000,
    N_k=500,
):
    """Generate training data for Example 5.5 inverse problem."""
    x1_r, x2_r, x3_r = _sample_interior(N_r)
    h_r_exact = source_exact(x1_r, x2_r, x3_r)
    h_r_noisy = h_r_exact + _gaussian_noise(h_r_exact, noise_level)

    x1_c, x2_c, x3_c = _sample_gamma1_x3_eq_0(N_b)
    u_c_exact = u_exact(x1_c, x2_c, x3_c)
    g_c_exact = normal_derivative_gamma1_exact(x1_c, x2_c)
    u_c_noisy = u_c_exact + _gaussian_noise(u_c_exact, noise_level)
    g_c_noisy = g_c_exact + _gaussian_noise(g_c_exact, noise_level)

    x1_g3, x2_g3, x3_g3 = _sample_gamma3_x1_eq_0(N_b)
    u_g3_exact = u_exact(x1_g3, x2_g3, x3_g3)
    u_g3_noisy = u_g3_exact + _gaussian_noise(u_g3_exact, noise_level)

    x1_g4, x2_g4, x3_g4 = _sample_gamma4_x1_eq_1(N_b)
    u_g4_exact = u_exact(x1_g4, x2_g4, x3_g4)
    u_g4_noisy = u_g4_exact + _gaussian_noise(u_g4_exact, noise_level)

    x1_g5, x2_g5, x3_g5 = _sample_gamma5_x2_eq_0(N_b)
    u_g5_exact = u_exact(x1_g5, x2_g5, x3_g5)
    u_g5_noisy = u_g5_exact + _gaussian_noise(u_g5_exact, noise_level)

    x1_g6, x2_g6, x3_g6 = _sample_gamma6_x2_eq_1(N_b)
    u_g6_exact = u_exact(x1_g6, x2_g6, x3_g6)
    u_g6_noisy = u_g6_exact + _gaussian_noise(u_g6_exact, noise_level)

    x1_k, x2_k, x3_k = _sample_gamma2_x3_eq_1(N_k)

    return {
        "residual": (x1_r, x2_r, x3_r, h_r_noisy),
        "cauchy": (x1_c, x2_c, x3_c, u_c_noisy, g_c_noisy),
        "gamma3": (x1_g3, x2_g3, x3_g3, u_g3_noisy),
        "gamma4": (x1_g4, x2_g4, x3_g4, u_g4_noisy),
        "gamma5": (x1_g5, x2_g5, x3_g5, u_g5_noisy),
        "gamma6": (x1_g6, x2_g6, x3_g6, u_g6_noisy),
        "regularization": (x1_k, x2_k, x3_k),
    }


def evaluation_grid(n=56):
    x1 = np.linspace(X1_MIN, X1_MAX, n)
    x2 = np.linspace(X2_MIN, X2_MAX, n)
    x3 = np.linspace(X3_MIN, X3_MAX, n)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing="ij")
    return X1, X2, X3


def evaluation_slice_x3_eq_1(nx1=90, nx2=90):
    x1 = np.linspace(X1_MIN, X1_MAX, nx1)
    x2 = np.linspace(X2_MIN, X2_MAX, nx2)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    X3 = np.ones_like(X1)
    return X1, X2, X3


def evaluation_line_x1_half_x3_one(n=600):
    x2 = np.linspace(X2_MIN, X2_MAX, n)
    x1 = np.full_like(x2, 0.5)
    x3 = np.ones_like(x2)
    return x1, x2, x3
