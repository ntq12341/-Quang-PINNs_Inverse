import numpy as np

# Domain: Omega = (0,1) x (0,0.5)
# Gamma1 = {x2=0},  Gamma2 = {x2=0.5}  (inaccessible boundary)
# Gamma3 = {x1=0, x2 in (0,0.5)},  Gamma4 = {x1=1, x2 in (0,0.5)}

X1_MIN, X1_MAX = 0.0, 1.0
X2_MIN, X2_MAX = 0.0, 0.5


# ---------------------------------------------------------------------------
# Exact solution and derived quantities
# ---------------------------------------------------------------------------

def u_exact(x1, x2):
    """Exact solution used in Example 5.6 data generation."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    return x1**2 - x2**2 + 5.0 * x1 + 2.0 * x2 - 3.0 * x1 * x2


def _grad_u_exact(x1, x2):
    """Gradient of exact solution: (du/dx1, du/dx2)."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    du_dx1 = 2.0 * x1 + 5.0 - 3.0 * x2
    du_dx2 = -2.0 * x2 + 2.0 - 3.0 * x1
    return du_dx1, du_dx2


def q_func(u):
    """Nonlinear coefficient: q(u) = 1 + u^2."""
    return 1.0 + u**2


def source_exact(x1, x2):
    """Right-hand side h = -div(q(u)*grad(u)) for the exact solution.

    h(x1,x2) = -2*u_bar * |grad(u_bar)|^2   (given in the paper).
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    u = u_exact(x1, x2)
    du_dx1, du_dx2 = _grad_u_exact(x1, x2)
    return -2.0 * u * (du_dx1**2 + du_dx2**2)


def cauchy_dirichlet_gamma1_exact(x1):
    """Dirichlet data on Gamma1 (x2=0): f(x1) = x1^2 + 5*x1."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.zeros_like(x1)
    return u_exact(x1, x2)


def cauchy_neumann_gamma1_exact(x1):
    """Neumann-like Cauchy data on Gamma1, written exactly as in the paper."""
    x1 = np.asarray(x1, dtype=np.float64)
    return (1.0 + (x1**2 + 5.0 * x1) ** 2) * (3.0 * x1 - 2.0)


def _gaussian_noise(signal, sigma, noise_relative=True):
    """Sample Gaussian noise.

    If noise_relative=True, `sigma` is interpreted as a relative level and
    scaled by RMS(signal), which matches "1%/3%/5% noise" semantics.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if sigma <= 0.0:
        return np.zeros_like(signal)

    scale = float(sigma)
    if noise_relative:
        rms = np.sqrt(np.mean(signal**2))
        scale *= max(float(rms), 1e-12)

    return np.random.normal(0.0, scale, size=signal.shape)


def _sample_interior(n):
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    return x1, x2


def _sample_gamma1_x2_eq_0(n):
    """Cauchy boundary: x2=0."""
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.zeros(n, dtype=np.float64)
    return x1, x2


def _sample_gamma2_x2_eq_05(n):
    """Inaccessible boundary for regularization: x2=0.5."""
    x1 = np.random.uniform(X1_MIN, X1_MAX, n)
    x2 = np.full(n, X2_MAX, dtype=np.float64)
    return x1, x2


def _sample_gamma3_x1_eq_0(n):
    """Dirichlet boundary: x1=0."""
    x1 = np.zeros(n, dtype=np.float64)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    return x1, x2


def _sample_gamma4_x1_eq_1(n):
    """Dirichlet boundary: x1=1."""
    x1 = np.ones(n, dtype=np.float64)
    x2 = np.random.uniform(X2_MIN, X2_MAX, n)
    return x1, x2


def generate_inverse_data_56(
    noise_level=0.01,
    N_r=6000,
    N_b=2000,
    N_k=600,
    noise_relative=True,
):
    """Generate training data for Example 5.6 inverse problem.

    Returns a dict with keys:
      'residual'       : (x1, x2, h_noisy)
      'cauchy'         : (x1, x2, u_noisy, g_noisy)   -- on Gamma1
      'gamma3'         : (x1, x2, u_noisy)             -- Dirichlet on x1=0
      'gamma4'         : (x1, x2, u_noisy)             -- Dirichlet on x1=1
      'regularization' : (x1, x2)                      -- points on Gamma2
    """
    # Interior residual points
    x1_r, x2_r = _sample_interior(N_r)
    h_r_exact = source_exact(x1_r, x2_r)
    h_r_noisy = h_r_exact + _gaussian_noise(
        h_r_exact, noise_level, noise_relative=noise_relative
    )

    # Cauchy data on Gamma1 (x2=0)
    x1_c, x2_c = _sample_gamma1_x2_eq_0(N_b)
    u_c_exact = cauchy_dirichlet_gamma1_exact(x1_c)
    g_c_exact = cauchy_neumann_gamma1_exact(x1_c)
    u_c_noisy = u_c_exact + _gaussian_noise(
        u_c_exact, noise_level, noise_relative=noise_relative
    )
    g_c_noisy = g_c_exact + _gaussian_noise(
        g_c_exact, noise_level, noise_relative=noise_relative
    )

    # Dirichlet on Gamma3 (x1=0)
    x1_g3, x2_g3 = _sample_gamma3_x1_eq_0(N_b)
    u_g3_exact = u_exact(x1_g3, x2_g3)
    u_g3_noisy = u_g3_exact + _gaussian_noise(
        u_g3_exact, noise_level, noise_relative=noise_relative
    )

    # Dirichlet on Gamma4 (x1=1)
    x1_g4, x2_g4 = _sample_gamma4_x1_eq_1(N_b)
    u_g4_exact = u_exact(x1_g4, x2_g4)
    u_g4_noisy = u_g4_exact + _gaussian_noise(
        u_g4_exact, noise_level, noise_relative=noise_relative
    )

    # Regularization points on Gamma2 (x2=0.5)
    x1_k, x2_k = _sample_gamma2_x2_eq_05(N_k)

    return {
        "residual": (x1_r, x2_r, h_r_noisy),
        "cauchy": (x1_c, x2_c, u_c_noisy, g_c_noisy),
        "gamma3": (x1_g3, x2_g3, u_g3_noisy),
        "gamma4": (x1_g4, x2_g4, u_g4_noisy),
        "regularization": (x1_k, x2_k),
    }


# ---------------------------------------------------------------------------
# Evaluation grids
# ---------------------------------------------------------------------------

def evaluation_grid(nx1=90, nx2=45):
    """Full 2D evaluation grid over Omega."""
    x1 = np.linspace(X1_MIN, X1_MAX, nx1)
    x2 = np.linspace(X2_MIN, X2_MAX, nx2)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    return X1, X2


def evaluation_line_x2_half(n=600):
    """Interior line x2=0.5 for comparison with paper Table 6 / Figure 8."""
    x1 = np.linspace(X1_MIN, X1_MAX, n)
    x2 = np.full_like(x1, X2_MAX)
    return x1, x2
