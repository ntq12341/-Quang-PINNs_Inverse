import math
import numpy as np

X1_MIN, X1_MAX = 0.0, math.pi
X2_MIN, X2_MAX = 0.0, 1.0


def u_exact(x, y):
    """Exact solution: u(x1, x2) = sin(x1) cosh(x2)."""
    return np.sin(x) * np.cosh(y)


def v_smooth(x):
    """Top boundary condition: v(x1) = sin(x1)*cosh(1)."""
    return np.sin(x) * np.cosh(1.0)


def _gaussian_noise(signal, noise_level):
    """Gaussian noise: sigma * xi, xi ~ N(0,1)."""
    return np.random.normal(0.0, noise_level, size=signal.shape)


def generate_forward_data(v_top_func=None, N_r=10000, N_b=1000):
    """
    Bài toán forward (5.4):
      u_xx + u_yy = 0
      u(0,y) = u(pi,y) = 0    → left, right  (Dirichlet)
      u(x,1) = v(x)           → top           (Dirichlet, cho trước)
      du/dy(x,0) = 0          → bottom        (Neumann, BC của forward)
    """
    if v_top_func is None:
        v_top_func = v_smooth

    x_r = np.random.uniform(X1_MIN, X1_MAX, N_r)
    y_r = np.random.uniform(X2_MIN, X2_MAX, N_r)

    x_b_left  = np.zeros(N_b)
    y_b_left  = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_right = np.full(N_b, X1_MAX)
    y_b_right = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_top = np.random.uniform(X1_MIN, X1_MAX, N_b)
    y_b_top = np.full(N_b, X2_MAX)
    u_b_top = v_top_func(x_b_top)

    x_b_bottom = np.random.uniform(X1_MIN, X1_MAX, N_b)
    y_b_bottom = np.zeros(N_b)

    return {
        "residual": (x_r, y_r),
        "left":     (x_b_left,   y_b_left),
        "right":    (x_b_right,  y_b_right),
        "top":      (x_b_top,    y_b_top, u_b_top),
        "bottom":   (x_b_bottom, y_b_bottom),  # Neumann=0
    }


def generate_inverse_data(
    u_bottom_func,
    noise_level=0.01,
    N_r=5000,
    N_b=1000,
    N_cauchy=1000,
):
    """
    Bài toán inverse (5.6) — đúng theo paper:
      u_xx + u_yy = 0
      u(0,y) = u(pi,y) = 0    → left, right   (Dirichlet, biết trước)
      u(x,0) = u_delta(x)     → cauchy         (Cauchy data DUY NHẤT, có noise)

    KHÔNG enforce Neumann du/dy(x,0)=0 trong inverse:
      Bài toán (5.6) trong paper chỉ cho u(x,0) = u^delta làm input.
      du/dy(x,0)=0 là BC của bài toán forward (5.4), không phải thông tin
      đã biết trong inverse. Nếu enforce thêm Neumann → đủ Cauchy data
      (Dirichlet + Neumann tại y=0) → bài toán well-posed → network học tốt
      dù không có regularization → đây chính là nguồn gốc của "kết quả tốt
      bất hợp lý". Bỏ Neumann ra mới tái hiện đúng ill-posedness của paper.
    """
    x_r = np.random.uniform(X1_MIN, X1_MAX, N_r)
    y_r = np.random.uniform(X2_MIN, X2_MAX, N_r)

    x_b_left  = np.zeros(N_b)
    y_b_left  = np.random.uniform(X2_MIN, X2_MAX, N_b)

    x_b_right = np.full(N_b, X1_MAX)
    y_b_right = np.random.uniform(X2_MIN, X2_MAX, N_b)

    # Cauchy Dirichlet tại y=0: dữ liệu đo DUY NHẤT, có noise
    x_cauchy = np.random.uniform(X1_MIN, X1_MAX, N_cauchy)
    y_cauchy = np.zeros(N_cauchy)
    u_c_exact = np.asarray(u_bottom_func(x_cauchy))
    u_c_noisy = u_c_exact + _gaussian_noise(u_c_exact, noise_level)

    return {
        "residual": (x_r, y_r),
        "left":     (x_b_left,  y_b_left),
        "right":    (x_b_right, y_b_right),
        # "bottom" bị loại bỏ — không enforce Neumann trong inverse problem
        "cauchy":   (x_cauchy, y_cauchy, u_c_noisy),
    }