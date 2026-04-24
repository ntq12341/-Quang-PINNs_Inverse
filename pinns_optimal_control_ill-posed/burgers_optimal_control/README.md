# Burgers PINN Optimal Control Example

This folder implements the optimal-control PINN solver for the Burgers equation on `(x,t) in (0,1) x (0,1)`.

State equation:

- `u_t + u u_x - nu u_xx = f(x,t)`, with `nu = 0.1`
- `u(0,t) = u(1,t) = 0`
- `u(x,0) = 2 nu pi sin(pi x) / (2 + cos(pi x))`

The target state is chosen analytically as:

- `u_o(x,t) = 2 nu pi exp(-t) sin(pi x) / (2 + cos(pi x))`

The terminal target is the step function:

- `g(x) = 0.5` for `x in [0.3, 0.7]`
- `g(x) = 0` otherwise

The true control is defined consistently by:

- `f(x,t) = u_t + u u_x - nu u_xx`, evaluated at `u = u_o`

Run a single optimal-control solve:

```bash
python pinns_optimal_control_ill-posed/burgers_optimal_control/train.py --alpha-method lcurve
```

Sweep `w_J`:

```bash
python pinns_optimal_control_ill-posed/burgers_optimal_control/sweep_wj.py --alpha-method lcurve
```

Create figures:

```bash
python pinns_optimal_control_ill-posed/burgers_optimal_control/plot_figures.py
```
