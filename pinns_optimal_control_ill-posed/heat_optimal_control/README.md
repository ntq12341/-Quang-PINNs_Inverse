# Heat Optimal Control PINN

This folder contains the optimal-control part of the heat-equation example, separated from the forward solver.

Problem:

- `u_t - u_xx = f(x,t)`, on `(x,t) in (0, pi) x (0, 1)`
- `u(0,t) = u(pi,t) = 0`
- `u(x,0) = sin(x)`
- terminal target `g(x) = 2 sin(x)`

Analytic optimal pair used in the report:

- `u*(x,t) = sin(x) (1 + t)`
- `f*(x,t) = sin(x) (2 + t)`

Run one optimal-control solve:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control/train.py --alpha-method lcurve
```

Sweep `wJ`:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control/sweep_wj.py --alpha-method lcurve
```

Create figures from generated CSV files:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control/plot_figures.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/heat_optimal_control`
