# Heat Optimal Control 0

This folder is an independent implementation of heat-equation optimal control for manuscript Section `5.1.2`.

Problem:

- `u_t - u_xx = f(x,t)`, on `(x,t) in (0, pi) x (0, 1)`
- `u(0,t) = u(pi,t) = 0`
- `u(x,0) = 0`
- desired terminal state `u_d(x) = sin(x)`

Exact minimum-energy control:

- `f_min(x,t) = g(t) sin(x)`
- `g(t) = A exp(sqrt(2)t) + B exp(-sqrt(2)t) + C exp(t)`
- the code computes `A, B, C` from the `3 x 3` optimality system

Run one solve:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control_0/train.py --alpha-method lcurve
```

Sweep `wJ`:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control_0/sweep_wj.py --alpha-method lcurve
```

Run the unregularized baseline:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control_0/no_reg/train.py
```

Create figures comparing regularized, no-reg, and exact:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control_0/plot_figures.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0`
- `pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0/no_reg`
