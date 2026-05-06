# Heat Optimal Control 0 Without Tikhonov

This variant uses the same Section `5.1.2` problem as `heat_optimal_control_0`, but removes the Tikhonov term.

Loss used in training:

- `L = L_pde + L_bc + L_ic + wJ * L_J`

Problem:

- `u_t - u_xx = f(x,t)` on `(0, pi) x (0, 1)`
- `u(x,0) = 0`
- `u_d(x) = sin(x)`
- exact reference control `f_min(x,t) = g(t) sin(x)`

Run one solve:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control_0/no_reg/train.py
```

Sweep `wJ`:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control_0/no_reg/sweep_wj.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/heat_optimal_control_0/no_reg`
