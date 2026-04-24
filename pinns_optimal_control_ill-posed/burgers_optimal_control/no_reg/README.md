# Burgers Optimal Control PINN Without Tikhonov Regularization

This variant removes the Tikhonov term completely.

The terminal target remains the same step function as the regularized case:

- `g(x) = 0.5` for `x in [0.3, 0.7]`
- `g(x) = 0` otherwise

Loss used in training:

- `L = L_pde + L_bc + L_ic + wJ * L_J`

There is:

- no `alpha`
- no L-curve search
- no H1 regularization term
- only a sweep over `wJ`

Run one solve:

```bash
python pinns_optimal_control_ill-posed/burgers_optimal_control/no_reg/train.py
```

Sweep `wJ`:

```bash
python pinns_optimal_control_ill-posed/burgers_optimal_control/no_reg/sweep_wj.py
```

Create figures:

```bash
python pinns_optimal_control_ill-posed/burgers_optimal_control/no_reg/plot_figures.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/burgers_optimal_control/no_reg`
