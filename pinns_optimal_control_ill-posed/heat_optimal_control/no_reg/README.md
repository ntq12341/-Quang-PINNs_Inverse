# Heat Optimal Control PINN Without Tikhonov Regularization

This variant removes the Tikhonov term completely.

Loss used in training:

- `L = L_pde + L_bc + L_ic + wJ * L_J`

There is:

- no `alpha`
- no L-curve search
- no H1 regularization term
- only a sweep over `wJ`

Run one solve:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control/no_reg/train.py
```

Sweep `wJ`:

```bash
python pinns_optimal_control_ill-posed/heat_optimal_control/no_reg/sweep_wj.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/heat_optimal_control/no_reg`
