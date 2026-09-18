# Heat PINN Forward Example

This folder implements the forward PINN solver for the heat-equation example described in `main.pdf`.

Problem:

- `u_t - u_xx = 0`, on `(x,t) in (0, pi) x (0, 1)`
- `u(0,t) = u(pi,t) = 0`
- `u(x,0) = sin(x)`

Analytic solution:

- `u(x,t) = sin(x) exp(-t)`

Run from repository root:

```bash
python pinns_optimal_control_ill-posed/heat/train_forward.py
```

Quick smoke test:

```bash
python pinns_optimal_control_ill-posed/heat/train_forward.py --device cpu --epochs 50 --print-every 10
```

Create figures from the generated CSV files:

```bash
python pinns_optimal_control_ill-posed/heat/plot_figures.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/heat/forward/forward_results`
- `pinns_optimal_control_ill-posed/outputs/heat/figures`
