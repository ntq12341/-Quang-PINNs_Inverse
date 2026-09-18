# Burgers PINN Forward Example

This folder implements a forward PINN solver for the Burgers equation on `(x,t) in (0,1) x (0,1)`.

Problem:

- `u_t + u u_x - nu u_xx = 0`, with `nu = 0.1`
- `u(0,t) = u(1,t) = 0`
- `u(x,0) = 2 nu pi sin(pi x) / (2 + cos(pi x))`

Analytic solution used in code:

- `u(x,t) = (2 nu pi exp(-pi^2 nu t) sin(pi x)) / (2 + exp(-pi^2 nu t) cos(pi x))`

Run from repository root:

```bash
python pinns_optimal_control_ill-posed/burgers/train_forward.py
```

Quick smoke test:

```bash
python pinns_optimal_control_ill-posed/burgers/train_forward.py --device cpu --epochs 50 --print-every 10
```

Create figures from the generated CSV files:

```bash
python pinns_optimal_control_ill-posed/burgers/plot_figures.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/burgers/forward/forward_results`
- `pinns_optimal_control_ill-posed/outputs/burgers/figures`
