# 2D Reaction-Diffusion PINN Forward Example

This folder implements manuscript example 5.3.1.

Forward problem:

- `u_t - Delta u + c u = 0`, on `(x,y,t) in (0,1)^2 x (0,1)`
- `u = 0` on the spatial boundary
- `u(x,y,0) = sin(pi x) sin(pi y)`

Analytic solution:

- `u(x,y,t) = exp(-(c + 2 pi^2)t) sin(pi x) sin(pi y)`

Run the forward solve:

```bash
python pinns_optimal_control_ill-posed/reaction_diffusion_2d/train_forward.py
```

Run a quick forward smoke test:

```bash
python pinns_optimal_control_ill-posed/reaction_diffusion_2d/train_forward.py --device cpu --epochs 5 --n-residual 200 --batch-residual 100 --n-boundary 40 --n-initial 40 --eval-nx 10 --eval-ny 10 --eval-nt 5
```

Create figures from generated CSV files:

```bash
python pinns_optimal_control_ill-posed/reaction_diffusion_2d/plot_figures.py
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d/forward/forward_results`
- `pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d/figures`
