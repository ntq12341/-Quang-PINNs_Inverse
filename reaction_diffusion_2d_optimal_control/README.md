# 2D Reaction-Diffusion Optimal Control PINN

This folder implements manuscript example 5.3.2, separated from the forward solver.

Problem:

- `u_t - Delta u + c u = f(x,y,t)`, on `(x,y,t) in (0,1)^2 x (0,1)`
- `u = 0` on the spatial boundary
- `u(x,y,0) = sin(pi x) sin(pi y)`
- terminal target `u_d(x,y) = exp(-(c + 2 pi^2)) sin(pi x) sin(pi y)`

Minimum-energy control:

- `f_min = 0`

Run one optimal-control solve:

```bash
python pinns_optimal_control_ill-posed/reaction_diffusion_2d_optimal_control/train.py --alpha-method lcurve
```

Run a quick smoke test:

```bash
python pinns_optimal_control_ill-posed/reaction_diffusion_2d_optimal_control/train.py --device cpu --epochs 5 --alpha-method fixed --alpha 1e-3 --n-residual 200 --batch-residual 100 --n-boundary 40 --n-initial 40 --n-terminal 40 --n-tikhonov 40 --eval-nx 8 --eval-ny 8 --eval-nt 4 --rollout-nx 9 --rollout-ny 9 --rollout-nt 5
```

Outputs are written under:

- `pinns_optimal_control_ill-posed/outputs/reaction_diffusion_2d_optimal_control/optimal_control_results`
