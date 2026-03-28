# Laplace PINN (Paper Section 3.1)

This folder implements PINN solvers for:

- `3.1.1` forward Laplace problem
- `3.1.2` Laplace optimal control problem

## Run

From repository root:

```bash
python pinns_optimal_control/pinns/laplace/forward_3_1_1.py
python pinns_optimal_control/pinns/laplace/optimal_control_3_1_2.py --wJ 100
python pinns_optimal_control/dal/laplace/train.py
```

Quick smoke test (very short training):

```bash
python pinns_optimal_control/pinns/laplace/forward_3_1_1.py --epochs 20 --print-every 10
python pinns_optimal_control/pinns/laplace/optimal_control_3_1_2.py --epochs 20 --print-every 10 --wJ 100
python pinns_optimal_control/dal/laplace/train.py --max-iters 20 --print-every 5
```

## Outputs

Results are saved to:

- `pinns_optimal_control/outputs/laplace/forward`
- `pinns_optimal_control/outputs/laplace/optimal_control`
- `pinns_optimal_control/outputs/laplace/dal`

Each run saves:

- CSV bundles with `manifest.csv` and one `.csv` per saved array

## Paper-Style Figures

Generate line-search data (for Fig.3a,b):

```bash
python pinns_optimal_control/pinns/laplace/sweep_wj_3_1_2.py
```

Create figure panels like the paper:

```bash
python pinns_optimal_control/pinns/laplace/plot_figures.py
```

Output figure files:

- `pinns_optimal_control/outputs/laplace/figures/fig2_forward_laplace.png`
- `pinns_optimal_control/outputs/laplace/figures/fig3_optimal_laplace.png`

Important:

- the scripts now write results as CSV directories such as
  `pinns_optimal_control/outputs/laplace/optimal_control/optimal_control_results`
- `plot_figures.py` reads those CSV directories directly
