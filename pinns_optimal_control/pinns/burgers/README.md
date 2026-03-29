# Burgers PINN / DAL (Paper Section 3.2)

Implemented:

- `3.2.1` forward Burgers problem with PINN
- `3.2.2` optimal control with PINN
- `3.2.2` optimal control with DAL
- visualization and CSV tables

## Run

```bash
python pinns_optimal_control/pinns/burgers/forward_3_2_1.py
python pinns_optimal_control/pinns/burgers/optimal_control_3_2_2.py --wJ 1
python pinns_optimal_control/dal/burgers/train.py
python pinns_optimal_control/pinns/burgers/sweep_wj_3_2_2.py
python pinns_optimal_control/pinns/burgers/plot_figures.py
```

## Outputs

CSV tables are written under:

- `pinns_optimal_control/outputs/burgers/forward/forward_results`
- `pinns_optimal_control/outputs/burgers/optimal_control/optimal_control_results`
- `pinns_optimal_control/outputs/burgers/dal/dal_burgers_optimal_control`
- `pinns_optimal_control/outputs/burgers/optimal_control/sweep_wj/sweep_wj_results`

Main CSV files:

- `field_u.csv`
- `control_u0.csv`
- `terminal_state.csv`
- `loss_history.csv`
- `history.csv`
- `summary.csv`

