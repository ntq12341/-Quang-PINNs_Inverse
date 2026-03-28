# DAL Laplace Optimal Control (Section 3.1.2)

This folder contains a discrete-adjoint-looping (DAL) implementation for the
Laplace optimal control problem in the paper.

## Run

From repository root:

```bash
python pinns_optimal_control/dal/laplace/train.py
```

Quick smoke test:

```bash
python pinns_optimal_control/dal/laplace/train.py --max-iters 20 --print-every 5
```

## Output

By default, outputs are saved in:

`pinns_optimal_control/outputs/laplace/dal/dal_laplace_optimal_control.npz`

