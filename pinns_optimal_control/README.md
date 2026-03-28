# PINNs Optimal Control (Sections 3.1-3.4)

Code in this folder follows the paper:
"Optimal control of PDEs using physics-informed neural networks"
(JCP 473, 2023, 111731).

Implemented examples:
- `3.1` Laplace
- `3.2` Burgers
- `3.3` Kuramoto-Sivashinsky
- `3.4` Steady Navier-Stokes

For each example there are two methods:
- `pinns/<example>/train.py`
- `dal/<example>/train.py`

Run pattern:

```bash
python pinns_optimal_control/pinns/laplace/train.py
python pinns_optimal_control/dal/laplace/train.py
```

Notes:
- These scripts focus on reproducible research code structure and equations from the paper.
- Default training/iteration counts are reduced so they can run on a normal workstation.
- You can increase epochs/iterations via CLI args for higher accuracy.
