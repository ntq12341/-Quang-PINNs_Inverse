# PINN Kuramoto-Sivashinsky (Paper Section 3.3)

Run the forward problem:

```bash
python pinns_optimal_control/pinns/ks/train_forward.py
```

Run the optimal control problem:

```bash
python pinns_optimal_control/pinns/ks/train.py
```

Run the weight sweep used for Fig. 7(a,b):

```bash
python pinns_optimal_control/pinns/ks/sweep_wj_3_3_2.py
```

Run the DAL benchmark:

```bash
python pinns_optimal_control/dal/ks/train.py
```

Create figures from the generated CSV files:

```bash
python pinns_optimal_control/pinns/ks/plot_figures.py
```

Run the full KS pipeline with one command in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File pinns_optimal_control/pinns/ks/run_full_ks.ps1
```
