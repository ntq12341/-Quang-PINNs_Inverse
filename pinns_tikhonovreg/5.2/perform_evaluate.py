import csv
from pathlib import Path

import numpy as np
import torch


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64).reshape(-1)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def calculate_absolute_error_L1(true_solution, predicted_solution):
    true_np = _to_numpy(true_solution)
    pred_np = _to_numpy(predicted_solution)
    return float(np.mean(np.abs(true_np - pred_np)))


def calculate_max_absolute_error_Linf(true_solution, predicted_solution):
    true_np = _to_numpy(true_solution)
    pred_np = _to_numpy(predicted_solution)
    return float(np.max(np.abs(true_np - pred_np)))


def calculate_l2_norm_error(true_solution, predicted_solution):
    true_np = _to_numpy(true_solution)
    pred_np = _to_numpy(predicted_solution)
    return float(np.sqrt(np.mean((true_np - pred_np) ** 2)))


def calculate_statistical_metrics(true_solution, predicted_solution):
    true_np = _to_numpy(true_solution)
    pred_np = _to_numpy(predicted_solution)
    errors = pred_np - true_np

    rmse = float(np.sqrt(np.mean(errors**2)))
    sde = float(np.std(errors))
    denom = np.mean(np.abs(true_np))
    cv = float((sde / denom) * 100.0) if denom != 0 else 0.0

    return {
        "RMSE": rmse,
        "SDE": sde,
        "CV_percent": cv,
    }


def calculate_all_errors(true_solution, predicted_solution):
    return {
        "L1": calculate_absolute_error_L1(true_solution, predicted_solution),
        "Linf": calculate_max_absolute_error_Linf(true_solution, predicted_solution),
        "L2": calculate_l2_norm_error(true_solution, predicted_solution),
        **calculate_statistical_metrics(true_solution, predicted_solution),
    }


def summarize_suite_errors(all_results):
    """
    Convert output of run_example_52_suite() into a flat table.

    Expected entry keys:
    - noise_level
    - alpha
    - u_top_exact
    - u_top_pred
    """
    rows = []
    for noise_level, result in sorted(all_results.items()):
        errors = calculate_all_errors(result["u_top_exact"], result["u_top_pred"])
        row = {
            "noise": float(noise_level),
            "alpha": float(result.get("alpha", np.nan)),
        }
        row.update(errors)
        rows.append(row)
    return rows


def export_error_tables(all_results, output_dir="results/tables"):
    """
    Export CSV files in format:
    - One file per error type: <error_type>-noise.csv
      columns: noise,error
    - A full summary file: summary-noise.csv
      columns: noise,alpha,L1,Linf,L2,RMSE,SDE,CV_percent
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = summarize_suite_errors(all_results)
    if not rows:
        return []

    summary_headers = ["noise", "alpha", "L1", "Linf", "L2", "RMSE", "SDE", "CV_percent"]
    summary_file = output_path / "summary-noise.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(rows)

    error_types = ["L1", "Linf", "L2", "RMSE", "SDE", "CV_percent"]
    written_files = [str(summary_file)]

    for err_type in error_types:
        file_path = output_path / f"{err_type}-noise.csv"
        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["noise", "error"])
            for row in rows:
                writer.writerow([f"{row['noise']:.6f}", f"{row[err_type]:.12e}"])
        written_files.append(str(file_path))

    return written_files
