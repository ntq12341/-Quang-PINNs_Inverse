from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def save_array_csv(path: Path, name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    out_path = path / f"{name}.csv"

    if arr.ndim == 0:
        np.savetxt(out_path, arr.reshape(1, 1), delimiter=",", fmt="%.12g")
        return

    if arr.ndim == 1:
        np.savetxt(out_path, arr.reshape(-1, 1), delimiter=",", fmt="%.12g")
        return

    if arr.ndim == 2:
        np.savetxt(out_path, arr, delimiter=",", fmt="%.12g")
        return

    flat = arr.reshape(arr.shape[0], -1)
    np.savetxt(out_path, flat, delimiter=",", fmt="%.12g")


def export_npz(npz_path: Path, outdir: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    target_dir = outdir / npz_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["name,shape,dtype,csv_file"]
    for key in data.files:
        arr = np.asarray(data[key])
        save_array_csv(target_dir, key, arr)
        shape_str = "x".join(str(v) for v in arr.shape) if arr.shape else "scalar"
        manifest_lines.append(f"{key},{shape_str},{arr.dtype},{key}.csv")

    (target_dir / "manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Exported {npz_path} -> {target_dir}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Laplace output .npz files to CSV files.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "pinns_optimal_control/outputs/laplace/forward/forward_results.npz",
            "pinns_optimal_control/outputs/laplace/optimal_control/optimal_control_results.npz",
            "pinns_optimal_control/outputs/laplace/dal/dal_laplace_optimal_control.npz",
        ],
    )
    parser.add_argument(
        "--outdir",
        default="pinns_optimal_control/outputs/laplace/csv",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for item in args.inputs:
        npz_path = Path(item)
        if not npz_path.exists():
            print(f"Skip missing file: {npz_path}")
            continue
        export_npz(npz_path, outdir)


if __name__ == "__main__":
    main()

