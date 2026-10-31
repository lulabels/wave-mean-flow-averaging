#!/usr/bin/env python3
r"""Extract the plotted fields and curves from completed RSW products.

After the trajectories/products have been generated (see run_rsw_truth.py and
run_refined_fig78.py), run from the repository directory:
    python generate_data/build_compact_rsw_data.py --only fig2 \
      --fig2-truth work/generated_data/fig2_truth/truth_trajectory_N256.npz \
      --output-dir work/generated_data
    python generate_data/build_compact_rsw_data.py --only fig78 \
      --fig78-products work/generated_data/fig78_truth/products_trapezoid_cadence0.00625 \
      --figure7-vmax 0.11222557907834244 --output-dir work/generated_data

Replot the newly generated files:
    python figures/figure2.py --data work/generated_data/fig2_rsw_fields.npz
    python figures/figures7_8.py --data work/generated_data/fig7_8_rsw_products.npz

N1024 displayed fields and scalar values are extracted without rounding. Source
history/time/product hashes remain in their metadata. Using a separate output
directory preserves the included compact inputs and their checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

from wave_averaging.rsw.projectors import slow_project_state
from wave_averaging.rsw.solver_pseudospectral import RSWParameters, RSWState, SpectralGrid
import figures7_8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_truth_sample(path: Path, target_time: float) -> tuple[dict, float, RSWState]:
    with np.load(path, allow_pickle=True) as truth:
        config = json.loads(str(truth["config_json"]))
        times = np.asarray(truth["times"], dtype=np.float64)
        index = int(np.argmin(np.abs(times - target_time)))
        if not np.isclose(times[index], target_time):
            raise ValueError(f"{path} has no snapshot at t={target_time:g}")
        state = RSWState.from_array(np.asarray(truth["samples"][index], dtype=np.float64))
    return config, float(times[index]), state


def build_fig2(truth_path: Path, output_dir: Path) -> Path:
    config, time_used, full = load_truth_sample(truth_path, 3.0)
    pconf = config["params"]
    grid = SpectralGrid(int(config["N"]), Lx=float(config["Lx"]), Ly=float(config["Ly"]))
    params = RSWParameters(f=float(pconf["f"]), g=float(pconf["g"]), H=float(pconf["H"]), nu=0.0)
    slow = slow_project_state(full, grid, params)
    wave = RSWState(full.u - slow.u, full.v - slow.v, full.eta - slow.eta)

    def anomaly(state: RSWState) -> np.ndarray:
        eta = np.asarray(state.eta, dtype=np.float64)
        return eta - float(np.mean(eta))

    compact_config = dict(config)
    compact_config["source_trajectory_sha256"] = sha256(truth_path)
    output = output_dir / "fig2_rsw_fields.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        config_json=np.asarray(json.dumps(compact_config, sort_keys=True)),
        time=np.asarray(time_used),
        eta_slow=anomaly(slow),
        eta_full=anomaly(full),
        eta_wave=anomaly(wave),
    )
    print(f"wrote {output}")
    return output


def build_fig78(truth_path: Path, output_dir: Path) -> Path:
    """Legacy NPZ trajectory processing; selected dense data use build_refined_fig78."""
    config, rows, retention, states = figures7_8.compute_centered_dns_products(truth_path)
    compact_config = dict(config)
    compact_config["source_trajectory_sha256"] = sha256(truth_path)
    nested_retention: dict[str, dict[str, dict[str, float]]] = {}
    for (window_name, window_length), values in retention.items():
        nested_retention.setdefault(window_name, {})[f"{window_length:g}"] = values

    output = output_dir / "fig7_8_rsw_products.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        config_json=np.asarray(json.dumps(compact_config, sort_keys=True)),
        comparison_rows_json=np.asarray(json.dumps(rows, sort_keys=True)),
        retention_json=np.asarray(json.dumps(nested_retention, sort_keys=True)),
        **{f"state_{name}": np.asarray(value, dtype=np.float64) for name, value in states.items()},
    )
    print(f"wrote {output}")
    return output


def build_refined_fig78(prefix: Path, output_dir: Path, figure7_vmax: float | None = None) -> Path:
    """Extract the five displayed fields and all plotted scalar values losslessly."""
    metadata_path = Path(str(prefix) + ".json")
    fields_path = Path(str(prefix) + ".npz")
    meta = json.loads(metadata_path.read_text())
    config = dict(meta["config"])
    if not config["complete"] or config["params"]["nu"] != 0.0:
        raise ValueError("Figures 7--8 require a complete inviscid trajectory")
    if meta["quadrature"] != "trapezoid":
        raise ValueError("selected Figures 7--8 use normalized trapezoidal weights")
    obs = meta["observables"]
    mapping = {
        "dns_wave_reference_error_vs_instantaneous_dns": "wave_error_full",
        "dns_lab_physical_window_error_vs_instantaneous_dns": "lab_error_full",
        "dns_wave_reference_error_vs_instantaneous_slow_projection": "wave_error_balanced",
        "dns_lab_physical_window_error_vs_instantaneous_slow_projection": "lab_error_balanced",
    }
    rows = [dict(window_name=kernel, window_length=T,
                 **{key: obs[f"t3_{kernel}_T{T:g}_{value}"] for key, value in mapping.items()})
            for kernel in ("boxcar", "bump") for T in (0.5, 1.0, 2.0)]
    retention = {kernel: {f"{T:g}": {
        f"{frame}_frame": obs[f"t8_{kernel}_T{T:g}_{frame}_retention"]
        for frame in ("lab", "wave")}
        for T in figures7_8.RETENTION_WINDOWS} for kernel in ("boxcar", "bump")}
    keys = {"instantaneous": "t3_instant"}
    keys.update({f"{frame}_T{T}": f"t3_boxcar_T{T}_{frame}"
                 for frame in ("lab", "wave") for T in (1, 2)})
    with np.load(fields_path, allow_pickle=False) as data:
        states = {name: np.asarray(data[key], dtype=np.float64) for name, key in keys.items()}
    if any(a.shape != (3, config["N"], config["N"]) or not np.isfinite(a).all()
           for a in states.values()):
        raise ValueError("invalid plotted field shape or values")
    config.update(analysis_cadence=meta["cadence"], quadrature=meta["quadrature"],
                  source_samples_sha256=meta["input_identities"]["samples_sha256"],
                  source_times_sha256=meta["input_identities"]["times_sha256"],
                  source_products_fields_sha256=sha256(fields_path),
                  source_products_metadata_sha256=sha256(metadata_path))
    if figure7_vmax is not None:
        if not np.isfinite(figure7_vmax) or figure7_vmax <= 0:
            raise ValueError("Figure 7 grayscale bound must be positive and finite")
        config["figure7_vmax"] = figure7_vmax
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fig7_8_rsw_products.npz"
    np.savez_compressed(
        output, config_json=np.asarray(json.dumps(config, sort_keys=True)),
        comparison_rows_json=np.asarray(json.dumps(rows, sort_keys=True)),
        retention_json=np.asarray(json.dumps(retention, sort_keys=True)),
        **{f"state_{name}": value for name, value in states.items()})
    print(f"wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig2-truth", type=Path)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--fig78-truth", type=Path, help="legacy single-file trajectory")
    inputs.add_argument("--fig78-products", type=Path, help="dense-products prefix, without .json/.npz")
    parser.add_argument("--figure7-vmax", type=float, help="fixed grayscale bound from the selected comparison")
    parser.add_argument("--only", choices=["all", "fig2", "fig78"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    if args.only in ("all", "fig2"):
        if args.fig2_truth is None:
            parser.error("--fig2-truth is required for Figure 2")
        build_fig2(args.fig2_truth.resolve(), args.output_dir.resolve())
    if args.only in ("all", "fig78"):
        if args.fig78_products is not None:
            build_refined_fig78(args.fig78_products.resolve(), args.output_dir.resolve(), args.figure7_vmax)
        elif args.fig78_truth is not None:
            build_fig78(args.fig78_truth.resolve(), args.output_dir.resolve())
        else:
            parser.error("--fig78-products or --fig78-truth is required for Figures 7--8")


if __name__ == "__main__":
    main()
