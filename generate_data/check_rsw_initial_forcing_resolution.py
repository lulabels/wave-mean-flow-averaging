#!/usr/bin/env python3
"""Check the spatial resolution of the Figure 8(d) initial-state diagnostic.

Run from any directory, using only the included compact data:
    python generate_data/check_rsw_initial_forcing_resolution.py

The script reconstructs the prescribed band-limited initial field from the
parameters in fig7_8_rsw_products.npz. It evaluates populated modes and their
quadratic outputs across grids, then compares forcing ratios with the archived
Figure 8(d) diagnostic. A failed check exits nonzero.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wave_averaging.rsw.initial_conditions import make_initial_state
from wave_averaging.rsw.modal_reduced import (
    BRANCHES, branch_frequency, finite_window_multiplier,
    modal_matrices, ordered_pair_bilinear,
)
from wave_averaging.rsw.solver_pseudospectral import RSWParameters, SpectralGrid

DEFAULT_PRODUCTS = ROOT / "data/fig7_8_rsw_products.npz"
DEFAULT_FORCING_DIR = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "work/fig8_initial_forcing_resolution"
GROUP_COLUMNS = {
    "exact": "rhs_exact_strict_unweighted_l2_fraction_of_finite_total",
    "sideband": "rhs_finite_sideband_same_branch_slow_wave_to_wave_weighted_l2_fraction_of_finite_total",
    "far": "rhs_finite_high_detuning_ge_4_over_T_weighted_l2_fraction_of_finite_total",
}
AMPLITUDE_TOL = 1.e-12
COEFFICIENT_TOL = 1.e-14
RESONANCE_TOL = 1.e-12
GRID_RELATIVE_TOL = 1.e-10
ARCHIVE_ABSOLUTE_TOL = 1.e-8
SUPPORT_RELATIVE_TOL = 1.e-12
NQUAD = 513


def require_finite(value, description):
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Nonfinite {description}.")


def evaluate_resolution(grids, *, products=DEFAULT_PRODUCTS,
                        forcing_dir=DEFAULT_FORCING_DIR):
    """Return the checks and comparisons; do not write files or build caches."""
    grids = list(grids)
    if len(grids) < 2 or len(set(grids)) != len(grids):
        raise ValueError("Supply at least two distinct grid sizes, including 64.")
    if 64 not in grids:
        raise ValueError("Include N=64 to compare with the archived diagnostic.")
    if any(not isinstance(n, int) or n < 4 for n in grids):
        raise ValueError("Grid sizes must be integers of at least 4.")
    with np.load(products, allow_pickle=False) as z:
        source_config = json.loads(str(z["config_json"]))
    config = {key: source_config[key] for key in
              ("Lx", "Ly", "params", "initial_condition")}
    params = RSWParameters(**config["params"])
    initial = config["initial_condition"]
    wavevectors = [(0, int(initial["slow_flow"]["meridional_mode"]))]
    wavevectors += [(int(m[0]), int(m[1])) for m in initial["waves"]["modes"]]
    support = sorted({(sign*x, sign*y) for x, y in wavevectors for sign in (-1, 1)})
    outputs = sorted({(a[0]+b[0], a[1]+b[1]) for a in support for b in support})
    for n in grids:
        if any(abs(x) >= n/3 or abs(y) >= n/3 for x, y in outputs):
            raise ValueError(f"N={n} does not retain every quadratic output under the strict two-thirds rule.")

    def physical_k(k):
        return (2*np.pi*k[0]/config["Lx"], 2*np.pi*k[1]/config["Ly"])

    @lru_cache(None)
    def basis(k):
        return modal_matrices(*physical_k(k), params)

    @lru_cache(None)
    def weight(kernel, delta, length):
        # Match the archived initial-state diagnostic, not the centered
        # field-history averages in the other panels.
        return finite_window_multiplier(kernel, delta, window_length=length,
                                        epsilon=1.0, nquad=NQUAD,
                                        window_origin="forward")

    def inputs_from_fields(fields):
        require_finite(fields, "initial-state field values")
        n = fields.shape[-1]
        fourier = np.fft.fft2(fields, axes=(-2, -1)) / n**2
        outside = fourier.copy()
        inputs, ignored_sq = [], 0.0
        for k in support:
            outside[:, k[0] % n, k[1] % n] = 0.0
            amplitudes = basis(k)[1] @ fourier[:, k[0] % n, k[1] % n]
            for j, sigma in enumerate(BRANCHES):
                if abs(amplitudes[j]) > AMPLITUDE_TOL:
                    inputs.append((k, sigma, complex(amplitudes[j])))
                else:
                    ignored_sq += abs(amplitudes[j])**2
        total_norm = float(np.linalg.norm(fourier))
        require_finite(total_norm, "initial-state Fourier norm")
        if total_norm == 0:
            raise ValueError("The initial state must be nonzero.")
        return (inputs, float(np.linalg.norm(outside)/total_norm),
                float(np.sqrt(ignored_sq)))

    def calculate(fields):
        inputs, leakage, ignored = inputs_from_fields(fields)
        records = []
        for ka, sa, aa in inputs:
            ra = basis(ka)[0][:, BRANCHES.index(sa)]
            for kb, sb, ab in inputs:
                kc = (ka[0]+kb[0], ka[1]+kb[1])
                rb = basis(kb)[0][:, BRANCHES.index(sb)]
                product = ordered_pair_bilinear(physical_k(ka), ra, physical_k(kb), rb)
                for j, sc in enumerate(BRANCHES):
                    coupling = complex(basis(kc)[1][j] @ product)
                    if abs(coupling) <= COEFFICIENT_TOL:
                        continue
                    delta = (branch_frequency(*physical_k(ka), sa, params)
                             + branch_frequency(*physical_k(kb), sb, params)
                             - branch_frequency(*physical_k(kc), sc, params))
                    records.append((kc + (sc,), sa, sb, delta, coupling*aa*ab))
        rows, full_vectors = {}, {}
        for kernel in ("boxcar", "bump"):
            for length in (0.5, 1.0, 2.0):
                groups = {name: {} for name in ("total", "exact", "sideband", "far")}
                for key, sa, sb, delta, value in records:
                    weighted = value * weight(kernel, delta, length)
                    groups["total"][key] = groups["total"].get(key, 0j) + weighted
                    if abs(delta) <= RESONANCE_TOL:
                        groups["exact"][key] = groups["exact"].get(key, 0j) + value
                    else:
                        sc = key[2]
                        if sc != 0 and ((sa == 0 and sb == sc) or (sb == 0 and sa == sc)):
                            groups["sideband"][key] = groups["sideband"].get(key, 0j) + weighted
                        if abs(delta) >= 4.0/length:
                            groups["far"][key] = groups["far"].get(key, 0j) + weighted
                norms = {name: float(np.linalg.norm(list(vector.values())))
                         for name, vector in groups.items()}
                require_finite(list(norms.values()), "calculated forcing norms")
                if norms["total"] == 0:
                    raise ValueError("The weighted initial forcing must be nonzero.")
                case = f"{kernel}_{length:g}"
                # Each group has its own vector norm, so these ratios are not additive.
                rows[case] = {name: norms[name]/norms["total"]
                              for name in ("exact", "sideband", "far")}
                rows[case]["total_norm"] = norms["total"]
                full_vectors[case] = groups["total"]
        return ({"input_modal_count": len(inputs),
                 "active_ordered_contributions": len(records),
                 "off_support_fourier_relative_l2": leakage,
                 "ignored_on_support_modal_l2": ignored,
                 "ratios": rows}, full_vectors)

    results, baseline_vectors = {}, None
    fields64 = None
    for n in grids:
        grid = SpectralGrid(n, Lx=config["Lx"], Ly=config["Ly"])
        if not all(grid.dealias_mask[x % n, y % n] for x, y in outputs):
            raise ValueError(f"N={n}: the actual dealias mask excludes a quadratic output.")
        state, _ = make_initial_state(initial, grid, params)
        fields = state.as_array()
        result, vectors = calculate(fields)
        if n == 64:
            fields64 = fields
        if baseline_vectors is None:
            baseline_vectors = vectors
        changes = []
        for case, vector in vectors.items():
            base = baseline_vectors[case]
            keys = sorted(set(base) | set(vector))
            v = np.array([vector.get(k, 0j) for k in keys])
            v0 = np.array([base.get(k, 0j) for k in keys])
            changes.append(float(np.linalg.norm(v-v0)/np.linalg.norm(v0)))
        require_finite(changes, "cross-grid forcing differences")
        result["max_weighted_forcing_relative_change_from_reference"] = max(changes)
        results[str(n)] = result

    comparisons = {}
    for kernel in ("boxcar", "bump"):
        path = Path(forcing_dir) / f"fig8_forcing_{kernel}.csv"
        with path.open(newline="") as handle:
            for archived in csv.DictReader(handle):
                case = f"{kernel}_{float(archived['window_length']):g}"
                if case in comparisons:
                    raise ValueError(f"Duplicate archived case: {case}.")
                comparisons[case] = {}
                for name, column in GROUP_COLUMNS.items():
                    value = float(archived[column])
                    require_finite(value, f"archived forcing ratio for {case}/{name}")
                    calculated = results["64"]["ratios"][case][name]
                    difference = abs(value-calculated)
                    require_finite([calculated, difference], "forcing-ratio comparison")
                    comparisons[case][name] = {
                        "archived": value, "ideal_sparse": calculated,
                        "absolute_difference": difference,
                    }
    expected_cases = {f"{k}_{t:g}" for k in ("boxcar", "bump") for t in (0.5, 1.0, 2.0)}
    if set(comparisons) != expected_cases:
        raise ValueError("The archive must contain both kernels and all three displayed windows.")
    max_change = max(r["max_weighted_forcing_relative_change_from_reference"]
                     for r in results.values())
    max_main_difference = max(v[k]["absolute_difference"] for v in comparisons.values()
                              for k in ("sideband", "far"))
    # Estimate float32 leakage from a regenerated initial sample; the archived
    # float32 sample is not included.
    regenerated_float32 = fields64.astype(np.float32).astype(np.float64)
    _, quantized_leakage, _ = inputs_from_fields(regenerated_float32)
    require_finite(quantized_leakage, "regenerated float32 leakage")
    checks = {
        "ideal_support": all(r["off_support_fourier_relative_l2"] < SUPPORT_RELATIVE_TOL
                             for r in results.values()),
        "spatial_resolution": max_change < GRID_RELATIVE_TOL,
        "main_archived_curves": max_main_difference < ARCHIVE_ABSOLUTE_TOL,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "scope": "spatial resolution of prescribed band-limited initial-state forcing",
        "configuration": config, "numpy_version": np.__version__,
        "initial_spatial_support": support,
        "quadratic_output_max_abs": [max(abs(k[0]) for k in outputs),
                                     max(abs(k[1]) for k in outputs)],
        "reference_grid": grids[0], "results": results,
        "tolerances": {"grid_relative": GRID_RELATIVE_TOL,
                       "archive_main_curves_absolute": ARCHIVE_ABSOLUTE_TOL,
                       "ideal_support_relative": SUPPORT_RELATIVE_TOL,
                       "input_amplitude": AMPLITUDE_TOL,
                       "coupling_coefficient": COEFFICIENT_TOL,
                       "exact_resonance": RESONANCE_TOL},
        "window_convention": {"origin": "forward", "quadrature_nodes": NQUAD},
        "max_forcing_relative_change": max_change,
        "max_main_archive_absolute_difference": max_main_difference,
        "regenerated_float32_off_support_fourier_relative_l2": quantized_leakage,
        "archived_ratio_comparisons": comparisons,
        "archive_checked_groups": ["sideband", "far"],
        "archive_reported_only_groups": ["exact"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128, 256, 1024])
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--forcing-dir", type=Path, default=DEFAULT_FORCING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = evaluate_resolution(args.grids, products=args.products,
                                     forcing_dir=args.forcing_dir)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for n, result in report["results"].items():
        change = result["max_weighted_forcing_relative_change_from_reference"]
        print(f"N={n}: populated modes={result['input_modal_count']}; relative forcing change={change:.3g}")
    print(f"Largest main-curve difference from archive: {report['max_main_archive_absolute_difference']:.3g}")
    print("Diagnostic: spatial resolution of prescribed initial-state forcing.")
    print(f"{report['status'].upper()}: {destination}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
