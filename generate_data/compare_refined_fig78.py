#!/usr/bin/env python3
r"""Compare two completed RSW products, including unresolved fine-scale tails.

    python generate_data/compare_refined_fig78.py \
      --low work/generated_data/fig78_N512_half/products_trapezoid_cadence0.00625 \
      --high work/generated_data/fig78_truth/products_trapezoid_cadence0.00625 \
      --output work/generated_data/refinement_spatial.json

Inputs are .npz/.json prefixes. See main() for the reference cases and how
to separate grid, timestep, sampling and quadrature changes. Full results
and source identities are in data/fig7_8_refinement.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import refined_fig78_support as study


def load(prefix):
    prefix = Path(prefix).resolve()
    meta = json.loads(Path(str(prefix) + ".json").read_text())
    with np.load(Path(str(prefix) + ".npz"), allow_pickle=False) as data:
        fields = {key: np.asarray(data[key], dtype=float) for key in data.files}
    return meta, (meta["config"], fields, meta["observables"])


def qualitative(obs):
    checks = []
    for kernel in ("boxcar", "bump"):
        for T in (0.5, 1, 2):
            p = f"t3_{kernel}_T{T:g}_"
            checks.extend((obs[p + "wave_error_full"] < obs[p + "lab_error_full"],
                           obs[p + "lab_error_balanced"] < obs[p + "wave_error_balanced"]))
        for T in study.fig.RETENTION_WINDOWS:
            p = f"t8_{kernel}_T{T:g}_"
            checks.append(obs[p + "wave_retention"] > obs[p + "lab_retention"])
    return all(checks)


def main():
    r"""Load two processed cases and write their field/scalar comparisons.

    Reference cases (same initial condition, t_end=16):
        N      dt           saved cadence   backend/workers
        256    0.0015625     0.00625         canonical
        512    0.0015625     0.00625         fast/4
        512    0.00078125    0.00625         fast/4
        1024   0.00078125    0.00625         fast/16

    Backend/RK4 checks are in tests/test_refined_fig78.py. Panel 8(d) uses
    the separate N64 initial-state forcing calculation in analyze_rsw_forcing.py.

    For the matched spatial check, generate the N512 half-step case:
        python generate_data/run_refined_fig78.py run \
          --N 512 --dt 0.00078125 --cadence 0.00625 --t-end 16 \
          --backend fast --workers 4 --output-dir work/generated_data/fig78_N512_half
        python generate_data/run_refined_fig78.py products \
          --cadence 0.00625 --rule trapezoid --output-dir work/generated_data/fig78_N512_half
    Then compare with the N1024 case using the command above.

    For sampling, reprocess the same N1024 history at --cadence 0.0125 and
    0.00625. For quadrature, compare --rule legacy and --rule trapezoid at
    fixed sampling. Product names distinguish these cases. Raw histories and
    larger processed archives must be regenerated. Field errors include
    unresolved tails, including float32 off-mask contributions, and use the
    high-reference RMS as denominator.

    The included results give a maximum N512-to-N1024 retained-fraction change
    of 0.366639 percentage points and early t=3 field difference of 0.00009086%.
    The instantaneous t=8 state still differs by 7.61487% RMS; short-window
    wave-frame fields also differ by several percent. All plotted orderings
    are preserved across these comparisons.

    Halving dt at N512 changes retained fractions by at most 0.002681 percentage
    points. N1024 sampling refinement changes them by at most 0.013147 percentage
    points and fields by 0.062907% RMS. The original-to-selected retention change
    reaches about 6.27 percentage points while changing grid, dt, sampling and
    quadrature together; it cannot be assigned to grid refinement alone.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True, help="path prefix, without .json/.npz")
    parser.add_argument("--high", type=Path, required=True, help="path prefix, without .json/.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(output.is_relative_to(study.PKG / name) for name in ("work", "outputs")):
        raise ValueError("comparison output must remain in this bundle's work/ or outputs/")
    lm, low = load(args.low)
    hm, high = load(args.high)
    result = study.compare(low, high)
    result["changed_factors"] = {
        key: [a, b] for key, a, b in (
            ("N", lm["config"]["N"], hm["config"]["N"]),
            ("dt", lm["config"]["dt"], hm["config"]["dt"]),
            ("saved_cadence", lm["config"]["cadence"], hm["config"]["cadence"]),
            ("analysis_cadence", lm["cadence"], hm["cadence"]),
            ("quadrature", lm["quadrature"], hm["quadrature"]),
            ("solver_backend", lm["config"].get("solver_backend", "canonical"), hm["config"].get("solver_backend", "canonical")),
            ("fft_workers", lm["config"].get("fft_workers", 1), hm["config"].get("fft_workers", 1)),
        ) if a != b
    }
    result["qualitative_orderings_preserved"] = {"low": qualitative(low[2]), "high": qualitative(high[2])}
    result["inputs"] = {name: {"prefix": str(prefix.resolve()),
                               "metadata_sha256": study.sha256(Path(str(prefix) + ".json")),
                               "fields_sha256": study.sha256(Path(str(prefix) + ".npz"))}
                        for name, prefix in (("low", args.low), ("high", args.high))}
    result["metadata"] = {"low": lm, "high": hm}
    result["comparison_source_sha256"] = study.sha256(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key.startswith("max_") or key == "changed_factors"}))


if __name__ == "__main__":
    main()
