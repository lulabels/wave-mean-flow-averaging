#!/usr/bin/env python3
r"""Nonlinear RSW trajectory for Figure 2 and the Figure 8(d) initial-state analysis.

Figure 2 (N256, nu=2e-4):
    python generate_data/run_rsw_truth.py \
      --N 256 --nu 2e-4 --dt 0.0125 --fast-periods 10 --snapshot-stride 8 \
      --output-dir work/generated_data/fig2_truth

N64 inviscid input for the forcing analysis:
    python generate_data/run_rsw_truth.py \
      --N 64 --nu 0 --dt 0.0125 --fast-periods 10 --snapshot-stride 8 \
      --output-dir work/generated_data/fig8_initial_reference

Other settings are command-line options in main(). Both examples use the
balanced jet and three Poincare waves defined by build_initial_config.
The archived Figure 2 run took 216 s and saved a 117 MB trajectory on the
source system. Build its compact fields with build_compact_rsw_data.py.
The dense N1024 Figure 7-8 history uses run_refined_fig78.py instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wave_averaging.rsw.initial_conditions import geostrophic_residual, linear_residual_norm, make_initial_state, poincare_frequency
from wave_averaging.rsw.projectors import projector_sanity, slow_project_state
from wave_averaging.rsw.solver_pseudospectral import PseudospectralRSWSolver, RSWParameters, RSWState, SpectralGrid


def parse_wave_modes(text: str) -> list[tuple[int, int, float]]:
    modes: list[tuple[int, int, float]] = []
    if not text.strip():
        return modes
    for item in text.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("wave modes must look like 'mx,my,phase;mx,my,phase'")
        modes.append((int(parts[0]), int(parts[1]), float(parts[2])))
    return modes


def build_initial_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "slow_flow": {
            "jet_amplitude": float(args.slow_amplitude),
            "meridional_mode": int(args.slow_meridional_mode),
        },
        "waves": {
            "amplitude": float(args.wave_amplitude),
            "modes": parse_wave_modes(args.wave_modes),
        },
    }


def state_energy(state: RSWState, params: RSWParameters) -> float:
    return float(0.5 * np.mean(params.H * (state.u * state.u + state.v * state.v) + params.g * state.eta * state.eta))


def state_mass_anomaly(state: RSWState) -> float:
    return float(np.mean(state.eta))


def min_depth(state: RSWState, params: RSWParameters) -> float:
    return float(np.min(params.H + state.eta))


def run(args: argparse.Namespace) -> Path:
    start_total = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = SpectralGrid(args.N, Lx=args.Lx, Ly=args.Ly)
    params = RSWParameters(f=args.f, g=args.g, H=args.H, nu=args.nu)
    init_config = build_initial_config(args)
    state0, wave_modes = make_initial_state(init_config, grid, params)

    if min_depth(state0, params) <= args.min_depth_floor:
        raise ValueError(
            f"initial minimum depth {min_depth(state0, params):.6e} is below floor {args.min_depth_floor:.6e}"
        )

    if args.t_end is None:
        if wave_modes:
            omega_ref = float(np.mean([mode.frequency for mode in wave_modes]))
        else:
            parsed_modes = parse_wave_modes(args.wave_modes)
            if parsed_modes:
                omega_ref = float(np.mean([poincare_frequency(mx, my, params, grid) for mx, my, _ in parsed_modes]))
            else:
                omega_ref = abs(params.f)
        fast_period = 2.0 * np.pi / omega_ref
        t_end = float(args.fast_periods) * fast_period
    else:
        omega_ref = float(np.mean([mode.frequency for mode in wave_modes])) if wave_modes else abs(params.f)
        fast_period = 2.0 * np.pi / omega_ref
        t_end = float(args.t_end)

    solver = PseudospectralRSWSolver(grid, params)
    checks: dict[str, float] = {
        "initial_min_depth": min_depth(state0, params),
        "initial_mass_anomaly": state_mass_anomaly(state0),
        "initial_energy": state_energy(state0, params),
        "geostrophic_residual_initial": geostrophic_residual(slow_project_state(state0, grid, params), grid, params),
        **projector_sanity(state0, grid, params),
    }
    for mode in wave_modes:
        checks[f"linear_residual_m{mode.mx}_{mode.my}"] = linear_residual_norm(mode.mx, mode.my, params, grid)

    integrate_start = time.perf_counter()
    print(
        " ".join(
            [
                "running truth trajectory",
                f"N={grid.N}",
                f"dt={args.dt}",
                f"t_end={t_end:.6f}",
                f"snapshot_stride={args.snapshot_stride}",
                f"nu={params.nu}",
            ]
        ),
        flush=True,
    )
    times, samples = solver.integrate(
        state0,
        dt=args.dt,
        t_end=t_end,
        snapshot_stride=args.snapshot_stride,
        progress=(
            lambda step, nsteps, t: print(f"  step {step}/{nsteps}, t={t:.6f}", flush=True)
            if args.progress
            else None
        ),
    )
    integrate_s = time.perf_counter() - integrate_start

    final_state = RSWState.from_array(samples[-1])
    checks.update(
        {
            "final_min_depth": min_depth(final_state, params),
            "final_mass_anomaly": state_mass_anomaly(final_state),
            "final_energy": state_energy(final_state, params),
            "relative_energy_change": (state_energy(final_state, params) - checks["initial_energy"])
            / max(abs(checks["initial_energy"]), 1.0e-14),
            "absolute_mass_anomaly_change": state_mass_anomaly(final_state) - checks["initial_mass_anomaly"],
        }
    )
    if checks["final_min_depth"] <= args.min_depth_floor:
        raise ValueError(
            f"final minimum depth {checks['final_min_depth']:.6e} is below floor {args.min_depth_floor:.6e}"
        )

    runtime = {
        "integrate_s": integrate_s,
        "total_s": time.perf_counter() - start_total,
        "nsteps": int(np.ceil(t_end / args.dt)),
        "snapshot_count": int(len(times)),
    }
    config = {
        "N": args.N,
        "Lx": args.Lx,
        "Ly": args.Ly,
        "params": {"f": args.f, "g": args.g, "H": args.H, "nu": args.nu},
        "dt": args.dt,
        "t_end": t_end,
        "fast_periods": args.fast_periods if args.t_end is None else t_end / fast_period,
        "fast_period": fast_period,
        "snapshot_stride": args.snapshot_stride,
        "initial_condition": init_config,
        "wave_modes": [mode.__dict__ for mode in wave_modes],
        "dtype": args.dtype,
        "kind": "full_nonlinear_2d_rsw_truth_trajectory",
    }

    output_path = output_dir / f"truth_trajectory_N{args.N}.npz"
    sample_dtype = np.float32 if args.dtype == "float32" else np.float64
    np.savez_compressed(
        output_path,
        times=times.astype(np.float64),
        samples=samples.astype(sample_dtype),
        initial_state=state0.as_array().astype(sample_dtype),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        checks_json=np.asarray(json.dumps(checks, sort_keys=True)),
        runtime_json=np.asarray(json.dumps(runtime, sort_keys=True)),
    )
    print(f"saved {output_path}", flush=True)
    print(
        " ".join(
            [
                f"N={grid.N}",
                f"snapshots={len(times)}",
                f"steps={runtime['nsteps']}",
                f"final_min_depth={checks['final_min_depth']:.6e}",
                f"relative_energy_change={checks['relative_energy_change']:.6e}",
                f"runtime_s={runtime['total_s']:.3f}",
            ]
        ),
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full nonlinear 2D RSW truth trajectory.")
    parser.add_argument("--N", type=int, default=64, help="even grid size")
    parser.add_argument("--Lx", type=float, default=2.0 * np.pi, help="domain length in x")
    parser.add_argument("--Ly", type=float, default=2.0 * np.pi, help="domain length in y")
    parser.add_argument("--f", type=float, default=1.0, help="Coriolis parameter")
    parser.add_argument("--g", type=float, default=1.0, help="gravity")
    parser.add_argument("--H", type=float, default=1.0, help="mean depth")
    parser.add_argument("--nu", type=float, default=2.0e-4, help="Laplacian viscosity applied to all fields")
    parser.add_argument("--dt", type=float, default=0.0125, help="RK4 step size")
    parser.add_argument("--t-end", type=float, default=None, help="absolute end time; overrides --fast-periods")
    parser.add_argument("--fast-periods", type=float, default=10.0, help="end time in mean initial Poincare periods")
    parser.add_argument("--snapshot-stride", type=int, default=8, help="save every k RK4 steps")
    parser.add_argument("--slow-amplitude", type=float, default=0.08, help="balanced jet amplitude")
    parser.add_argument("--slow-meridional-mode", type=int, default=1, help="balanced jet meridional mode")
    parser.add_argument("--wave-amplitude", type=float, default=0.018, help="Poincare wave amplitude scale")
    parser.add_argument("--wave-modes", default="4,0,0.0;3,1,1.1;3,-2,2.2", help="mx,my,phase triples")
    parser.add_argument("--min-depth-floor", type=float, default=0.25, help="abort if H+eta falls below this value")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="snapshot storage dtype")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "work" / "truth_trajectory"),
        help="directory for the truth trajectory bundle",
    )
    parser.add_argument("--progress", action="store_true", help="print coarse integration progress")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
