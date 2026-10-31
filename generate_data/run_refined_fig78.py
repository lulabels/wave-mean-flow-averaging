#!/usr/bin/env python3
r"""Dense inviscid RSW trajectory and centered averages for Figures 7-8(a-c).

Generate the selected N1024 history and its plotted products:
    python generate_data/run_refined_fig78.py run \
      --N 1024 --dt 0.00078125 --cadence 0.00625 --t-end 16 \
      --backend fast --workers 16 --output-dir work/generated_data/fig78_truth
    python generate_data/run_refined_fig78.py products \
      --cadence 0.00625 --rule trapezoid --output-dir work/generated_data/fig78_truth

The domain is [0,2*pi]^2, with f=g=H=1, nu=0, strict two-thirds dealiasing
and classical RK4. The selected run has 20,480 steps and 2,561 saved states.
Evolution is float64; streamed samples.npy states are float32; averaging and
compact plotted arrays are float64. The fast real-FFT backend implements the
same RHS/RK4 calculation as the canonical complex-FFT backend.

The source N1024 run took about 2 h 28 min on 16 CPUs and requested 48 GB RAM.
Its history is about 32.22 GB, plus processed arrays and temporary storage.
Choose --workers for the available CPUs and keep BLAS thread counts at one
when using multiple FFT workers. --backend fast requires SciPy.

Build plotted inputs with build_compact_rsw_data.py. Refinement comparisons
are in compare_refined_fig78.py. The independent N64 initial-state
forcing calculation for panel 8(d) is in analyze_rsw_forcing.py.
"""
from __future__ import annotations

import argparse
from functools import cached_property
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

import refined_fig78_support as study
from wave_averaging.rsw.initial_conditions import make_initial_state
from wave_averaging.rsw.solver_pseudospectral import (
    PseudospectralRSWSolver, RSWParameters, SpectralGrid, dealias_state,
)

PKG = study.PKG
ROOT = PKG / "work" / "refined_fig78"
# Balanced jet: (U*cos(y), 0, -U*sin(y)), U=0.08.
# Positive-frequency eigenvectors have eta=1; wave j has amplitude 0.018/sqrt(j),
# j=1,2,3, and the listed phase. Take the real part, then dealias before stepping.
INITIAL = {"slow_flow": {"jet_amplitude": 0.08, "meridional_mode": 1},
           "waves": {"amplitude": 0.018, "modes": [(4, 0, 0.0), (3, 1, 1.1), (3, -2, 2.2)]}}


class CachedGrid(SpectralGrid):
    x = cached_property(SpectralGrid.x.fget)
    y = cached_property(SpectralGrid.y.fget)
    X = cached_property(SpectralGrid.X.fget)
    Y = cached_property(SpectralGrid.Y.fget)
    kx_1d = cached_property(SpectralGrid.kx_1d.fget)
    ky_1d = cached_property(SpectralGrid.ky_1d.fget)
    KX = cached_property(SpectralGrid.KX.fget)
    KY = cached_property(SpectralGrid.KY.fget)
    K2 = cached_property(SpectralGrid.K2.fget)
    dealias_mask = cached_property(SpectralGrid.dealias_mask.fget)


def output_path(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(PKG / name) for name in ("work", "outputs")):
        raise ValueError("outputs must remain under this bundle's work/ or outputs/")
    return path


def run(args):
    out = output_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "samples.npy"
    partial = out / "samples.partial.npy"
    if target.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    stride = round(args.cadence / args.dt)
    nsteps = round(args.t_end / args.dt)
    nsave = round(args.t_end / args.cadence)
    if stride < 1 or not np.isclose(stride * args.dt, args.cadence, rtol=0, atol=1e-12):
        raise ValueError("saved cadence must be an integer multiple of dt")
    if not np.isclose(nsteps * args.dt, args.t_end, rtol=0, atol=1e-12) or nsteps != nsave * stride:
        raise ValueError("end time must lie on both integration and saved time grids")
    backend = getattr(args, "backend", "canonical")
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError("workers must be positive")
    cfg = {"N": args.N, "Lx": float(2 * np.pi), "Ly": float(2 * np.pi),
           "params": {"f": 1.0, "g": 1.0, "H": 1.0, "nu": 0.0},
           "initial_condition": INITIAL, "dt": args.dt, "cadence": args.cadence,
           "t_end": args.t_end, "dtype": "float32", "complete": False,
           "solver_backend": backend, "fft_workers": workers,
           "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
           "source_sha256": {str(p.relative_to(study.REPO)): study.sha256(p) for p in (
               Path(__file__), PKG / "wave_averaging/rsw/solver_pseudospectral.py",
               PKG / "wave_averaging/rsw/initial_conditions.py")}}
    grid, params = CachedGrid(args.N), RSWParameters(nu=0)
    initial, _ = make_initial_state(INITIAL, grid, params)
    state = dealias_state(initial.copy(), grid)
    if backend == "fast":
        import scipy
        from fast_fig78_solver import FastRSWSolver
        cfg["scipy"] = scipy.__version__
        fast_source = Path(__file__).with_name("fast_fig78_solver.py")
        cfg["source_sha256"][str(fast_source.relative_to(study.REPO))] = study.sha256(fast_source)
        solver = FastRSWSolver(grid, params, workers=workers)
    else:
        solver = PseudospectralRSWSolver(grid, params)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    saved = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                     shape=(nsave + 1, 3, args.N, args.N))
    saved[0] = state.as_array()
    times = np.arange(nsave + 1) * args.cadence
    np.save(out / "times.npy", times)
    progress_stride = max(stride, (max(1, nsteps // 10) // stride) * stride)
    start = time.perf_counter()
    print(json.dumps({"N": args.N, "dt": args.dt, "cadence": args.cadence,
                      "steps": nsteps, "saved_states": nsave + 1,
                      "raw_bytes": int(saved.nbytes)}), flush=True)
    for step in range(1, nsteps + 1):
        state = solver.step_rk4(state, args.dt)
        if step % stride == 0:
            arr = state.as_array()
            if not np.isfinite(arr).all() or np.min(params.H + state.eta) <= 0.25:
                raise FloatingPointError(f"invalid state at t={step * args.dt:g}")
            saved[step // stride] = arr
        if step % progress_stride == 0 or step == nsteps:
            elapsed = time.perf_counter() - start
            print(f"step={step}/{nsteps} t={step * args.dt:.6f} elapsed_s={elapsed:.2f}", flush=True)
            saved.flush()
    saved.flush()
    del saved
    partial.rename(target)
    cfg.update(complete=True, runtime_s=time.perf_counter() - start,
               samples_sha256=study.sha256(target), times_sha256=study.sha256(out / "times.npy"))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps({"complete": str(out), "runtime_s": cfg["runtime_s"]}), flush=True)


def window_weights(times, t0, T, kernel, rule):
    r"""Normalized weights for [t0-T/2,t0+T/2], with both endpoints present.

    Boxcar has shape 1; bump has shape exp(-1/(1-(2*s/T)^2)) for |s|<T/2,
    zero outside. The continuous kernel divides this shape by its integral.
    Multiplying the shape by trapezoidal node measures q_i and
    normalizing gives w_i=q_i*b_i/sum(q_j*b_j). On a uniform grid q_i is half
    a cadence at endpoints and one cadence inside. No endpoint interpolation
    is used here. The legacy option uses np.gradient node measures instead.
    """
    left, right = t0 - T / 2, t0 + T / 2
    ix = np.flatnonzero((times >= left - 1e-12) & (times <= right + 1e-12))
    local = times[ix]
    if len(ix) < 2 or not np.isclose(local[0], left) or not np.isclose(local[-1], right):
        raise ValueError("saved nodes must contain both window endpoints")
    if rule == "legacy":
        quadrature = np.gradient(local)
    else:
        intervals = np.diff(local)
        quadrature = np.r_[intervals[0] / 2, (intervals[:-1] + intervals[1:]) / 2, intervals[-1] / 2]
    raw = study.fig.window_values(kernel, local - left, T) * quadrature
    weights = np.zeros(times.size)
    weights[ix] = raw / raw.sum()
    return weights


def average_arrays(times, samples, grid, params, specs, rule, block_size=8):
    r"""Average the same saved nonlinear history in the lab and wave frames.

    The lab frame weights physical states directly. The wave frame removes each
    sample's linear phase, averages, then restores the phase at the window center.
    These are two diagnostics of one trajectory, not separately evolved models.
    """
    system = study.BatchedModalMap(grid, params)
    weights = np.asarray([window_weights(times, *spec, rule) for spec in specs])
    lab = np.zeros((len(specs), 3, grid.N, grid.N))
    wave = np.zeros((len(specs), len(system.modal_indices)), dtype=complex)
    for left in range(0, len(times), block_size):
        right = min(left + block_size, len(times))
        block = np.asarray(samples[left:right], dtype=float)
        local_weights = weights[:, left:right]
        lab += np.tensordot(local_weights, block, axes=(1, 0))
        interaction = np.asarray([
            system.state_to_modal(study.fig.RSWState.from_array(sample)) * np.exp(1j * system.omega * t)
            for sample, t in zip(block, times[left:right])
        ])
        wave += local_weights @ interaction
    return [(study.fig.RSWState.from_array(lab[j]),
             system.modal_to_state(wave[j] * np.exp(-1j * system.omega * spec[0])))
            for j, spec in enumerate(specs)]


def products(args):
    out = output_path(args.output_dir)
    cfg = json.loads((out / "config.json").read_text())
    if not cfg["complete"]:
        raise ValueError("trajectory is incomplete")
    all_times = np.load(out / "times.npy", mmap_mode="r")
    all_samples = np.load(out / "samples.npy", mmap_mode="r")
    stride = round(args.cadence / cfg["cadence"])
    if stride < 1 or not np.isclose(stride * cfg["cadence"], args.cadence):
        raise ValueError("analysis cadence must be an integer multiple of saved cadence")
    times, samples = all_times[::stride], all_samples[::stride]
    grid, params = CachedGrid(cfg["N"]), RSWParameters(**cfg["params"])
    specs = [(t0, T, kernel)
             for t0, windows in ((3.0, (0.5, 1.0, 2.0)), (8.0, study.fig.RETENTION_WINDOWS))
             for kernel in ("boxcar", "bump") for T in windows]
    start = time.perf_counter()
    averages = average_arrays(times, samples, grid, params, specs, args.rule)
    fields, obs = {}, {}
    for t0 in (3.0, 8.0):
        state = study.fig.sample_at(times, samples, t0)
        fields[f"t{t0:g}_instant"] = state.as_array()
        fields[f"t{t0:g}_balanced"] = study.fig.slow_project_state(state, grid, params).as_array()
    for (t0, T, kernel), (lab, wave) in zip(specs, averages):
        instant = study.fig.RSWState.from_array(fields[f"t{t0:g}_instant"])
        balanced = study.fig.RSWState.from_array(fields[f"t{t0:g}_balanced"])
        for frame, state in (("lab", lab), ("wave", wave)):
            key = f"t{t0:g}_{kernel}_T{T:g}_{frame}"
            fields[key] = state.as_array()
            if t0 == 3:
                obs[key + "_error_full"] = study.fig.relative_state_error(state, instant)
                obs[key + "_error_balanced"] = study.fig.relative_state_error(state, balanced)
            else:
                obs[key + "_retention"] = study.fig.ageostrophic_rms(state, grid, params) / study.fig.ageostrophic_rms(instant, grid, params)
    label = f"products_{args.rule}_cadence{args.cadence:g}"
    np.savez_compressed(out / (label + ".npz"), **fields)
    payload = {"config": cfg, "cadence": args.cadence, "quadrature": args.rule,
               "observables": obs, "analysis_s": time.perf_counter() - start,
               "analysis_source_sha256": study.sha256(Path(__file__)),
               "input_identities": {"directory": str(out),
                   "samples_sha256": cfg["samples_sha256"],
                   "times_sha256": cfg["times_sha256"],
                   "config_sha256": study.sha256(out / "config.json")},
               "analysis_dependency_sha256": {str(p.relative_to(study.REPO)): study.sha256(p) for p in (
                   Path(__file__), Path(__file__).with_name("refined_fig78_support.py"),
                   PKG / "figures/figures7_8.py",
                   *sorted((PKG / "wave_averaging/rsw").glob("*.py")),
)}}
    (out / (label + ".json")).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"products": str(out / label), "analysis_s": payload["analysis_s"]}), flush=True)


def self_test():
    rng = np.random.default_rng(20261020)
    grid, cached, params = SpectralGrid(8), CachedGrid(8), RSWParameters(nu=0)
    initial, _ = make_initial_state(INITIAL, grid, params)
    a = dealias_state(initial, grid)
    b = dealias_state(initial, cached)
    for _ in range(20):
        a = PseudospectralRSWSolver(grid, params).step_rk4(a, 0.0125)
        b = PseudospectralRSWSolver(cached, params).step_rk4(b, 0.0125)
    np.testing.assert_array_equal(a.as_array(), b.as_array())
    times = np.arange(81) * 0.0125
    samples = rng.normal(size=(81, 3, 8, 8))
    specs = [(0.5, 0.5, k) for k in ("boxcar", "bump")]
    batched = average_arrays(times, samples, cached, params, specs, "legacy")
    original = study.fig.CompactModalSystem(grid, params)
    for spec, result in zip(specs, batched):
        ref = study.fig.centered_states(window_name=spec[2], window_length=spec[1],
                                       evaluation_time=spec[0], times=times, samples=samples, system=original)
        for x, y in zip(result, ref):
            np.testing.assert_allclose(x.as_array(), y.as_array(), atol=2e-14, rtol=0)
    # Trapezoidal normalization preserves constants and integrates a linear
    # record exactly; its centered harmonic response tends to the boxcar sinc.
    errors = []
    for h in (0.05, 0.025, 0.0125):
        t = np.arange(round(1 / h) + 1) * h
        w = window_weights(t, 0.5, 1.0, "boxcar", "trapezoid")
        np.testing.assert_allclose(w.sum(), 1, atol=1e-15)
        np.testing.assert_allclose(w @ t, 0.5, atol=1e-15)
        errors.append(abs(w @ np.cos(4 * (t - 0.5)) - np.sinc(2 / np.pi)))
    assert errors[0] / errors[1] > 3.9 and errors[1] / errors[2] > 3.9
    print(json.dumps({"cached_rk4_bit_equal": True, "legacy_average_equivalence": True,
                      "boxcar_trapezoid_errors": errors}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "products", "self-test"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pilot")
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--dt", type=float, default=0.00078125)
    parser.add_argument("--cadence", type=float, default=0.00625)
    parser.add_argument("--t-end", type=float, default=16.0)
    parser.add_argument("--rule", choices=("trapezoid", "legacy"), default="trapezoid")
    parser.add_argument("--backend", choices=("canonical", "fast"), default="fast")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.action == "run":
        run(args)
    elif args.action == "products":
        products(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
