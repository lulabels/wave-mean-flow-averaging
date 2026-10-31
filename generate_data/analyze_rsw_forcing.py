#!/usr/bin/env python3
r"""Initial-state modal forcing grouped by detuning, for Figure 8(d).

Prepare the N64 initial-state input using the example in run_rsw_truth.py.
Then run both kernels:
    python generate_data/analyze_rsw_forcing.py \
      --truth work/generated_data/fig8_initial_reference/truth_trajectory_N64.npz \
      --window-name boxcar --window-lengths 0.5 1 2 \
      --output-dir work/generated_data/fig8_forcing
    python generate_data/analyze_rsw_forcing.py \
      --truth work/generated_data/fig8_initial_reference/truth_trajectory_N64.npz \
      --window-name bump --window-lengths 0.5 1 2 \
      --output-dir work/generated_data/fig8_forcing
    python figures/figures7_8.py --only 8 \
      --mechanism-dir work/generated_data/fig8_forcing

This uses the initial state, separate from the N1024 field-history analysis.
It constructs 51,941,763 ordered interactions; the source caches occupied
3.7 GB. Allow at least 4 GB of cache space and a high-memory workstation or
cluster. Cached source runs took about 60 s per kernel. Caches are not included.
The supplied compact CSVs are enough to replot the panel without this analysis.

For a lightweight check of the ideal initial forcing across spatial grids,
using only the included compact inputs, run:
    python generate_data/check_rsw_initial_forcing_resolution.py
That check uses the populated initial modes without constructing the full archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wave_averaging.rsw.modal_backend_compact import CompactFiniteWindowWeights, CompactModalSystem
from wave_averaging.rsw.solver_pseudospectral import RSWParameters, RSWState, SpectralGrid


def metadata_digest(metadata: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def state_from_sample(sample: np.ndarray) -> RSWState:
    return RSWState.from_array(np.asarray(sample, dtype=np.float64))


def load_truth_system(truth_path: Path, *, resonance_tol: float, chunk_size: int, modal_nu: float) -> tuple[CompactModalSystem, np.ndarray]:
    truth = np.load(truth_path, allow_pickle=True)
    truth_config = json.loads(str(truth["config_json"]))
    grid = SpectralGrid(
        int(truth_config["N"]),
        Lx=float(truth_config["Lx"]),
        Ly=float(truth_config["Ly"]),
    )
    pconf = truth_config["params"]
    params = RSWParameters(f=float(pconf["f"]), g=float(pconf["g"]), H=float(pconf["H"]), nu=float(modal_nu))
    system = CompactModalSystem(grid, params, resonance_tol=resonance_tol, chunk_size=chunk_size)
    initial_state = state_from_sample(np.asarray(truth["initial_state"], dtype=np.float64))
    return system, system.state_to_modal(initial_state)


def load_or_build_interactions(system: CompactModalSystem, cache_dir: Path, rebuild: bool) -> tuple[str, Path]:
    metadata = system.interaction_cache_metadata()
    path = cache_dir / f"compact_interactions_{metadata_digest(metadata)}.npz"
    if path.exists() and not rebuild:
        system.load_interaction_arrays(path)
        return "loaded", path
    _ = system.interaction_arrays
    system.save_interaction_arrays(path)
    return "built", path


def load_or_build_weights(
    system: CompactModalSystem,
    cache_dir: Path,
    rebuild: bool,
    finite_kwargs: dict[str, Any],
) -> tuple[CompactFiniteWindowWeights, str, Path]:
    metadata = system.finite_window_weight_cache_metadata(**finite_kwargs)
    path = cache_dir / f"finite_weights_{metadata_digest(metadata)}.npz"
    if path.exists() and not rebuild:
        return system.load_finite_window_weights(path, **finite_kwargs), "loaded", path
    weights = system.precompute_finite_window_weights(**finite_kwargs)
    system.save_finite_window_weights(path, weights, **finite_kwargs)
    return weights, "built", path


def fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def count_fraction(mask: np.ndarray, active_count: int) -> float:
    if active_count <= 0:
        return 0.0
    return float(np.count_nonzero(mask) / active_count)


def band(delta_abs: np.ndarray, mask: np.ndarray) -> str:
    vals = delta_abs[mask]
    if vals.size == 0:
        return "none"
    return f"[{float(vals.min()):.9g}, {float(vals.max()):.9g}]"


def branch_triple_counts(system: CompactModalSystem, mask: np.ndarray, *, max_rows: int = 20) -> list[dict[str, int]]:
    if np.count_nonzero(mask) == 0:
        return []
    interactions = system.interaction_arrays
    sigma_by_index = np.asarray([idx.sigma for idx in system.modal_indices], dtype=np.int8)
    triples = np.column_stack(
        [
            sigma_by_index[interactions.c_idx[mask]],
            sigma_by_index[interactions.a_idx[mask]],
            sigma_by_index[interactions.b_idx[mask]],
        ]
    )
    unique, counts = np.unique(triples, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:max_rows]
    return [
        {
            "sigma_c": int(unique[i, 0]),
            "sigma_a": int(unique[i, 1]),
            "sigma_b": int(unique[i, 2]),
            "count": int(counts[i]),
        }
        for i in order
    ]


def mode_descriptor(system: CompactModalSystem, flat_index: int) -> dict[str, float | int]:
    idx = system.modal_indices[int(flat_index)]
    return {
        "flat_index": int(idx.flat_index),
        "mx": int(idx.mode.mx),
        "my": int(idx.mode.my),
        "sigma": int(idx.sigma),
        "omega": float(idx.omega),
    }


def rhs_piece_stats(
    system: CompactModalSystem,
    amplitudes: np.ndarray,
    mask: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, float | int]:
    interactions = system.interaction_arrays
    selected = np.flatnonzero(mask)
    rhs = system.zero_amplitudes()
    unsigned_l1 = 0.0
    unsigned_l2_sq = 0.0
    for start in range(0, int(selected.size), system.chunk_size):
        local = selected[start : start + system.chunk_size]
        values = (
            interactions.coefficient[local]
            * amplitudes[interactions.a_idx[local]]
            * amplitudes[interactions.b_idx[local]]
        )
        if weights is not None:
            values = values * weights[local]
        abs_values = np.abs(values)
        unsigned_l1 += float(np.sum(abs_values))
        unsigned_l2_sq += float(np.sum(abs_values * abs_values))
        np.add.at(rhs, interactions.c_idx[local], values)
    rhs_l2 = float(np.linalg.norm(rhs))
    rhs_linf = float(np.max(np.abs(rhs))) if rhs.size else 0.0
    unsigned_l2 = float(np.sqrt(unsigned_l2_sq))
    return {
        "interaction_count": int(selected.size),
        "rhs_l2_norm": rhs_l2,
        "rhs_linf_norm": rhs_linf,
        "unsigned_contribution_l1": unsigned_l1,
        "unsigned_contribution_l2": unsigned_l2,
        "cancellation_ratio_l2_over_l1": fraction(rhs_l2, unsigned_l1),
        "coherence_ratio_l2_over_unsigned_l2": fraction(rhs_l2, unsigned_l2),
    }


def cancellation_aware_rhs_pieces(
    system: CompactModalSystem,
    finite_weights: CompactFiniteWindowWeights,
    *,
    amplitudes0: np.ndarray,
    active: np.ndarray,
    exact: np.ndarray,
    near: np.ndarray,
    delta_abs: np.ndarray,
    window_length: float,
) -> dict[str, dict[str, float | int]]:
    sigma_by_index = np.asarray([idx.sigma for idx in system.modal_indices], dtype=np.int8)
    interactions = system.interaction_arrays
    sigma_c = sigma_by_index[interactions.c_idx]
    sigma_a = sigma_by_index[interactions.a_idx]
    sigma_b = sigma_by_index[interactions.b_idx]
    same_branch_slow_wave_to_wave = (
        near
        & (sigma_c != 0)
        & (((sigma_a == 0) & (sigma_b == sigma_c)) | ((sigma_b == 0) & (sigma_a == sigma_c)))
    )
    high_detuning = near & (delta_abs >= 4.0 / window_length)
    pieces = {
        "exact_strict_unweighted": (exact, None),
        "finite_total_weighted": (active, finite_weights.weights),
        "finite_nonexact_weighted": (near, finite_weights.weights),
        "finite_high_detuning_ge_4_over_T_weighted": (high_detuning, finite_weights.weights),
        "finite_sideband_same_branch_slow_wave_to_wave_weighted": (
            same_branch_slow_wave_to_wave,
            finite_weights.weights,
        ),
    }
    stats = {
        name: rhs_piece_stats(system, amplitudes0, mask, weights=weights)
        for name, (mask, weights) in pieces.items()
    }
    finite_total = float(stats["finite_total_weighted"]["rhs_l2_norm"])
    for values in stats.values():
        values["rhs_l2_fraction_of_finite_total"] = fraction(float(values["rhs_l2_norm"]), finite_total)
    return stats


def top_near_interactions(
    system: CompactModalSystem,
    score: np.ndarray,
    mask: np.ndarray,
    *,
    delta_abs: np.ndarray,
    abs_weight: np.ndarray,
    coeff_abs: np.ndarray,
    amp_pair_abs: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    candidates = np.flatnonzero(mask & np.isfinite(score) & (score > 0.0))
    if candidates.size == 0:
        return []
    k = min(top_k, int(candidates.size))
    if candidates.size > k:
        local = np.argpartition(score[candidates], -k)[-k:]
        candidates = candidates[local]
    candidates = candidates[np.argsort(score[candidates])[::-1]]
    interactions = system.interaction_arrays
    out: list[dict[str, Any]] = []
    for j in candidates:
        out.append(
            {
                "score_state_weighted_absG": float(score[j]),
                "delta": float(interactions.delta[j]),
                "abs_delta": float(delta_abs[j]),
                "abs_transfer": float(abs_weight[j]),
                "abs_coefficient": float(coeff_abs[j]),
                "abs_initial_amplitude_pair": float(amp_pair_abs[j]),
                "c": mode_descriptor(system, int(interactions.c_idx[j])),
                "a": mode_descriptor(system, int(interactions.a_idx[j])),
                "b": mode_descriptor(system, int(interactions.b_idx[j])),
            }
        )
    return out


def summarize_window(
    system: CompactModalSystem,
    finite_weights: CompactFiniteWindowWeights,
    *,
    amplitudes0: np.ndarray,
    window_length: float,
    coeff_tol: float,
    active_thresholds: tuple[float, ...],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    interactions = system.interaction_arrays
    delta_abs = np.abs(interactions.delta)
    coeff_abs = np.abs(interactions.coefficient)
    abs_amp = np.abs(amplitudes0)
    amp_pair_abs = abs_amp[interactions.a_idx] * abs_amp[interactions.b_idx]
    state_strength = coeff_abs * amp_pair_abs
    abs_weight = np.abs(finite_weights.weights)

    active = coeff_abs > coeff_tol
    exact = active & (delta_abs <= system.resonance_tol)
    near = active & ~exact
    active_count = int(np.count_nonzero(active))
    exact_count = int(np.count_nonzero(exact))

    coeff_total = float(np.sum(coeff_abs[active]))
    state_total = float(np.sum(state_strength[active]))
    coeff_exact = float(np.sum(coeff_abs[exact]))
    state_exact = float(np.sum(state_strength[exact]))
    coeff_finite = float(np.sum(coeff_abs[active] * abs_weight[active]))
    state_finite = float(np.sum(state_strength[active] * abs_weight[active]))
    coeff_near_finite = float(np.sum(coeff_abs[near] * abs_weight[near]))
    state_near_finite = float(np.sum(state_strength[near] * abs_weight[near]))

    summary: dict[str, Any] = {
        "window_length": float(window_length),
        "interaction_count": int(interactions.count),
        "active_coefficient_count": active_count,
        "exact_active_count": exact_count,
        "exact_active_count_fraction": count_fraction(exact, active_count),
        "unique_detuning_count": int(finite_weights.unique_delta.size),
        "unique_exact_detuning_count": int(np.count_nonzero(np.abs(finite_weights.unique_delta) <= system.resonance_tol)),
        "coefficient_l1_total": coeff_total,
        "coefficient_l1_exact_fraction": fraction(coeff_exact, coeff_total),
        "coefficient_l1_finite_absG_fraction": fraction(coeff_finite, coeff_total),
        "coefficient_l1_finite_near_absG_fraction": fraction(coeff_near_finite, coeff_total),
        "state_weighted_total": state_total,
        "state_weighted_exact_fraction": fraction(state_exact, state_total),
        "state_weighted_finite_absG_fraction": fraction(state_finite, state_total),
        "state_weighted_finite_near_absG_fraction": fraction(state_near_finite, state_total),
        "near_detuning_band_absG_ge_0p5": band(delta_abs, near & (abs_weight >= 0.5)),
        "near_detuning_band_absG_ge_0p1": band(delta_abs, near & (abs_weight >= 0.1)),
        "max_abs_transfer_nonexact": float(np.max(abs_weight[near])) if np.any(near) else 0.0,
        "max_abs_transfer_high_detuning_ge_4_over_T": float(np.max(abs_weight[active & (delta_abs >= 4.0 / window_length)]))
        if np.any(active & (delta_abs >= 4.0 / window_length))
        else 0.0,
        "top_near_branch_triples_absG_ge_0p5": branch_triple_counts(system, near & (abs_weight >= 0.5)),
    }
    for threshold in active_thresholds:
        label = f"absG_ge_{str(threshold).replace('.', 'p')}"
        kept = active & (abs_weight >= threshold)
        near_kept = near & (abs_weight >= threshold)
        summary[f"{label}_count"] = int(np.count_nonzero(kept))
        summary[f"{label}_count_fraction"] = count_fraction(kept, active_count)
        summary[f"{label}_near_count"] = int(np.count_nonzero(near_kept))
        summary[f"{label}_near_count_fraction"] = count_fraction(near_kept, active_count)
        summary[f"{label}_near_unique_delta_count"] = int(np.unique(interactions.delta[near_kept]).size)
        summary[f"{label}_near_detuning_band"] = band(delta_abs, near_kept)

    rhs_stats = cancellation_aware_rhs_pieces(
        system,
        finite_weights,
        amplitudes0=amplitudes0,
        active=active,
        exact=exact,
        near=near,
        delta_abs=delta_abs,
        window_length=window_length,
    )
    summary["cancellation_aware_rhs_pieces"] = rhs_stats
    for name, values in rhs_stats.items():
        summary[f"rhs_{name}_count"] = int(values["interaction_count"])
        summary[f"rhs_{name}_l2_norm"] = float(values["rhs_l2_norm"])
        summary[f"rhs_{name}_l2_fraction_of_finite_total"] = float(values["rhs_l2_fraction_of_finite_total"])
        summary[f"rhs_{name}_cancellation_l2_over_l1"] = float(values["cancellation_ratio_l2_over_l1"])

    score = state_strength * abs_weight
    top_rows = top_near_interactions(
        system,
        score,
        near,
        delta_abs=delta_abs,
        abs_weight=abs_weight,
        coeff_abs=coeff_abs,
        amp_pair_abs=amp_pair_abs,
        top_k=top_k,
    )
    return summary, top_rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in rows]
    fieldnames: list[str] = []
    for row in scalar_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def run(args: argparse.Namespace) -> Path:
    start = time.perf_counter()
    truth_path = Path(args.truth)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    system, amplitudes0 = load_truth_system(
        truth_path,
        resonance_tol=float(args.resonance_tol),
        chunk_size=int(args.chunk_size),
        modal_nu=float(args.modal_nu),
    )
    if args.max_N is not None and system.grid.N > args.max_N:
        raise ValueError(f"truth N={system.grid.N} exceeds --max-N={args.max_N}")

    interaction_status, interaction_cache = load_or_build_interactions(system, cache_dir, args.rebuild_cache)
    rows: list[dict[str, Any]] = []
    top: dict[str, list[dict[str, Any]]] = {}
    for window_length in args.window_lengths:
        finite_kwargs = {
            "epsilon": float(args.epsilon),
            "window_name": args.window_name,
            "window_length": float(window_length),
            "time_unit": args.time_unit,
            "nquad": int(args.nquad),
        }
        finite_weights, weights_status, weights_cache = load_or_build_weights(
            system,
            cache_dir,
            args.rebuild_cache,
            finite_kwargs,
        )
        summary, top_rows = summarize_window(
            system,
            finite_weights,
            amplitudes0=amplitudes0,
            window_length=float(window_length),
            coeff_tol=float(args.coeff_tol),
            active_thresholds=tuple(float(x) for x in args.active_thresholds),
            top_k=int(args.top_k),
        )
        summary.update(
            {
                "N": int(system.grid.N),
                "window_name": args.window_name,
                "epsilon": float(args.epsilon),
                "time_unit": args.time_unit,
                "nquad": int(args.nquad),
                "finite_weight_cache_status": weights_status,
                "finite_weight_cache": str(weights_cache),
            }
        )
        rows.append(summary)
        top[f"T={float(window_length):g}"] = top_rows

    payload = {
        "kind": "near_resonance_retained_content_for_compact_modal_interactions",
        "truth": str(truth_path),
        "interaction_cache": str(interaction_cache),
        "interaction_cache_status": interaction_status,
        "resonance_tol": float(args.resonance_tol),
        "coefficient_tol": float(args.coeff_tol),
        "active_thresholds": [float(x) for x in args.active_thresholds],
        "frame": "interaction table uses delta=omega_a+omega_b-omega_c; finite transfer magnitude is abs(G_T(-delta)) through CompactModalSystem weights",
        "runtime_s": float(time.perf_counter() - start),
        "rows": rows,
        "top_state_weighted_nonexact_interactions": top,
    }
    output_path = output_dir / f"near_resonance_content_N{system.grid.N}_{args.window_name}.json"
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    write_summary_csv(output_dir / f"near_resonance_content_N{system.grid.N}_{args.window_name}.csv", rows)
    print(f"saved {output_path}", flush=True)
    for row in rows:
        print(
            " ".join(
                [
                    f"T={row['window_length']:g}",
                    f"exact_count_frac={row['exact_active_count_fraction']:.6e}",
                    f"finite_state_frac={row['state_weighted_finite_absG_fraction']:.6e}",
                    f"finite_state_near_frac={row['state_weighted_finite_near_absG_fraction']:.6e}",
                    f"near_absG_ge_0p5={row['absG_ge_0p5_near_count']}",
                ]
            ),
            flush=True,
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exact and finite-window retained modal interactions.")
    parser.add_argument("--truth", required=True, help="truth trajectory npz; its initial state/config define the modal system")
    parser.add_argument("--window-name", default="hann", help="finite-window filter name")
    parser.add_argument("--window-lengths", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="window lengths to analyze")
    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon for finite-window physical-time convention")
    parser.add_argument("--time-unit", choices=["fast", "physical"], default="fast", help="finite-window time unit")
    parser.add_argument("--nquad", type=int, default=513, help="finite-window transfer quadrature nodes")
    parser.add_argument("--modal-nu", type=float, default=0.0, help="viscosity value recorded in modal cache metadata")
    parser.add_argument("--resonance-tol", type=float, default=1.0e-12, help="strict exact-resonance tolerance")
    parser.add_argument("--coeff-tol", type=float, default=1.0e-14, help="ignore smaller interaction coefficients in summaries")
    parser.add_argument("--active-thresholds", type=float, nargs="+", default=[0.5, 0.1, 0.01], help="abs(G) thresholds")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="RHS chunk size recorded with compact system")
    parser.add_argument("--top-k", type=int, default=20, help="number of top state-weighted nonexact interactions to save")
    parser.add_argument("--max-N", type=int, default=64, help="guardrail for all-interaction diagnostics")
    parser.add_argument("--cache-dir", default=None, help="cache directory for compact interactions/finite weights")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--rebuild-cache", action="store_true", help="rebuild matching compact caches")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
