#!/usr/bin/env python3
r"""Figures 7 and 8: RSW field averages, errors, retained waves and forcing.

    python figures/figures7_8.py --only 7
    python figures/figures7_8.py --only 8

Writes outputs/fig7.png and outputs/fig8.png. Inputs are
    data/fig7_8_rsw_products.npz
    data/fig8_forcing_boxcar.csv
    data/fig8_forcing_bump.csv

The NPZ contains five displayed N1024 states and the plotted error/retention
curves. Fields and scalars are retained without rounding. Its config records
source-history, saved-time and processed-product hashes. The CSVs contain only
the plotted N64 initial-state forcing norms. Input hashes: data/checksums.json.

Simulation settings and averaging: generate_data/run_refined_fig78.py.
Refinement comparisons: generate_data/compare_refined_fig78.py and the included
data/fig7_8_refinement.json. Forcing analysis: generate_data/analyze_rsw_forcing.py.
The legacy history-processing functions below are separate from the N1024 path.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


sys.path.insert(0, str(ROOT))

from wave_averaging.rsw.diagnostics_state import relative_state_error
from wave_averaging.rsw.filters import window_values
from wave_averaging.rsw.modal_backend_compact import CompactModalSystem
from wave_averaging.rsw.projectors import slow_project_state
from wave_averaging.rsw.solver_pseudospectral import RSWParameters, RSWState, SpectralGrid

# Initial modal-amplitude norm for the N64 jet and three-wave setup,
# calculated with CompactModalSystem.state_to_modal(initial_state).
STATE_MODAL_NORM_N64 = 5.913544e-02

# Figure 7 and the error panels use windows centered at t=3.
# Center retention windows at t=8 so T=16 fits within the saved history [0,16].
FIELD_EVALUATION_TIME = 3.0
RETENTION_WINDOWS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
RETENTION_EVALUATION_TIME = 8.0

DEFAULT_DATA = ROOT / "data" / "fig7_8_rsw_products.npz"
DEFAULT_MECHANISM_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

WINDOW_ORDER = ["boxcar", "bump", "hann"]
WINDOW_LABELS = {
    "boxcar": "Boxcar",
    "bump": "Smooth bump",
    "hann": "Hann taper",
}
WINDOW_STYLES = {
    "boxcar": {"color": "0.0", "marker": "o", "linestyle": "-"},
    "bump": {"color": "0.55", "marker": "o", "linestyle": "-"},
    "hann": {"color": "0.62", "marker": "^", "linestyle": "-."},
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.linewidth": 0.85,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 13,
            "figure.titlesize": 13,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique_dns_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, float]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (row["window_name"], float(row["window_length"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def rows_by_window(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {name: [] for name in WINDOW_ORDER}
    for row in rows:
        out.setdefault(row["window_name"], []).append(row)
    for series in out.values():
        series.sort(key=lambda row: float(row["window_length"]))
    return out


def load_mechanism_rows(directory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = list(directory.glob("near_resonance_content_N64_*.csv"))
    paths.extend(directory.glob("fig8_forcing_*.csv"))
    for path in sorted(set(paths)):
        rows.extend(read_csv(path))
    return rows


def load_compact_products(
    path: Path,
) -> tuple[
    dict,
    list[dict[str, float | str]],
    dict[tuple[str, float], dict[str, float]],
    dict[str, np.ndarray],
]:
    """Load the saved fields, error curves and retained-wave fractions."""
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config_json"]))
        rows = json.loads(str(data["comparison_rows_json"]))
        raw_retention = json.loads(str(data["retention_json"]))
        retention = {
            (name, float(window)): values
            for name, by_window in raw_retention.items()
            for window, values in by_window.items()
        }
        states = {
            name: np.asarray(data[f"state_{name}"], dtype=np.float64)
            for name in ("instantaneous", "lab_T1", "wave_T1", "lab_T2", "wave_T2")
        }
    return config, rows, retention, states


def require_inviscid_comparison_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("comparison CSV contains no rows")
    for row in rows:
        fields = [field.strip() for field in row.get("case_label", "").split(",")]
        nu_fields = [field for field in fields if field.startswith("nu=")]
        if len(nu_fields) != 1 or not np.isclose(float(nu_fields[0].split("=", 1)[1]), 0.0):
            raise ValueError("paper-ready RSW figures require an inviscid comparison CSV")


def ageostrophic_rms(state: RSWState, grid: SpectralGrid, params: RSWParameters) -> float:
    """RMS of the ageostrophic (u,v,eta) components, obtained by subtracting P state."""
    slow = slow_project_state(state, grid, params)
    wave = state.as_array() - slow.as_array()
    return float(np.sqrt(np.mean(wave * wave)))


def sample_at(times: np.ndarray, samples: np.ndarray, time: float) -> RSWState:
    index = int(np.argmin(np.abs(times - time)))
    if not np.isclose(times[index], time):
        raise ValueError(f"truth trajectory has no snapshot at t={time:g}")
    return RSWState.from_array(samples[index])


def centered_states(
    *,
    window_name: str,
    window_length: float,
    evaluation_time: float,
    times: np.ndarray,
    samples: np.ndarray,
    system: CompactModalSystem,
) -> tuple[RSWState, RSWState]:
    """Return lab- and wave-frame averages on [t-T/2,t+T/2].

    Interpolate missing endpoints in physical variables before modal conversion.
    This is the legacy np.gradient quadrature path; the selected N1024 products
    use generate_data/run_refined_fig78.py instead.
    """
    T = float(window_length)
    t0 = float(evaluation_time)
    window_start = t0 - 0.5 * T
    window_end = t0 + 0.5 * T
    if window_start < times[0] - 1.0e-12 or window_end > times[-1] + 1.0e-12:
        raise ValueError(f"incomplete centered {window_name} window at t={t0:g}, T={T:g}")

    interior = times[(times > window_start + 1.0e-12) & (times < window_end - 1.0e-12)]
    local_times = np.concatenate(([window_start], interior, [window_end]))

    def interpolate_sample(time: float) -> np.ndarray:
        right = int(np.searchsorted(times, time, side="left"))
        if right < times.size and np.isclose(times[right], time):
            return samples[right]
        if right <= 0 or right >= times.size:
            raise ValueError(f"cannot interpolate truth trajectory at t={time:g}")
        left = right - 1
        alpha = (time - times[left]) / (times[right] - times[left])
        return (1.0 - alpha) * samples[left] + alpha * samples[right]

    local_samples = np.asarray([interpolate_sample(float(time)) for time in local_times])
    raw = window_values(window_name, local_times - window_start, T)
    weights = raw * np.gradient(local_times)
    weights = weights / np.sum(weights)

    lab_frame = RSWState.from_array(np.tensordot(weights, local_samples, axes=(0, 0)))
    interaction_amplitudes = np.empty((local_times.size, len(system.modal_indices)), dtype=np.complex128)
    for j, (time, sample) in enumerate(zip(local_times, local_samples, strict=True)):
        physical_amplitudes = system.state_to_modal(RSWState.from_array(sample))
        interaction_amplitudes[j] = physical_amplitudes * np.exp(1j * system.omega * float(time))
    averaged_amplitudes = np.tensordot(weights, interaction_amplitudes, axes=(0, 0))
    mapped_amplitudes = averaged_amplitudes * np.exp(-1j * system.omega * t0)
    wave_frame = system.modal_to_state(mapped_amplitudes)
    return lab_frame, wave_frame


def compute_centered_dns_products(
    truth_path: Path,
) -> tuple[
    dict,
    list[dict[str, float | str]],
    dict[tuple[str, float], dict[str, float]],
    dict[str, np.ndarray],
]:
    """Recompute centered products from a legacy truth trajectory."""
    with np.load(truth_path, allow_pickle=True) as truth:
        config = json.loads(str(truth["config_json"]))
        times = np.asarray(truth["times"], dtype=np.float64)
        samples = np.asarray(truth["samples"], dtype=np.float64)

    pconf = config["params"]
    if not np.isclose(float(pconf["nu"]), 0.0):
        raise ValueError("paper-ready RSW figures require an inviscid truth trajectory")
    grid = SpectralGrid(int(config["N"]), Lx=float(config["Lx"]), Ly=float(config["Ly"]))
    params = RSWParameters(
        f=float(pconf["f"]),
        g=float(pconf["g"]),
        H=float(pconf["H"]),
        nu=0.0,
    )
    system = CompactModalSystem(grid, params)

    instantaneous = sample_at(times, samples, FIELD_EVALUATION_TIME)
    balanced = slow_project_state(instantaneous, grid, params)
    comparison_rows: list[dict[str, float | str]] = []
    operating_states: dict[tuple[str, float], tuple[RSWState, RSWState]] = {}
    for window_name in ("boxcar", "bump"):
        for window_length in (0.5, 1.0, 2.0):
            lab_frame, wave_frame = centered_states(
                window_name=window_name,
                window_length=window_length,
                evaluation_time=FIELD_EVALUATION_TIME,
                times=times,
                samples=samples,
                system=system,
            )
            operating_states[(window_name, window_length)] = (lab_frame, wave_frame)
            comparison_rows.append(
                {
                    "window_name": window_name,
                    "window_length": window_length,
                    "dns_wave_reference_error_vs_instantaneous_dns": relative_state_error(wave_frame, instantaneous),
                    "dns_lab_physical_window_error_vs_instantaneous_dns": relative_state_error(lab_frame, instantaneous),
                    "dns_wave_reference_error_vs_instantaneous_slow_projection": relative_state_error(wave_frame, balanced),
                    "dns_lab_physical_window_error_vs_instantaneous_slow_projection": relative_state_error(lab_frame, balanced),
                }
            )

    retention_reference = sample_at(times, samples, RETENTION_EVALUATION_TIME)
    instantaneous_wave_rms = ageostrophic_rms(retention_reference, grid, params)
    if instantaneous_wave_rms <= 0.0:
        raise ValueError("instantaneous truth state has zero wave content")
    retention: dict[tuple[str, float], dict[str, float]] = {}
    for window_name in ("boxcar", "bump"):
        for window_length in RETENTION_WINDOWS:
            lab_frame_state, wave_frame_state = centered_states(
                window_name=window_name,
                window_length=window_length,
                evaluation_time=RETENTION_EVALUATION_TIME,
                times=times,
                samples=samples,
                system=system,
            )
            retention[(window_name, window_length)] = {
                "wave_frame": ageostrophic_rms(wave_frame_state, grid, params) / instantaneous_wave_rms,
                "lab_frame": ageostrophic_rms(lab_frame_state, grid, params) / instantaneous_wave_rms,
            }

    velocity_states = {
        "instantaneous": instantaneous.as_array(),
        "lab_T1": operating_states[("boxcar", 1.0)][0].as_array(),
        "wave_T1": operating_states[("boxcar", 1.0)][1].as_array(),
        "lab_T2": operating_states[("boxcar", 2.0)][0].as_array(),
        "wave_T2": operating_states[("boxcar", 2.0)][1].as_array(),
    }
    return config, comparison_rows, retention, velocity_states


def _plain_decimal(y: float, _pos: int | None = None) -> str:
    if y <= 0:
        return ""
    return f"{y:.3g}"


def window_handles(window_names: list[str] | None = None) -> list[Line2D]:
    if window_names is None:
        window_names = WINDOW_ORDER
    return [
        Line2D(
            [0],
            [0],
            linewidth=1.7,
            markersize=5,
            label=WINDOW_LABELS[name],
            **WINDOW_STYLES[name],
        )
        for name in window_names
    ]


def assert_legends_clear_axes(fig: plt.Figure, axes: list[plt.Axes] | np.ndarray) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_boxes = [ax.get_tightbbox(renderer).expanded(1.01, 1.02) for ax in np.asarray(axes).ravel()]
    for legend in fig.legends:
        box = legend.get_window_extent(renderer).expanded(1.02, 1.05)
        for axes_box in axes_boxes:
            if box.overlaps(axes_box):
                raise RuntimeError("figure legend overlaps axes content")


def save(fig: plt.Figure, path: Path, dpi: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_dns_observable_figure(
    rows: list[dict[str, str]],
    mechanism_rows: list[dict[str, str]],
    retention_vs_window: dict[tuple[str, float], dict[str, float]],
    output_dir: Path,
) -> Path:
    r"""Plot state errors, retained-wave amplitudes and initial-state forcing norms.

    Errors are RMS(average-reference)/RMS(reference). Retention is
    RMS((I-P)average)/RMS((I-P)full), an amplitude ratio rather than energy.
    Both averaging frames use the same nonlinear trajectory. Panel (d) uses
    separate N64 initial-state data; refining the trajectory does not refine it.
    """
    with plt.rc_context():
        setup_velocity_matplotlib()
        plt.rcParams["axes.titlesize"] = 19.25
        plt.rcParams["axes.labelsize"] = 18.25
        dns_window_order = ["boxcar", "bump"]
        by_window = rows_by_window(unique_dns_rows(rows))
        # Top: field errors. Bottom: retained wave content and forcing norms.
        fig, axgrid = plt.subplots(2, 2, figsize=(9.6, 10.7), constrained_layout=False)
        ax_a, ax_b, ax_c, ax_d = axgrid[0, 0], axgrid[0, 1], axgrid[1, 0], axgrid[1, 1]
        axes = [ax_a, ax_b, ax_c, ax_d]

        # (a,b) Errors at t=3 for T=0.5, 1, 2.
        for window_name in dns_window_order:
            series = by_window.get(window_name, [])
            if not series:
                continue
            T = [float(row["window_length"]) for row in series]
            style = WINDOW_STYLES[window_name]
            # (a) Full-field reference; solid = wave-frame, dashed = lab-frame.
            axes[0].plot(
                T,
                [float(row["dns_wave_reference_error_vs_instantaneous_dns"]) for row in series],
                linewidth=1.7,
                markersize=4,
                **style,
            )
            axes[0].plot(
                T,
                [float(row["dns_lab_physical_window_error_vs_instantaneous_dns"]) for row in series],
                linewidth=1.7,
                markersize=4,
                linestyle="--",
                marker="o",
                color=style["color"],
            )
            # (b) Balanced-field reference
            axes[1].plot(
                T,
                [float(row["dns_wave_reference_error_vs_instantaneous_slow_projection"]) for row in series],
                linewidth=1.7,
                markersize=4,
                **style,
            )
            axes[1].plot(
                T,
                [float(row["dns_lab_physical_window_error_vs_instantaneous_slow_projection"]) for row in series],
                linewidth=1.7,
                markersize=4,
                linestyle="--",
                marker="o",
                color=style["color"],
            )

        # (c) Retained wave content at t=8
        # At T=2, lab/wave ratios are 0.150533631367/0.969775946301 (boxcar)
        # and 0.216564464676/0.978346659185 (bump). At T=16 the wave ratios
        # are 0.684439518733 and 0.781633106546, respectively.
        for window_name in dns_window_order:
            style = WINDOW_STYLES[window_name]
            Ts = [T for T in RETENTION_WINDOWS if (window_name, T) in retention_vs_window]
            axes[2].plot(
                Ts,
                [retention_vs_window[(window_name, T)]["wave_frame"] for T in Ts],
                linewidth=1.7,
                markersize=4,
                **style,
            )
            axes[2].plot(
                Ts,
                [retention_vs_window[(window_name, T)]["lab_frame"] for T in Ts],
                linewidth=1.7,
                markersize=4,
                linestyle="--",
                marker="o",
                color=style["color"],
            )

        # (d) Forcing norms from the N64 initial-state calculation.
        # Stored ratios agree with the full centered-weight calculation to 6e-10.
        # At T=1, near-sideband ratios are 0.930374746795 (boxcar) and
        # 0.926915931291 (bump); exact-resonant ratios are about 6.34e-9/6.30e-9.
        # These are vector-norm ratios: cancellations prevent additive percentages.
        mech_by_window = rows_by_window(mechanism_rows)
        for window_name in dns_window_order:
            series = mech_by_window.get(window_name, [])
            if not series:
                continue
            T = [float(row["window_length"]) for row in series]
            color = WINDOW_STYLES[window_name]["color"]
            # Markers distinguish detuning groups; grayscale distinguishes kernels.
            axes[3].plot(
                T,
                [float(row["rhs_finite_sideband_same_branch_slow_wave_to_wave_weighted_l2_fraction_of_finite_total"]) for row in series],
                linewidth=1.5, markersize=6, linestyle="-", marker="D", color=color,
            )
            axes[3].plot(
                T,
                [float(row["rhs_finite_high_detuning_ge_4_over_T_weighted_l2_fraction_of_finite_total"]) for row in series],
                linewidth=1.5, markersize=5, linestyle="-", marker="s", color=color,
            )
            axes[3].plot(
                T,
                [float(row["rhs_exact_strict_unweighted_l2_fraction_of_finite_total"]) for row in series],
                linewidth=1.5, markersize=6, linestyle="-", marker="x", color=color,
            )

        axes[0].set_title("Target: full field", pad=12)
        axes[0].set_ylabel("Relative error")
        axes[1].set_title("Target: balanced field", pad=12)
        axes[1].set_ylabel("Relative error")
        axes[2].set_title("Retained wave content", pad=12)
        axes[2].set_ylabel("Fraction retained")
        # Match the error-panel axes; panel (c) spans longer windows.
        ab_major_ticks = [0.001, 0.01, 0.1, 0.5]
        for ax in (axes[0], axes[1]):
            ax.set_yscale("log")
            ax.grid(True, which="both", color="0.85", linewidth=0.7)
            ax.set_xlabel(r"$T$")
            # Leave room above the error curves for the legend.
            ax.set_ylim(0.001, 1.4)
            ax.yaxis.set_major_locator(FixedLocator(ab_major_ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(_plain_decimal))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0), numticks=12))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xticks([0.5, 1.0, 2.0])
            ax.xaxis.set_major_formatter(FuncFormatter(_plain_decimal))
        # Logarithmic T-axis for the longer retention windows
        axes[2].axhline(1.0, color="0.75", linewidth=0.8, linestyle=":", zorder=0)
        axes[2].set_yscale("linear")
        axes[2].grid(True, which="major", color="0.85", linewidth=0.7)
        axes[2].set_ylim(0.0, 1.2)
        axes[2].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[2].set_xlabel(r"$T$")
        axes[2].set_xscale("log")
        axes[2].set_xlim(0.44, 18.0)
        c_xticks = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        axes[2].xaxis.set_major_locator(FixedLocator(c_xticks))
        axes[2].xaxis.set_major_formatter(FuncFormatter(_plain_decimal))
        axes[2].xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=(0.6, 0.7, 0.8, 0.9, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numticks=24)
        )
        axes[2].xaxis.set_minor_formatter(NullFormatter())

        axes[3].set_title("Composition of the forcing", pad=12)
        axes[3].set_ylabel("Normalized forcing magnitude")
        axes[3].set_ylim(-0.03, 1.38)
        axes[3].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[3].grid(True, which="major", color="0.85", linewidth=0.7)
        axes[3].set_xlabel(r"$T$")
        axes[3].set_xticks([0.5, 1.0, 2.0])
        axes[3].xaxis.set_major_formatter(FuncFormatter(_plain_decimal))
        axes[3].set_xlim(0.4, 2.1)
        # Detuning-group legend
        regime_handles = [
            Line2D([0], [0], color="0.35", linestyle="-", linewidth=1.3, marker="x", markersize=6, label="Exact resonance"),
            Line2D([0], [0], color="0.35", linestyle="-", linewidth=1.3, marker="D", markersize=6, label="Near-resonance"),
            Line2D([0], [0], color="0.35", linestyle="-", linewidth=1.3, marker="s", markersize=5, label="Far-detuned"),
        ]
        axes[3].legend(
            handles=regime_handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=1.0,
            fontsize=17.25,
            borderpad=0.05,
            labelspacing=0.05,
            handletextpad=0.4,
            handlelength=1.5,
            borderaxespad=0.1,
        )

        # The lab/wave line-style key applies to (a-c); panel (d) uses markers.
        frame_handles = [
            Line2D([0], [0], color="0.0", linestyle="--", linewidth=1.7, label="Lab-frame"),
            Line2D([0], [0], color="0.0", linestyle="-", linewidth=1.7, label="Wave-frame"),
        ]
        for ax in (axes[0], axes[1], axes[2]):
            ax.legend(
                handles=frame_handles,
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=1.0,
                fontsize=17.25,
                borderpad=0.05,
                labelspacing=0.05,
                handletextpad=0.4,
                handlelength=1.5,
                borderaxespad=0.1,
            )
        # Shared kernel legend
        fig.legend(
            handles=window_handles(dns_window_order),
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
            fontsize=18,
        )
        fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.17, wspace=0.44, hspace=0.42)
        add_velocity_panel_labels(fig, axes, fontsize=19.25)
        assert_legends_clear_axes(fig, axes)
        path = output_dir / "fig8.png"
        save(fig, path)
        return path


def state_speed(state: np.ndarray) -> np.ndarray:
    return np.sqrt(state[0] * state[0] + state[1] * state[1])


def draw_velocity_panel(
    ax: plt.Axes,
    state: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    title: str,
    lx: float,
    ly: float,
    vmax: float,
    stride: int,
) -> None:
    speed = state_speed(state)
    speed_display = np.pad(speed, ((0, 1), (0, 1)), mode="wrap")
    ax.imshow(
        speed_display.T,
        origin="lower",
        extent=(0.0, lx, 0.0, ly),
        vmin=0.0,
        vmax=vmax,
        cmap="Greys",
        alpha=0.55,
        interpolation="nearest",
        aspect="equal",
    )
    sl = slice(None, None, stride)
    ax.quiver(
        X[sl, sl],
        Y[sl, sl],
        state[0, sl, sl],
        state[1, sl, sl],
        color="0.05",
        pivot="mid",
        scale=1.35,
        width=0.0042,
        headwidth=3.0,
        headaxislength=3.0,
        alpha=0.88,
    )
    ax.set_title(title, pad=18)
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0.0, np.pi, 2.0 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0, labelpad=11, va="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)


def setup_velocity_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.linewidth": 0.85,
            "axes.labelsize": 18,
            "axes.titlesize": 18.75,
            "axes.titlepad": 10,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_velocity_panel_labels(fig: plt.Figure, axes: list[plt.Axes] | np.ndarray, fontsize: float = 13) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, ax in zip(list("abcdefgh")[: len(axes)], axes, strict=True):
        bbox = ax.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        title_baseline = ax.title.get_transform().transform(ax.title.get_position())
        label_y = fig.transFigure.inverted().transform(title_baseline)[1]
        fig.text(
            bbox.x0 - 0.020,
            label_y,
            rf"($\it{{{label}}}$)",
            ha="left",
            va="baseline",
            fontsize=fontsize,
        )


def make_velocity_figure(config: dict, states: dict[str, np.ndarray], output_dir: Path, stride: int) -> Path:
    r"""Plot the full state and its centered T=1,2 averages at t=3.

    The compact config fixes figure7_vmax=0.11222557907834244. At N1024 the
    export is 342 dpi; the default N/8 arrow stride keeps the physical density
    unchanged when the grid is refined.
    """
    setup_velocity_matplotlib()
    lx = float(config["Lx"])
    ly = float(config["Ly"])
    n = int(config["N"])
    x = np.linspace(0.0, lx, n, endpoint=False)
    y = np.linspace(0.0, ly, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    instantaneous = states["instantaneous"]
    lab1 = states["lab_T1"]
    wave1 = states["wave_T1"]
    lab2 = states["lab_T2"]
    wave2 = states["wave_T2"]
    vmax = max(
        float(np.percentile(state_speed(s), 99.5))
        for s in (instantaneous, lab1, wave1, lab2, wave2)
    )
    vmax = float(config.get("figure7_vmax", vmax))

    # Top row: full field and T=1 averages. Bottom: T=2 averages.
    # Leave the lower-left panel empty; the full field is the same for both windows.
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 8.0), constrained_layout=False)
    draw_velocity_panel(axes[0, 0], instantaneous, X, Y, title="Full instantaneous field", lx=lx, ly=ly, vmax=vmax, stride=stride)
    draw_velocity_panel(axes[0, 1], lab1, X, Y, title="Lab-frame average", lx=lx, ly=ly, vmax=vmax, stride=stride)
    draw_velocity_panel(axes[0, 2], wave1, X, Y, title="Wave-frame average", lx=lx, ly=ly, vmax=vmax, stride=stride)
    axes[1, 0].axis("off")
    draw_velocity_panel(axes[1, 1], lab2, X, Y, title="", lx=lx, ly=ly, vmax=vmax, stride=stride)
    draw_velocity_panel(axes[1, 2], wave2, X, Y, title="", lx=lx, ly=ly, vmax=vmax, stride=stride)
    for ax in (axes[0, 1], axes[0, 2], axes[1, 2]):
        ax.set_ylabel("")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.09, wspace=0.32, hspace=0.45)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, text in ((axes[0, 0], "a"), (axes[0, 1], "b"), (axes[0, 2], "c"), (axes[1, 1], "d"), (axes[1, 2], "e")):
        bbox = ax.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        title_baseline = ax.title.get_transform().transform(ax.title.get_position())
        label_y = fig.transFigure.inverted().transform(title_baseline)[1]
        fig.text(bbox.x0 - 0.045, label_y, rf"($\it{{{text}}}$)", ha="left", va="baseline", fontsize=18.75)

    path = output_dir / "fig7.png"
    save(fig, path, dpi=max(320, int(np.ceil(n / 3))))
    return path


def run(args: argparse.Namespace) -> None:
    setup_matplotlib()
    output_dir = Path(args.output_dir)
    mechanism_rows = load_mechanism_rows(Path(args.mechanism_dir))
    config, centered_rows, retention_vs_window, velocity_states = load_compact_products(Path(args.data))
    if not np.isclose(float(config["params"]["nu"]), 0.0):
        raise ValueError("manuscript Figures 7--8 require inviscid compact data")

    paths = []
    if args.only in ("all", "8"):
        paths.append(make_dns_observable_figure(centered_rows, mechanism_rows, retention_vs_window, output_dir))
    if args.only in ("all", "7"):
        stride = args.quiver_stride if args.quiver_stride is not None else max(1, int(config["N"]) // 8)
        if stride < 1:
            raise ValueError("quiver stride must be positive")
        paths.append(make_velocity_figure(config, velocity_states, output_dir, stride))
    for path in paths:
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce manuscript Figures 7 and 8 from compact RSW products.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--mechanism-dir", default=str(DEFAULT_MECHANISM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--quiver-stride", type=int, default=None,
                        help="grid stride for arrows (default: eight locations per coordinate)")
    parser.add_argument("--only", choices=["all", "7", "8"], default="all")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
