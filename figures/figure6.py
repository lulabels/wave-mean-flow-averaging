#!/usr/bin/env python3
r"""Figure 6: phase-window dependence for the Figure 5 two-wave model.

    python figures/figure6.py

No data files are needed. Output: ``outputs/fig6.png``.

Panel (a) is the exact fixed-reference horizontal quadratic diagnostic from
Figure 5. Panels (b--d) retain only its centered-boxcar difference-frequency
contribution at ``|Delta omega|T = 2pi/3, 4pi/3, 2pi``. Rapid sum- and
double-frequency terms are deliberately omitted; complete-boxcar arrays are
computed as controls. The background-following phase uses the paper's positive
``+omega*t`` convention, with the same parameters, height, gauge and reference
as Figure 5. Model parameters and analytical checks are in
``wave_averaging.andrews_textbook_two_wave``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT))
from wave_averaging import andrews_textbook_two_wave as model


PHASE_WINDOW_MULTIPLES = np.array([2.0 / 3.0, 4.0 / 3.0, 2.0])
PHASE_TITLES = [
    r"$A_T$, $|\Delta\omega|T=2\pi/3$",
    r"$A_T$, $|\Delta\omega|T=4\pi/3$",
    r"$A_T$, $|\Delta\omega|T=2\pi$",
]


def sinc(value):
    """Return the unnormalised sinc, ``sin(value)/value``."""
    return np.sinc(np.asarray(value, float) / math.pi)


def full_boxcar_for_duration(q, y, duration, p=model.P):
    """Complete centred boxcar control for the horizontal quadratic."""
    result = model.exact_drift(q, y, p.z_ref, p)
    result = result + float(
        sinc(0.5 * p.delta_sigma * duration)
    ) * model.difference_drift(q, y, p.z_ref, 0.0, p)
    result = result + float(
        sinc(0.5 * np.sum(p.sigma) * duration)
    ) * model.sum_drift(q, y, p.z_ref, 0.0, p)
    theta = model.phases(q, p.z_ref, 0.0, p)
    for j in range(2):
        weight = float(sinc(p.sigma[j] * duration))
        coefficient = (
            p.amplitude[j] ** 2
            * p.ell**2
            * math.exp(p.z_ref / p.H)
            * p.k[j]
            / (2.0 * p.sigma[j])
        )
        result[..., 0] += weight * coefficient * np.cos(2.0 * theta[..., j])
    return result


def build_arrays(p=model.P) -> dict[str, np.ndarray]:
    """Build the 26 physical, control and parameter arrays for Figure 6."""
    figure5_fields, _ = model.build_arrays(p)
    x = np.array(figure5_fields["field_q"], copy=True)
    y = np.array(figure5_fields["field_y"], copy=True)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    exact = model.exact_drift(x_grid, y_grid, p.z_ref, p)
    difference = model.difference_drift(x_grid, y_grid, p.z_ref, 0.0, p)

    durations = math.pi * PHASE_WINDOW_MULTIPLES / abs(p.delta_sigma)
    difference_weights = sinc(0.5 * p.delta_sigma * durations)
    selected = np.stack(
        [exact + weight * difference for weight in difference_weights]
    )
    complete_boxcar = np.stack(
        [
            full_boxcar_for_duration(x_grid, y_grid, duration, p)
            for duration in durations
        ]
    )
    one_base_period_selected = exact + p.difference_weight * difference

    return {
        "x": x,
        "y": y,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "exact": exact,
        "difference": difference,
        "durations": durations,
        "durations_in_base_periods": durations / p.duration,
        "difference_weights": difference_weights,
        "selected": selected,
        "complete_boxcar": complete_boxcar,
        "one_base_period_selected": one_base_period_selected,
        "f5_exact_control": figure5_fields["exact_drift"],
        "f5_selected_control": figure5_fields["selected_drift"],
    }


def array_payload(arrays, p=model.P) -> dict[str, np.ndarray]:
    """Add scalar model parameters to the Figure 6 reproducibility arrays."""
    payload = dict(arrays)
    payload.update(
        {
            "parameter_k": p.k,
            "parameter_amplitude": p.amplitude,
            "parameter_sigma_positive_phase_rate": p.sigma,
            "parameter_delta_sigma_positive_phase_rate": np.array(p.delta_sigma),
            "parameter_Ubar": np.array(p.Ubar),
            "parameter_ell": np.array(p.ell),
            "parameter_H": np.array(p.H),
            "parameter_m": np.array(p.m),
            "parameter_f0": np.array(p.f0),
            "parameter_N2": np.array(p.N2),
            "parameter_z_ref": np.array(p.z_ref),
            "parameter_base_period": np.array(p.duration),
        }
    )
    return payload


def validate_arrays(arrays, p=model.P) -> dict[str, float]:
    """Check Figure 5 controls and the three analytic phase windows."""
    expected_durations = np.array(
        [
            2.0 * math.pi / (3.0 * abs(p.delta_sigma)),
            4.0 * math.pi / (3.0 * abs(p.delta_sigma)),
            2.0 * math.pi / abs(p.delta_sigma),
        ]
    )
    expected_weights = np.array(
        [
            3.0 * math.sqrt(3.0) / (2.0 * math.pi),
            3.0 * math.sqrt(3.0) / (4.0 * math.pi),
            0.0,
        ]
    )
    return {
        "duration_max_abs_residual": float(
            np.max(np.abs(arrays["durations"] - expected_durations))
        ),
        "difference_weight_max_abs_residual": float(
            np.max(np.abs(arrays["difference_weights"] - expected_weights))
        ),
        "panel_a_vs_figure5_exact_max_abs_residual": float(
            np.max(np.abs(arrays["exact"] - arrays["f5_exact_control"]))
        ),
        "one_base_period_selected_vs_figure5_max_abs_residual": float(
            np.max(
                np.abs(
                    arrays["one_base_period_selected"]
                    - arrays["f5_selected_control"]
                )
            )
        ),
        "final_selected_minus_exact_max_abs_residual": float(
            np.max(np.abs(arrays["selected"][-1] - arrays["exact"]))
        ),
        "final_complete_boxcar_minus_exact_max_abs_residual": float(
            np.max(np.abs(arrays["complete_boxcar"][-1] - arrays["exact"]))
        ),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.linewidth": 0.9,
            "axes.labelsize": 18,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.family": "Times New Roman",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )


def format_axes(ax: plt.Axes) -> None:
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi])
    ax.set_yticks([0.0, np.pi / 2.0, np.pi])
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, np.pi)


def plot_field(ax, x_grid, y_grid, field, levels) -> None:
    cf = ax.contourf(
        x_grid,
        y_grid,
        field,
        levels=levels,
        cmap="Greys",
        alpha=0.26,
        antialiased=True,
    )
    cf.set_clim(levels[0], levels[-1])
    ax.contour(x_grid, y_grid, field, levels=levels, colors="0.25", linewidths=0.9)
    ax.contour(x_grid, y_grid, field, levels=[0.0], colors="black", linewidths=1.2)
    format_axes(ax)


def render(arrays, destination: Path) -> None:
    """Render the maintained four-panel Figure 6 artwork."""
    configure_style()
    fields = [arrays["exact"][..., 0], *(arrays["selected"][..., 0])]
    amplitude = max(float(np.max(np.abs(field))) for field in fields)
    if amplitude <= 0.0:
        amplitude = 1.0
    levels = np.linspace(-amplitude, amplitude, 15)

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(11.2, 8.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for ax, field in zip(axs.flat, fields):
        plot_field(ax, arrays["x_grid"], arrays["y_grid"], field, levels)

    axs[0, 0].set_title(r"$A_\infty$", fontsize=19, pad=16)
    for ax, title in zip(axs.flat[1:], PHASE_TITLES):
        ax.set_title(title, fontsize=19, pad=16)

    for i in range(2):
        for j in range(2):
            ax = axs[i, j]
            if i == 1:
                ax.set_xlabel(r"$x$")
                ax.set_xticks(
                    [0.0, np.pi, 2.0 * np.pi],
                    labels=["0", r"$\pi$", r"$2\pi$"],
                )
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if j == 0:
                ax.set_ylabel(r"$y$", rotation=0, labelpad=18, va="center")
                ax.set_yticks(
                    [0.0, np.pi / 2.0, np.pi],
                    labels=["0", r"$L/2$", r"$L$"],
                )
            else:
                ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.91,
        bottom=0.09,
        wspace=0.20,
        hspace=0.30,
    )
    fig.canvas.draw()
    panel_labels = [r"($\it{a}$)", r"($\it{b}$)", r"($\it{c}$)", r"($\it{d}$)"]
    for ax, label in zip(axs.flat, panel_labels):
        box = ax.get_position()
        title_baseline = ax.title.get_transform().transform(ax.title.get_position())
        label_y = fig.transFigure.inverted().transform(title_baseline)[1]
        fig.text(
            box.x0 - 0.03,
            label_y,
            label,
            fontsize=19,
            ha="left",
            va="baseline",
        )

    fig.savefig(destination, dpi=320)
    plt.close(fig)


def main(output_path: Path | None = None):
    if output_path is None:
        output_path = ROOT / "outputs" / "fig6.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = build_arrays(model.P)
    checks = validate_arrays(arrays, model.P)
    if checks["duration_max_abs_residual"] > 1.0e-14:
        raise RuntimeError(f"Figure 6 duration check failed: {checks}")
    if checks["difference_weight_max_abs_residual"] > 1.0e-14:
        raise RuntimeError(f"Figure 6 weight check failed: {checks}")
    if checks["panel_a_vs_figure5_exact_max_abs_residual"] != 0.0:
        raise RuntimeError(f"Figure 6 exact-field control failed: {checks}")
    if checks["one_base_period_selected_vs_figure5_max_abs_residual"] != 0.0:
        raise RuntimeError(f"Figure 6 one-period control failed: {checks}")

    render(arrays, output_path)
    print(output_path)
    return arrays, checks


if __name__ == "__main__":
    main()
