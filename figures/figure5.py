#!/usr/bin/env python3
r"""Figure 5: two-wave Rossby displacement and drift.

    python figures/figure5.py

No data files are needed. Output: ``outputs/fig5.png``.

The base mode and its Figure 3 phase, centres, background current and height
are preserved. A weaker dispersion-consistent mode is added in the coordinate
moving with that current; its real phase uses the paper's positive ``+omega*t``
convention. Panels (b,c) show fixed-reference, first-order displacement from
both waves plus a separately enlarged fixed-reference horizontal drift
diagnostic. Solid curves retain the sinc-weighted difference-frequency term;
dashed curves contain the exact self terms only.

Model parameters and analytical checks are in
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
from matplotlib.patches import FancyArrowPatch
import numpy as np

sys.path.insert(0, str(ROOT))
from wave_averaging import andrews_textbook_two_wave as model


def orbit_axes(
    ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_family(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str = "0.2",
    linestyle: str = "-",
    linewidth: float = 1.2,
    arrows: bool = True,
) -> None:
    i0 = x.shape[1] // 8
    i1 = i0 + round(10 * x.shape[1] / 400)
    for xx, yy in zip(x, y):
        ax.plot(xx, yy, color=color, linestyle=linestyle, linewidth=linewidth)
        if arrows:
            ax.annotate(
                "",
                xy=(xx[i1], yy[i1]),
                xytext=(xx[i0], yy[i0]),
                arrowprops={"arrowstyle": "->", "lw": 0.85, "color": "0.2"},
            )


def add_panel_a(ax: plt.Axes, fields: dict[str, np.ndarray]) -> None:
    q = fields["field_q"]
    y = fields["field_y"]
    values = fields["streamfunction"]
    q_grid, y_grid = np.meshgrid(q, y, indexing="xy")
    amp = float(np.max(np.abs(values))) + 1e-30
    levels = np.linspace(-amp, amp, 15)
    cf = ax.contourf(
        q_grid,
        y_grid,
        values,
        levels=levels,
        cmap="Greys",
        alpha=0.26,
        antialiased=True,
    )
    ax.contour(
        q_grid,
        y_grid,
        values,
        levels=levels,
        colors="black",
        linewidths=0.9,
        linestyles="solid",
    )
    cf.set_clim(-amp, amp)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0, labelpad=18, va="center")
    ax.set_xticks([0.0, math.pi, 2.0 * math.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0.0, math.pi / 2.0, math.pi], labels=["0", r"$L/2$", r"$L$"])
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_ylim(0.0, math.pi)


def add_panel_b_drifted_loops(
    ax: plt.Axes, orbits: dict[str, np.ndarray]
) -> None:
    plot_family(
        ax,
        orbits["b_exact_x"],
        orbits["b_exact_y"],
        color="0.55",
        linestyle="--",
        linewidth=1.0,
        arrows=False,
    )
    plot_family(
        ax,
        orbits["b_selected_x"],
        orbits["b_selected_y"],
        color="0.15",
        linestyle="-",
        linewidth=1.2,
        arrows=True,
    )
    orbit_axes(ax, (-0.26, 0.26), (0.0, 1.0))


def add_panel_c_orbit_loops_yz(
    ax: plt.Axes, orbits: dict[str, np.ndarray]
) -> None:
    plot_family(
        ax,
        orbits["c_selected_y"],
        orbits["c_reference_z"],
        color="0.15",
        linestyle="-",
        linewidth=1.25,
        arrows=True,
    )
    plot_family(
        ax,
        orbits["c_exact_y"],
        orbits["c_reference_z"],
        color="0.60",
        linestyle="--",
        linewidth=1.05,
        arrows=False,
    )
    orbit_axes(ax, (0.03, 0.97), (0.31, 0.73))


def add_drift_panel(
    ax: plt.Axes,
    fields: dict[str, np.ndarray],
    key: str,
    title: str | None = None,
) -> None:
    q = fields["field_q"]
    y = fields["field_y"]
    q_grid, y_grid = np.meshgrid(q, y, indexing="xy")
    values = fields[key][..., 0]
    amp = float(np.max(np.abs(values))) + 1e-30
    levels = np.linspace(-amp, amp, 15)
    cf = ax.contourf(
        q_grid,
        y_grid,
        values,
        levels=levels,
        cmap="Greys",
        alpha=0.26,
        antialiased=True,
    )
    cf.set_clim(-amp, amp)
    ax.contour(q_grid, y_grid, values, levels=levels, colors="0.25", linewidths=0.9)
    ax.contour(q_grid, y_grid, values, levels=[0.0], colors="black", linewidths=1.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0, labelpad=18, va="center")
    ax.set_xticks([0.0, math.pi, 2.0 * math.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0.0, math.pi / 2.0, math.pi], labels=["0", r"$L/2$", r"$L$"])
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_ylim(0.0, math.pi)
    if title:
        ax.set_title(title, fontsize=19, pad=10)


def add_transition_annotation(
    fig: plt.Figure,
    ax_b: plt.Axes,
    ax_c: plt.Axes,
    ax_d: plt.Axes,
) -> None:
    bb = ax_b.get_position()
    bc = ax_c.get_position()
    bd = ax_d.get_position()
    x_mid = 0.5 * (bd.x0 + bd.x1)
    y_gap = float(min(bb.y0, bc.y0) - bd.y1)
    pad = min(0.012, 0.25 * y_gap)
    y_top = min(bb.y0, bc.y0) - pad
    y_bottom = bd.y1 + pad
    arrow = FancyArrowPatch(
        (x_mid, y_top),
        (x_mid, y_bottom),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="->",
        mutation_scale=11,
        linewidth=2.2,
        color="black",
        transform=fig.transFigure,
        zorder=20,
    )
    fig.add_artist(arrow)
    fig.text(
        x_mid + 0.016,
        0.5 * (y_top + y_bottom),
        r"$A_\infty$",
        ha="left",
        va="center",
        fontsize=19,
    )


def enlarge_orbit_panels(ax_b: plt.Axes, ax_c: plt.Axes) -> None:
    """Magnify complete orbit panels for print readability without changing data."""
    for ax, center_x in ((ax_b, 0.245), (ax_c, 0.715)):
        box = ax.get_position()
        width = 1.45 * box.width
        height = 1.45 * box.height
        ax.set_position(
            [
                center_x - width / 2.0,
                box.y0 - (height - box.height) / 2.0 - 0.010,
                width,
                height,
            ]
        )
        for line in ax.lines:
            line.set_linewidth(1.25 * line.get_linewidth())
        for annotation in ax.texts:
            annotation.arrow_patch.set_linewidth(
                1.25 * annotation.arrow_patch.get_linewidth()
            )
            annotation.arrow_patch.set_mutation_scale(
                1.15 * annotation.arrow_patch.get_mutation_scale()
            )


def add_row_spacing(fig: plt.Figure, axes: tuple[plt.Axes, ...], gap_inches: float = 0.25) -> None:
    """Add space after (a), (b,c) and (d), preserving physical panel sizes."""
    width, height = fig.get_size_inches()
    boxes = [ax.get_position().frozen() for ax in axes]
    new_height = height + 3.0 * gap_inches
    fig.set_size_inches(width, new_height, forward=False)
    for ax, box, shift in zip(axes, boxes, (3, 2, 2, 1, 0), strict=True):
        ax.set_position(
            [box.x0, (box.y0 * height + shift * gap_inches) / new_height,
             box.width, box.height * height / new_height]
        )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.linewidth": 0.9,
            "axes.labelsize": 19,
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


def main(
    output_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float | bool]]:
    if output_path is None:
        output_path = ROOT / "outputs" / "fig5.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields, orbits = model.build_arrays()
    checks = model.validate_arrays(fields, orbits)
    one_wave_keys = [key for key in checks if key.startswith("one_wave_")]
    if max(float(checks[key]) for key in one_wave_keys) > 2.0e-14:
        raise RuntimeError(f"Figure 5 one-wave checks failed: {checks}")
    if float(checks["instantaneous_decomposition_max_abs_residual"]) > 3.0e-14:
        raise RuntimeError(f"Figure 5 quadratic decomposition failed: {checks}")
    if float(checks["full_boxcar_quadrature_max_abs_residual"]) > 2.0e-9:
        raise RuntimeError(f"Figure 5 boxcar quadrature failed: {checks}")
    if float(checks["selected_drift_integration_max_abs_residual"]) > 2.0e-9:
        raise RuntimeError(f"Figure 5 source checks failed: {checks}")
    if not checks["all_curves_inside_fixed_axes"]:
        raise RuntimeError("Figure 5 curves exceed the preserved axes")

    p = model.P
    print(
        f"k0={p.k0:g}, k1={p.k1:g}, omega0={p.omega[0]:.6g}, "
        f"omega1={p.omega[1]:.6g}, sigma0={p.sigma[0]:.6g}, "
        f"sigma1={p.sigma[1]:.6g}, dsigma={p.delta_sigma:.6g}, "
        f"T={p.duration:.6g}"
    )

    configure_style()
    fig = plt.figure(figsize=(10.8, 15.2), constrained_layout=False)
    gs = fig.add_gridspec(
        7,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.95, 0.20, 1.45, 0.26, 0.95, 0.36, 0.95],
        wspace=0.25,
        hspace=0.38,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[2, 1])
    ax_d = fig.add_subplot(gs[4, :])
    ax_e = fig.add_subplot(gs[6, :])

    add_panel_a(ax_a, fields)
    add_panel_b_drifted_loops(ax_b, orbits)
    add_panel_c_orbit_loops_yz(ax_c, orbits)
    add_drift_panel(ax_d, fields, "exact_drift")
    add_drift_panel(ax_e, fields, "selected_minus_exact", r"$A_T-A_\infty$")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.06)
    fig.canvas.draw()
    enlarge_orbit_panels(ax_b, ax_c)
    fig.canvas.draw()
    add_row_spacing(fig, (ax_a, ax_b, ax_c, ax_d, ax_e))
    fig.canvas.draw()

    ba = ax_a.get_position()
    bb = ax_b.get_position()
    bc = ax_c.get_position()
    bd = ax_d.get_position()
    be = ax_e.get_position()

    fig.text(ba.x0 - 0.045, ba.y1 + 0.020, r"($\it{a}$)", fontsize=21, ha="left", va="bottom")
    bc_label_y = max(bb.y1, bc.y1) + 0.005
    fig.text(ba.x0 - 0.045, bc_label_y, r"($\it{b}$)", fontsize=21, ha="left", va="bottom")
    fig.text(bc.x0 - 0.01, bc_label_y, r"($\it{c}$)", fontsize=21, ha="left", va="bottom")
    fig.text(bd.x0 - 0.045, bd.y1 + 0.020, r"($\it{d}$)", fontsize=21, ha="left", va="bottom")
    fig.text(be.x0 - 0.045, be.y1 + 0.020, r"($\it{e}$)", fontsize=21, ha="left", va="bottom")

    bottom_label_y = min(bb.y0, bc.y0) - 0.012
    fig.text(0.5 * (bb.x0 + bb.x1), bottom_label_y, r"$x$", fontsize=17, ha="center", va="top")
    fig.text(0.5 * (bc.x0 + bc.x1), bottom_label_y, r"$y$", fontsize=17, ha="center", va="top")
    fig.text(bb.x0 - 0.02, 0.5 * (bb.y0 + bb.y1), r"$y$", fontsize=17, ha="right", va="center")
    fig.text(bc.x0 - 0.02, 0.5 * (bc.y0 + bc.y1), r"$z$", fontsize=17, ha="right", va="center")

    add_transition_annotation(fig, ax_b, ax_c, ax_d)
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        ax.xaxis.label.set_fontsize(27)
        ax.yaxis.label.set_fontsize(27)
        ax.tick_params(axis="both", labelsize=27)
        ax.title.set_fontsize(28.5)
    for text in fig.texts:
        text.set_fontsize(28.5 if r"\it{" in text.get_text() else 27)

    fig.savefig(output_path, dpi=320)
    plt.close(fig)
    print(output_path)
    return fields, orbits, checks


if __name__ == "__main__":
    main()
