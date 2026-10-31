#!/usr/bin/env python3
r"""Figure 3: stationary Rossby wave, parcel displacements and zonal Stokes drift.

    python figures/figure3.py

No data files are needed. Output: outputs/fig3.png.
Displacements follow Andrews, Holton & Leovy (1987), (4.5.24a-c), and the
orbit diagrams follow their Figure 4.13. They are calculated from the formulas;
no textbook artwork is included. Parameters are in Parameters below.
Panels (b,c) use normalized diagram coordinates and the common display scales
below, rather than a shared physical metric scale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


@dataclass(frozen=True)
class Parameters:
    """Parameters for the stationary Rossby wave."""

    k: float = 2.0
    ell: float = 1.0
    m: float = 1.1
    H: float = 1.4
    psi0: float = 1.0
    z_ref: float = 0.2
    U: float = 10.0
    f0: float = 1.0
    N2: float = 10.0

    @property
    def L(self) -> float:
        return np.pi / self.ell

    @property
    def epsilon_ahl(self) -> float:
        return self.f0**2 / self.N2

    @property
    def beta(self) -> float:
        # Stationary-wave dispersion relation; beta=51.33755102040816 for these parameters.
        denominator = (
            self.k**2
            + self.ell**2
            + self.epsilon_ahl * (self.m**2 + 1.0 / (4.0 * self.H**2))
        )
        return self.U * denominator

    @property
    def amplitude_at_z_ref(self) -> float:
        return self.psi0 * np.exp(self.z_ref / (2.0 * self.H))


P = Parameters()
PHASE = np.linspace(0.0, 2.0 * np.pi, 721)
# Orbit centers, normalized by L
XY_CENTERS = np.linspace(0.14, 0.86, 6)
YZ_CENTERS = np.linspace(0.16, 0.84, 6)
# Display scales; the same scale applies to every loop in a panel.
XY_FAMILY_GAIN = 2.0
YZ_FAMILY_GAIN = 1.8
X_DISPLAY_MAGNIFICATION = 4.0
VERTICAL_DIAGRAM_CENTER = 0.52


def displacement(centers: np.ndarray, phase: np.ndarray, p: Parameters = P) -> dict[str, np.ndarray]:
    """First-order AHL (4.5.24a-c) parcel displacement at z=z_ref."""
    s = np.asarray(centers, dtype=float)[:, None]
    theta = np.asarray(phase, dtype=float)[None, :]
    ly = np.pi * s
    amplitude = p.amplitude_at_z_ref
    xi = -(p.ell * amplitude / (p.k * p.U)) * np.sin(theta) * np.cos(ly)
    eta = +(amplitude / p.U) * np.cos(theta) * np.sin(ly)
    zeta = -(p.f0 * amplitude / p.N2) * (
        np.cos(theta) / (2.0 * p.H) - p.m * np.sin(theta)
    ) * np.sin(ly)
    return {"xi": xi, "eta": eta, "zeta": zeta}


def xy_display_paths(p: Parameters = P) -> tuple[np.ndarray, np.ndarray]:
    """Apply the x-y display scales to the first-order displacements."""
    raw = displacement(XY_CENTERS, PHASE, p)
    x = X_DISPLAY_MAGNIFICATION * XY_FAMILY_GAIN * raw["xi"] / p.L
    y = XY_CENTERS[:, None] + XY_FAMILY_GAIN * raw["eta"] / p.L
    return x, y


def yz_display_paths(p: Parameters = P) -> tuple[np.ndarray, np.ndarray]:
    """Scale the y-z displacements and shift the displayed height to z=0.52."""
    raw = displacement(YZ_CENTERS, PHASE, p)
    y = YZ_CENTERS[:, None] + YZ_FAMILY_GAIN * raw["eta"] / p.L
    z = VERTICAL_DIAGRAM_CENTER + YZ_FAMILY_GAIN * raw["zeta"] / p.L
    return y, z


def ahl_zonal_stokes_drift(y: np.ndarray, z: np.ndarray, p: Parameters = P) -> np.ndarray:
    """Zonal Stokes drift from AHL (4.5.25a).

    In this log-pressure/QG example, v^S=0 and w^S is nonzero;
    only u^S is plotted in panel (d).
    """
    return (
        -(p.ell**2 / (2.0 * p.U))
        * p.psi0**2
        * np.exp(z / p.H)
        * np.cos(2.0 * p.ell * y)
    )


def plot_orbits(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, center_arrows_on_sides: bool = False) -> None:
    """Draw the sampled paths; use side arrows for the narrow central xy loops."""
    for orbit_index, (xx, yy) in enumerate(zip(x, y)):
        ax.plot(xx, yy, color="0.25", linewidth=1.2)
        i0 = len(xx) // 8
        i1 = i0 + round(10 * len(xx) / 400)
        if center_arrows_on_sides and orbit_index in (2, 3):
            midpoint = (len(xx) - 1) // 4
            half_span = round((len(xx) - 1) / 40)
            i0, i1 = midpoint - half_span, midpoint + half_span
        ax.annotate(
            "",
            xy=(xx[i1], yy[i1]),
            xytext=(xx[i0], yy[i0]),
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "0.2"},
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def streamfunction(x: np.ndarray, y: np.ndarray, z: float, *, k: float, l: float, m: float, H: float, psi0: float) -> np.ndarray:
    """Stationary Rossby-wave streamfunction, following AHL (4.5.20)."""
    return psi0 * np.exp(z / (2.0 * H)) * np.cos(k * x + m * z) * np.sin(l * y)


def add_panel_a(ax: plt.Axes) -> None:
    """Streamfunction contours (panel a)."""
    x = np.linspace(0.0, 2.0 * np.pi, 360)
    y = np.linspace(0.0, np.pi, 180)
    X, Y = np.meshgrid(x, y, indexing="xy")

    psi = streamfunction(X, Y, z=0.2, k=2.0, l=1.0, m=1.1, H=1.4, psi0=1.0)
    levels = np.linspace(-1.0, 1.0, 15)

    cf = ax.contourf(X, Y, psi, levels=levels, cmap="Greys", alpha=0.26, antialiased=True)
    ax.contour(X, Y, psi, levels=levels, colors="black", linewidths=0.9, linestyles="solid")
    cf.set_clim(-1.0, 1.0)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0, labelpad=18, va="center")
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0.0, np.pi / 2.0, np.pi], labels=["0", r"$L/2$", r"$L$"])
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, np.pi)


def add_panel_b(ax: plt.Axes) -> None:
    """First-order x-y orbits."""
    x, y = xy_display_paths()
    plot_orbits(ax, x, y, center_arrows_on_sides=True)
    ax.set_xlim(-0.26, 0.26)
    ax.set_ylim(0.0, 1.0)


def add_panel_c(ax: plt.Axes) -> None:
    """First-order y-z orbits."""
    y, z = yz_display_paths()
    plot_orbits(ax, y, z)
    ax.set_xlim(0.03, 0.97)
    ax.set_ylim(0.31, 0.73)


def add_panel_d(ax: plt.Axes) -> None:
    """Zonal Stokes drift in the y-z plane."""
    y = np.linspace(0.0, P.L, 320)
    z = np.linspace(0.0, P.H, 180)
    Y, Z = np.meshgrid(y, z, indexing="xy")

    uS = ahl_zonal_stokes_drift(Y, Z)
    levels = np.linspace(-np.max(np.abs(uS)), np.max(np.abs(uS)), 15)

    cf = ax.contourf(
        Y,
        Z,
        uS,
        levels=levels,
        cmap="Greys",
        alpha=0.26,
        antialiased=True,
    )
    cf.set_clim(levels[0], levels[-1])
    ax.contour(Y, Z, uS, levels=levels, colors="0.25", linewidths=0.9)
    ax.contour(Y, Z, uS, levels=[0.0], colors="black", linewidths=1.2)

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$", rotation=0, labelpad=18, va="center")
    ax.set_xticks([0.0, P.L / 2.0, P.L], labels=["0", r"$L/2$", r"$L$"])
    ax.set_yticks([0.0, P.H / 2.0, P.H], labels=["0", r"$H/2$", r"$H$"])
    ax.set_xlim(0.0, P.L)
    ax.set_ylim(0.0, P.H)


def add_transition_annotation(fig: plt.Figure, ax_b: plt.Axes, ax_c: plt.Axes, ax_d: plt.Axes) -> None:
    """Averaging symbol between the orbit panels and the drift field."""
    bb = ax_b.get_position()
    bc = ax_c.get_position()
    bd = ax_d.get_position()

    # Center the symbol over panel (d).
    x_mid = 0.5 * (bd.x0 + bd.x1)
    y_top = min(bb.y0, bc.y0) - 0.035
    y_bottom = bd.y1 + 0.008

    arrow = FancyArrowPatch(
        (x_mid, y_top),
        (x_mid, y_bottom),
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="black",
        transform=fig.transFigure,
    )
    fig.add_artist(arrow)
    fig.text(
        x_mid + 0.016,
        0.5 * (y_top + y_bottom),
        r"$A_\infty$",
        ha="left",
        va="center",
        fontsize=26,
    )


def main() -> None:
    outdir = ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.linewidth": 0.9,
            "axes.labelsize": 26,
            "xtick.labelsize": 23,
            "ytick.labelsize": 23,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )

    fig = plt.figure(figsize=(10.8, 11.0), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.95, 1.45, 0.95],
        wspace=0.25,
        hspace=0.42,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    add_panel_a(ax_a)
    add_panel_b(ax_b)
    add_panel_c(ax_c)
    add_panel_d(ax_d)

    ax_d.xaxis.labelpad = 0
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.07)

    # Resolve the equal-aspect axes before positioning the labels.
    fig.canvas.draw()

    ba = ax_a.get_position()
    bb = ax_b.get_position()
    bc = ax_c.get_position()
    bd = ax_d.get_position()

    # Panel letters use a common left margin and a shared middle-row baseline.
    for bounds, letter in ((ba, "a"), (bd, "d")):
        fig.text(bounds.x0 - 0.045, bounds.y1 + 0.020,
                 rf"($\it{{{letter}}}$)", fontsize=28, ha="left", va="bottom")

    orbit_bounds = []
    for ax, coordinates, letter, horizontal, vertical in (
        (ax_b, xy_display_paths(), "b", "x", "y"),
        (ax_c, yz_display_paths(), "c", "y", "z"),
    ):
        xx, yy = coordinates
        corners = fig.transFigure.inverted().transform(ax.transData.transform([
            (np.min(xx), np.min(yy)), (np.max(xx), np.max(yy)),
        ]))
        (left, bottom), (right, top) = corners
        orbit_bounds.append((left, bottom, right, top, letter, horizontal, vertical))
    row_label_y = max(bounds[3] for bounds in orbit_bounds) + 0.025
    coordinate_label_y = min(bounds[1] for bounds in orbit_bounds) - 0.035
    for left, bottom, right, top, letter, horizontal, vertical in orbit_bounds:
        panel_x = ba.x0 - 0.045 if letter == "b" else 0.5 * (ba.x0 + ba.x1)
        fig.text(panel_x, row_label_y, rf"($\it{{{letter}}}$)",
                 fontsize=28, ha="left", va="bottom")
        fig.text((left + right) / 2, coordinate_label_y, f"${horizontal}$",
                 fontsize=24, ha="center", va="top")
        fig.text(left - 0.045, (bottom + top) / 2, f"${vertical}$",
                 fontsize=24, ha="right", va="center")

    add_transition_annotation(fig, ax_b, ax_c, ax_d)

    png_path = outdir / "fig3.png"
    fig.savefig(png_path, dpi=320)
    plt.close(fig)

    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
