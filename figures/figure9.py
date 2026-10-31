#!/usr/bin/env python3
r"""Figure 9: Eulerian, Lagrangian and instantaneous-projection comparison.

From the repository directory:
    python generate_data/download_baker_data.py
    python figures/figure9.py

Reads the Baker NetCDF and data/fig9_projection_N256.npz;
writes outputs/fig9.png. This replots archived fields. Rebuilding the
projection also requires Baker's code.zip and the present-study driver,
which are not included. load_fields documents the data and caption checks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file


DEFAULT_BAKER_DATA = (
    ROOT
    / "data/external/baker/"
    / "solver_lowpass_x_y_omega_2_Nint_1_strat_3_T_40.nc"
)
DEFAULT_PROJECTION_DATA = (
    ROOT
    / "data/"
    / "fig9_projection_N256.npz"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "fig9.png"


def load_fields(
    baker_path: Path,
    projection_path: Path,
) -> tuple[list[list[np.ndarray]], dict[str, float]]:
    r"""Load rows of (full, mean, wave) for the common midpoint state.

    Baker variables are z_inst, z_EM and z_LM_at_mid. The included NPZ holds
    instant_zeta, numerical_mean_zeta and numerical_wave_zeta, their analytic
    cross-check and run metadata. It contains no Baker mean fields. Eulerian
    and Lagrangian wave residuals are total minus their archived means.
    Expected midpoint relative L2 mismatch: 2.437357e-14; additivity Linf: 6.66e-16.
    Numerical/analytic projector relative L2 errors are 5.52e-16 and 3.82e-16.

    Caption statistics, from the repository directory:
        import sys
        import numpy as np
        sys.path.insert(0, "figures")
        from figure9 import load_fields, DEFAULT_BAKER_DATA, DEFAULT_PROJECTION_DATA
        rows, checks = load_fields(DEFAULT_BAKER_DATA, DEFAULT_PROJECTION_DATA)
        projected = rows[2][2].ravel()
        for name, row in zip(("Eulerian", "Lagrangian"), rows[:2]):
            wave = row[2].ravel()
            correlation = np.corrcoef(wave, projected)[0, 1]
            relative_l2 = np.linalg.norm(wave - projected) / np.linalg.norm(projected)
            print(name, correlation, relative_l2)

    Expected correlation / relative L2:
        Eulerian    0.887888016063 / 0.518949096383
        Lagrangian  0.993014916573 / 0.190507568878
    Correlation centers both flattened fields. L2 uses uncentered fields and
    the projected-wave norm as denominator, with no mask or extra weighting.
    The plotter checks midpoint agreement and additivity; these four caption
    statistics are calculated separately by the example above.

    Eulerian filtering uses a scalar history at fixed positions. Baker's
    strategy 3 evolves the mean and velocity-dependent material maps through
    PDEs alongside the flow, without explicit particle tracking or saved velocity
    history (doi:10.1017/jfm.2025.42, sections 3.5 and 9). Instantaneous projection
    needs the full midpoint (u,v,eta) and fast operator; it selects the zero
    mode of each 3x3 Fourier block without requiring a time history.
    """
    with netcdf_file(baker_path, "r", mmap=False) as dataset:
        baker = {
            name: np.asarray(variable.data, dtype=float).copy()
            for name, variable in dataset.variables.items()
        }

    with np.load(projection_path, allow_pickle=False) as projection:
        projected_total = np.asarray(projection["instant_zeta"], dtype=float)
        projected_mean = np.asarray(projection["numerical_mean_zeta"], dtype=float)
        projected_wave = np.asarray(projection["numerical_wave_zeta"], dtype=float)

    total = baker["z_inst"]
    eulerian_mean = baker["z_EM"]
    lagrangian_mean = baker["z_LM_at_mid"]

    if projected_total.shape != total.shape:
        raise ValueError(
            f"Projection and Baker fields have different shapes: "
            f"{projected_total.shape} != {total.shape}"
        )

    # Check that all rows compare the same midpoint state.
    total_scale = max(float(np.linalg.norm(total.ravel())), 1.0e-30)
    total_mismatch = float(np.linalg.norm((projected_total - total).ravel()) / total_scale)
    projection_additivity = float(
        np.max(np.abs(projected_total - projected_mean - projected_wave))
    )
    if total_mismatch > 1.0e-10:
        raise ValueError(
            "The numerical projection and Baker rows do not use the same midpoint state: "
            f"relative L2 mismatch = {total_mismatch:.3e}"
        )
    if projection_additivity > 1.0e-10:
        raise ValueError(
            "The numerical projection does not close additively: "
            f"Linf residual = {projection_additivity:.3e}"
        )

    # Columns: full field, mean and wave residual
    rows = [
        [total, eulerian_mean, total - eulerian_mean],
        [total, lagrangian_mean, total - lagrangian_mean],
        [projected_total, projected_mean, projected_wave],
    ]
    diagnostics = {
        "baker_projection_midpoint_relative_l2": total_mismatch,
        "projection_additivity_linf": projection_additivity,
    }
    return rows, diagnostics


def make_figure(
    rows: list[list[np.ndarray]],
    output: Path,
    *,
    vmax: float,
    dpi: int,
    cmap: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.linewidth": 0.8,
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

    column_titles = [
        ("Instantaneous field", None),
        ("Mean component", "Geostrophic-turbulence\ntarget"),
        ("Wave component", "Poincaré-wave\ntarget"),
    ]
    row_labels = [
        r"Eulerian filter $A_T^{\mathrm{E}}$",
        r"Lagrangian filter $A_T^{\mathrm{L}}$",
        r"Exact projection $P$",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10.8, 9.8), constrained_layout=False)
    fig.subplots_adjust(
        left=0.145,
        right=0.875,
        top=0.90,
        bottom=0.055,
        wspace=0.055,
        hspace=0.17,
    )

    meshes = []
    for row_index, (axes_row, fields_row) in enumerate(zip(axes, rows, strict=True)):
        for column_index, (ax, field) in enumerate(
            zip(axes_row, fields_row, strict=True)
        ):
            mesh = ax.pcolormesh(
                field,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
                rasterized=True,
            )
            meshes.append(mesh)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            if row_index == 0:
                title, subtitle = column_titles[column_index]
                ax.set_title(title, fontsize=16, pad=65)
                if subtitle is not None:
                    ax.text(
                        0.5,
                        1.075,
                        f"({subtitle})",
                        transform=ax.transAxes,
                        fontsize=16.5,
                        ha="center",
                        va="bottom",
                    )

            panel_letter = chr(ord("a") + 3 * row_index + column_index)
            ax.text(
                0.025,
                0.975,
                rf"($\it{{{panel_letter}}}$)",
                transform=ax.transAxes,
                fontsize=16,
                ha="left",
                va="top",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
                zorder=10,
            )

        axes_row[0].set_ylabel(row_labels[row_index], fontsize=15, labelpad=29)

    colorbar_axis = fig.add_axes([0.90, 0.12, 0.020, 0.70])
    colorbar = fig.colorbar(meshes[0], cax=colorbar_axis)
    colorbar.set_label("Relative vorticity", fontsize=16, labelpad=10)
    colorbar.ax.tick_params(labelsize=17, width=0.8)
    colorbar.outline.set_linewidth(0.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    r"""Plot the archived comparison at t=20, T=40, cutoff 2 and vmax=1.

    Source-run timing definitions for the caption:
        324.571256 s includes solver construction/setup, 6660 RK4 steps to t=20,
        numerical/analytic projection and vorticity. It stops before external
        comparison, NPZ writing and plotting.
        1504.806066 s brackets solver.solve() on [0,40], including evolution,
        online mean/maps and regridding. Setup + solve took 1505.539603 s.
    Both use N=256, Ro=0.4, Fr=0.3, A=0.5, (kw,lw)=(1,0), requested dt=0.003
    adjusted to 1/333, strat=3 and dtslow=1. The Lagrangian run also uses
    solve_LM=True, Nint=1, cutoff=2, ReL_hyp=None and all four map solves.
    These are whole-workflow timings with different evolution durations.

    Full regeneration needs SW_GLM.py and uvr_2Dturbulence_256.mat from Baker's
    code.zip, plus the present-study projection driver. scipy.io.netcdf_file
    is sufficient to read the archived input for this plot.
    """
    parser = argparse.ArgumentParser(
        description="Make the Section 5 three-way wave--mean decomposition figure."
    )
    parser.add_argument("--baker-data", type=Path, default=DEFAULT_BAKER_DATA)
    parser.add_argument("--projection-data", type=Path, default=DEFAULT_PROJECTION_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap (default: gray; use RdBu_r to reproduce the archived color version).",
    )
    args = parser.parse_args()

    rows, diagnostics = load_fields(args.baker_data, args.projection_data)
    make_figure(rows, args.output, vmax=args.vmax, dpi=args.dpi, cmap=args.cmap)
    print(f"wrote {args.output}")
    for name, value in diagnostics.items():
        print(f"{name}: {value:.6e}")


if __name__ == "__main__":
    main()
