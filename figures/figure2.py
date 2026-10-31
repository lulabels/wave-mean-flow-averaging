#!/usr/bin/env python3
r"""Figure 2: slow, full and wave components of the RSW height field.

Run from the repository directory:
    python figures/figure2.py

Reads data/fig2_rsw_fields.npz and writes outputs/fig2.png.
The included fields are height anomalies at t=3 from the viscous N256 run
(nu=2e-4). Full = slow + wave to roundoff; their approximate maximum absolute
values are 0.113, 0.080 and 0.051. The input config records the trajectory hash.
To regenerate the input, see generate_data/run_rsw_truth.py and
generate_data/build_compact_rsw_data.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_DATA = ROOT / "data" / "fig2_rsw_fields.npz"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

PANEL_DEFS = [
    ("slow", "Slow balanced field"),
    ("full", "Full field"),
    ("wave", "Fast wave field"),
]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.linewidth": 0.85,
            "axes.labelsize": 16.5,
            "axes.titlesize": 17.25,
            "xtick.labelsize": 16.5,
            "ytick.labelsize": 16.5,
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


def load_compact_fields(path: Path) -> tuple[dict, float, dict[str, np.ndarray]]:
    r"""Load eta_slow, eta_full and eta_wave, with their grid/time metadata.

    Only the plotted height fields are stored here; the full trajectory is
    not needed for this figure. Input-file hashes are in data/checksums.json.
    """
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config_json"]))
        time_used = float(data["time"])
        components = {
            key: np.asarray(data[f"eta_{key}"], dtype=np.float64)
            for key, _ in PANEL_DEFS
        }
    return config, time_used, components


def grid_from_config(config: dict) -> tuple[np.ndarray, np.ndarray]:
    n = int(config["N"])
    lx = float(config["Lx"])
    ly = float(config["Ly"])
    x = np.linspace(0.0, lx, n, endpoint=False)
    y = np.linspace(0.0, ly, n, endpoint=False)
    return x, y


def levels_for(field: np.ndarray, minimum_vmax: float) -> tuple[np.ndarray, np.ndarray]:
    vmax = max(float(np.max(np.abs(field))), minimum_vmax)
    fill_levels = np.linspace(0.0, vmax, 9)
    contour_levels = np.linspace(vmax / 6.0, vmax, 6)
    return fill_levels, contour_levels


def draw_height_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    eta: np.ndarray,
    *,
    title: str,
    fill_levels: np.ndarray,
    contour_levels: np.ndarray,
) -> None:
    # Shade |eta|; use the contour line styles to show its sign.
    ax.contourf(
        x,
        y,
        np.abs(eta).T,
        levels=fill_levels,
        cmap="Greys",
        alpha=0.28,
        antialiased=True,
    )
    ax.contour(
        x,
        y,
        eta.T,
        levels=-contour_levels[::-1],
        colors="0.15",
        linewidths=0.85,
        linestyles="dashed",
    )
    ax.contour(
        x,
        y,
        eta.T,
        levels=contour_levels,
        colors="0.05",
        linewidths=0.85,
        linestyles="solid",
    )
    ax.contour(x, y, eta.T, levels=[0.0], colors="0.55", linewidths=1.15)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x[0]), float(x[-1] + (x[1] - x[0])))
    ax.set_ylim(float(y[0]), float(y[-1] + (y[1] - y[0])))
    ax.set_title(title, pad=14, fontsize=17.25)
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0.0, np.pi, 2.0 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0, labelpad=13, va="center")


def add_panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, ax in zip(["a", "b", "c"], axes, strict=True):
        bbox = ax.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        fig.text(
            bbox.x0 - 0.020,
            bbox.y1 + 0.028,
            rf"($\it{{{label}}}$)",
            ha="left",
            va="bottom",
            fontsize=17.25,
        )


def assert_legend_outside_axes(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_boxes = [ax.get_window_extent(renderer) for ax in axes]
    for legend in fig.legends:
        legend_box = legend.get_window_extent(renderer)
        for axes_box in axes_boxes:
            if legend_box.overlaps(axes_box):
                raise RuntimeError("figure legend overlaps a plotting axis")


def save_figure(components: dict[str, np.ndarray], config: dict, time_used: float, output_dir: Path) -> Path:
    x, y = grid_from_config(config)
    eta_fields = {key: np.asarray(components[key], dtype=float) for key, _ in PANEL_DEFS}

    # Common minimum color range for the three panels
    full_slow_vmax = max(
        float(np.max(np.abs(eta_fields["slow"]))),
        float(np.max(np.abs(eta_fields["full"]))),
    )

    fig, axes_arr = plt.subplots(1, 3, figsize=(10.2, 5.0), constrained_layout=False)
    axes = list(axes_arr)
    for ax, (key, title) in zip(axes, PANEL_DEFS, strict=True):
        fill_levels, contour_levels = levels_for(eta_fields[key], full_slow_vmax)
        draw_height_panel(
            ax,
            x,
            y,
            eta_fields[key],
            title=title,
            fill_levels=fill_levels,
            contour_levels=contour_levels,
        )

    for ax in axes[1:]:
        ax.set_ylabel("")

    n = int(config["N"])
    legend_handles = [
        Line2D([0], [0], color="0.05", linestyle="solid", linewidth=1.2, label=r"Positive $\eta$"),
        Line2D([0], [0], color="0.55", linestyle="solid", linewidth=1.3, label=r"Zero contour"),
        Line2D([0], [0], color="0.15", linestyle="dashed", linewidth=1.2, label=r"Negative $\eta$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
        fontsize=16.5,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.29, wspace=0.23)
    add_panel_labels(fig, axes)
    assert_legend_outside_axes(fig, axes)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig2.png"
    fig.savefig(output_path, dpi=320)
    plt.close(fig)
    return output_path


def run(args: argparse.Namespace) -> None:
    setup_matplotlib()
    config, time_used, components = load_compact_fields(Path(args.data))
    output_path = save_figure(components, config, time_used, Path(args.output_dir))
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 2 from compact RSW fields.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="compact Figure 2 NPZ")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
