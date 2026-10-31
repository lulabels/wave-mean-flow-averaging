#!/usr/bin/env python3
r"""Figure 4: exact, near and non-resonant triads.

    python figures/figure4.py

No data files are needed. Output: outputs/fig4.png.
The upper row compares mode oscillations and quadratic forcing. The lower
row shows cos(Delta_omega*tau) over [-T/2,T/2]. T=10, omega_1=3, omega_3=8;
omega_2 is 5, 4.75 or 2.4. The absolute boxcar weights are about 1, 0.76, 0.03.
The helpers below use omega_k-omega_p-omega_q for signed detuning; reversing
its sign leaves the plotted cosine and absolute sinc weight unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
import numpy as np

from draw_mode_oscillations import draw_mode_oscillations
from draw_wave_sketch import draw_wave_sketch


OUTDIR = ROOT / "outputs"

T = 10.0
OMEGA_P = 3.0
OMEGA_K = 8.0

CASES = [
    {"label": "Exact resonance", "omega_q": 5.0,  "panel": "a"},
    {"label": "Near resonance",  "omega_q": 4.75, "panel": "b"},
    {"label": "Non-resonance",   "omega_q": 2.4,  "panel": "c"},
]

for c in CASES:
    c["delta_omega"] = OMEGA_P + c["omega_q"] - OMEGA_K


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.linewidth": 0.8,
            "axes.labelsize": 21,
            "axes.titlesize": 21,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "font.family": "Times New Roman",
            "font.serif": [
                "Times New Roman", "Times",
                "Nimbus Roman No9 L", "DejaVu Serif",
            ],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_vertical_separator(fig, x, y0, y1, color="0.75", linewidth=0.7):
    separator = Line2D(
        [x, x],
        [y0, y1],
        color=color,
        linewidth=linewidth,
        transform=fig.transFigure,
        clip_on=False,
        zorder=0,
    )
    fig.lines.append(separator)


def main() -> None:
    setup_matplotlib()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.0, 12.0))

    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.78, 1.0],
        hspace=0.25,
        wspace=0.85,
        left=0.145, right=0.945,
        top=0.90, bottom=0.10,
    )

    kbar_texts = []
    column_bboxes = []
    for col, case in enumerate(CASES):
        ax_modes = fig.add_subplot(gs[0, col])
        ax_wave = fig.add_subplot(gs[1, col])

        draw_mode_oscillations(
            ax_modes, OMEGA_P, case["omega_q"], OMEGA_K, T,
        )
        draw_wave_sketch(ax_wave, case["delta_omega"], T)

        mean_val = np.mean(np.cos(
            case["delta_omega"] * np.linspace(-T/2, T/2, 1000)))
        mag = abs(mean_val)

        txt = ax_wave.text(
            0.96, 0.95,
            rf"$|\widehat{{K}}| = {mag:.2f}$",
            fontsize=19,
            ha="right",
            va="top",
            transform=ax_wave.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
            zorder=5,
            clip_on=False,
        )
        kbar_texts.append(txt)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()

    # Include tick labels and the |K-hat| annotation inside each column box.
    pad_x = 0.016
    pad_bot = 0.014
    title_head = 0.045

    for col, case in enumerate(CASES):
        ax_modes = fig.axes[col * 2]
        ax_wave = fig.axes[col * 2 + 1]

        content = mtransforms.Bbox.union([
            ax_modes.get_tightbbox(renderer),
            ax_wave.get_tightbbox(renderer),
            kbar_texts[col].get_window_extent(renderer),
        ]).transformed(inv_fig)

        box_bbox = mtransforms.Bbox.from_extents(
            content.x0 - pad_x,
            content.y0 - pad_bot,
            content.x1 + pad_x,
            content.y1 + title_head,
        )
        column_bboxes.append(box_bbox)

        fig.text(
            box_bbox.x0 + 0.008,
            box_bbox.y1 - 0.013,
            rf"($\it{{{case['panel']}}}$)",
            fontsize=20.25,
            ha="left",
            va="top",
        )

        fig.text(
            0.5 * (box_bbox.x0 + box_bbox.x1),
            box_bbox.y1 - 0.013,
            case["label"],
            fontsize=20.25,
            ha="center",
            va="top",
        )

    separator_y0 = min(bbox.y0 for bbox in column_bboxes)
    separator_y1 = max(bbox.y1 for bbox in column_bboxes)
    for left_bbox, right_bbox in zip(column_bboxes[:-1], column_bboxes[1:]):
        separator_x = 0.5 * (left_bbox.x1 + right_bbox.x0)
        draw_vertical_separator(
            fig, separator_x, separator_y0, separator_y1,
        )

    png_path = OUTDIR / "fig4.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
