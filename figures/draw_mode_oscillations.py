#!/usr/bin/env python3
"""Mode oscillations and quadratic forcing for the upper row of Figure 4.

Modes p and q occupy the first two rows. The third compares the forcing
at omega_p+omega_q with the linear mode at omega_k.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_mode_oscillations(
    ax: plt.Axes,
    omega_p: float,
    omega_q: float,
    omega_k: float,
    T: float,
    n_points: int = 800,
) -> None:
    """Draw three stacked oscillations (modes p, q, and k) on *ax*."""

    tau = np.linspace(-T / 2, T / 2, n_points)
    delta_omega = omega_p + omega_q - omega_k

    offsets = [2.4, 0.0, -2.4]
    amplitude = 0.85

    signal_p = amplitude * np.cos(omega_p * tau)
    ax.plot(tau, signal_p + offsets[0], color="0.10", linewidth=1.15, zorder=2)
    ax.axhline(offsets[0], color="0.75", linewidth=0.3, zorder=0)

    signal_q = amplitude * np.cos(omega_q * tau)
    ax.plot(tau, signal_q + offsets[1], color="0.10", linewidth=1.15, zorder=2)
    ax.axhline(offsets[1], color="0.75", linewidth=0.3, zorder=0)

    omega_resonant = omega_p + omega_q
    signal_resonant = amplitude * np.cos(omega_resonant * tau)
    ax.plot(tau, signal_resonant + offsets[2], color="0.0", linewidth=1.35, zorder=2)

    if abs(delta_omega) > 1e-10:
        signal_k = amplitude * np.cos(omega_k * tau)
        ax.plot(
            tau, signal_k + offsets[2],
            color="0.35", linewidth=1.05, linestyle="--", zorder=1,
        )
    ax.axhline(offsets[2], color="0.75", linewidth=0.3, zorder=0)

    label_x = -0.07
    labels = [r"$\omega_1$", r"$\omega_2$"]
    for offset, label in zip(offsets[:2], labels):
        ax.text(
            label_x, offset,
            label,
            fontsize=18,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )

    # Solid: forcing at omega_p+omega_q; dashed: linear mode at omega_k.
    # Keep both labels at exact resonance, where the curves coincide.
    key_transform = ax.get_yaxis_transform()
    key_x0, key_x1, key_text_x = -0.10, -0.02, -0.12
    key_entries = [
        (offsets[2] + 0.28, "0.0", "-", r"$\omega_1+\omega_2$"),
        (offsets[2] - 0.28, "0.35", "--", r"$\omega_3$"),
    ]
    for y, color, linestyle, label in key_entries:
        ax.plot(
            [key_x0, key_x1], [y, y],
            color=color,
            linewidth=1.35 if linestyle == "-" else 1.05,
            linestyle=linestyle,
            transform=key_transform,
            clip_on=False,
        )
        ax.text(
            key_text_x, y,
            label,
            fontsize=20,
            ha="right",
            va="center",
            transform=key_transform,
        )

    ax.set_xlim(-T / 2, T / 2)
    ax.set_ylim(offsets[-1] - 1.3, offsets[0] + 1.3)

    ax.axvline(-T / 2, color="0.5", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(T / 2, color="0.5", linewidth=0.6, linestyle="--", zorder=1)

    ax.set_xticks([-T / 2, 0, T / 2])
    ax.set_xticklabels([r"$-T/2$", r"$0$", r"$T/2$"], fontsize=19)
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", length=3, width=0.6, color="0.3")
    ax.set_xlabel(r"$\tau$", fontsize=20)
