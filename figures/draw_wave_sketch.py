#!/usr/bin/env python3
"""Relative-phase factor over [-T/2, T/2], for the lower row of Figure 4.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def draw_wave_sketch(
    ax: plt.Axes,
    delta_omega: float,
    T: float,
    n_points: int = 600,
    shade: bool = True,
) -> None:
    """Plot cos(Delta_omega*tau), with optional shading."""
    tau = np.linspace(-T / 2, T / 2, n_points)
    signal = np.cos(delta_omega * tau)

    if shade:
        ax.fill_between(tau, 0, signal, color="0.88", linewidth=0, zorder=1)

    mean_val = np.mean(signal)

    if abs(delta_omega) < 1e-10:
        ax.plot(tau, signal, color="0.0", linewidth=0.9, linestyle="--", zorder=2)
    else:
        ax.plot(tau, signal, color="0.15", linewidth=0.9, zorder=2)
        ax.axhline(mean_val, color="0.0", linewidth=0.9, linestyle="--", zorder=3)

    ax.axhline(0, color="0.35", linewidth=0.6, zorder=1)

    ax.set_xlim(-T / 2, T / 2)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([-T / 2, 0, T / 2])
    ax.set_xticklabels([r"$-T/2$", r"$0$", r"$T/2$"], fontsize=19)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([r"$-1$", r"$0$", r"$1$"], fontsize=19)
    ax.set_ylabel(
        r"Re$(e^{-i\Delta\omega\tau})$",
        fontsize=18,
        labelpad=2,
    )
    ax.set_xlabel(r"$\tau$", fontsize=20)
