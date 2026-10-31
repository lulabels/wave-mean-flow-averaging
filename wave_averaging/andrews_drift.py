#!/usr/bin/env python3
"""Exact and finite-window quadratic drift for prescribed plane waves.

The drift is A_T[(xi dot grad)u_fast], with A_inf[xi]=0. These general
mode and kernel routines support the analytical regression tests.
Figures 5 and 6 use the two-wave Rossby model in
``andrews_textbook_two_wave``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ----------------------------
# Averaging kernel helpers
# ----------------------------

def sinc(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    out = np.ones_like(x_arr)
    m = np.abs(x_arr) > 1e-12
    out[m] = np.sin(x_arr[m]) / x_arr[m]
    if np.isscalar(x):
        return float(out)
    return out


def avg_factor_tophat(omega: float, T: float, t0: float = 0.0) -> complex:
    """
    Centered top-hat average over window length T around t0:
      (1/T)∫_{t0-T/2}^{t0+T/2} e^{i omega s} ds = sinc(omega T/2) e^{i omega t0}
    """
    return complex(sinc(omega * T / 2.0) * np.exp(1j * omega * t0))


# ----------------------------
# Plane-wave representation
# ----------------------------

@dataclass(frozen=True)
class Mode:
    k: np.ndarray            # wavevector (kx, ky)
    omega: float             # fast frequency
    U: np.ndarray            # complex velocity polarization (2,)
    amp: float = 1.0


def build_spatial_amplitudes(modes: List[Mode], X: np.ndarray, Y: np.ndarray) -> List[np.ndarray]:
    """
    a_j(x) = amp_j U_j e^{i k_j·x}, returned as list of arrays shape (2,Nx,Ny).
    """
    a_list: List[np.ndarray] = []
    for md in modes:
        phase = np.exp(1j * (md.k[0] * X + md.k[1] * Y))
        a = md.amp * md.U[:, None, None] * phase[None, :, :]
        a_list.append(a)
    return a_list


def u_fast_grid(X: np.ndarray, Y: np.ndarray, t: float, modes: List[Mode]) -> np.ndarray:
    """Evaluate u_fast(x,t) on grid (X,Y). Returns array shape (2,Nx,Ny)."""
    u = np.zeros((2,) + X.shape, dtype=float)
    for md in modes:
        phase = np.exp(1j * (md.k[0] * X + md.k[1] * Y - md.omega * t))
        u += np.real(md.amp * md.U[:, None, None] * phase[None, :, :])
    return u


def modes_for_streamfunction_cos_sin(*, kx: float, l: float, omega: float, psi0: float, amp: float = 1.0) -> List[Mode]:
    """
    Build plane-wave modes whose real part reconstructs the streamfunction

        psi(x,y,t) = amp * psi0 * cos(kx x - omega t) * sin(l y).

    We represent psi using two positive-frequency plane waves with ky=±l. The coefficients
    are chosen so the resulting real field matches the target amplitude.
    """
    # With the coefficients below, the reconstructed streamfunction is (A/2) cos(...) sin(...).
    # Set A=2*(amp*psi0) so the final amplitude is amp*psi0.
    A = 2.0 * (amp * psi0)

    # Positive-frequency streamfunction coefficients for ky=±l
    Psi_plus = -1j * A / 4.0   # coefficient of e^{i(kx x + l y - omega t)}
    Psi_minus = +1j * A / 4.0  # coefficient of e^{i(kx x - l y - omega t)}

    # Velocity polarizations from streamfunction: u = -∂y psi, v = ∂x psi.
    U_plus = np.array([-1j * (+l) * Psi_plus, 1j * kx * Psi_plus], dtype=complex)
    U_minus = np.array([-1j * (-l) * Psi_minus, 1j * kx * Psi_minus], dtype=complex)

    return [
        Mode(k=np.array([kx, +l], dtype=float), omega=float(omega), U=U_plus),
        Mode(k=np.array([kx, -l], dtype=float), omega=float(omega), U=U_minus),
    ]


# ----------------------------
# Drift diagnostics (finite vs infinite window)
# ----------------------------

def drift_uS_tophat(
    X: np.ndarray,
    Y: np.ndarray,
    modes: List[Mode],
    T: float,
    t0: float = 0.0,
    include_sum_terms: bool = False,
    use_gauge_constants: bool = False,
) -> np.ndarray:
    """
    Compute u^S_T = A_T[(xi · ∇) u_fast] for a prescribed set of plane-wave modes.

    u_fast = Re[ Σ a_l(x) e^{-i ω_l t} ]
    xi satisfies dxi/dt = u_fast, which fixes xi only up to a constant in fast time.

    Default (use_gauge_constants=False) is sec. 4's exact-mean gauge A_inf[xi]=0:
      xi = Re[ Σ b_j(x) e^{-i ω_j t} ], b_j = i a_j/ω_j,
    since A_inf[e^{-i ω_j t}]=0 for every ω_j ≠ 0.

    Setting use_gauge_constants=True pins the constant with the finite-window
    average instead, A_T[xi]=0:
      xi = Re[ Σ b_j(x) (e^{-i ω_j t} - H_j) ], H_j=A_T[e^{-i ω_j t}],
    which adds (xi_const · ∇) A_T[u_fast] to the returned drift. That term is
    quadratic in wave amplitude, so it enters at the same order as the drift
    itself, and it vanishes only for windows where every individual H_j does.
    The manuscript uses use_gauge_constants=False.
    """
    M = len(modes)
    a = build_spatial_amplitudes(modes, X, Y)  # (2,Nx,Ny) complex per mode
    w = [md.omega for md in modes]
    k = [md.k for md in modes]

    # displacement amplitudes b_j = i a_j / ω_j
    b = [1j * a[j] / w[j] for j in range(M)]

    # H_j = A_T[e^{-i ω_j t}] = avg_factor_tophat(-ω_j, T)
    H = [avg_factor_tophat(-w[j], T, t0=t0) for j in range(M)]

    uS = np.zeros_like(a[0], dtype=float)  # (2,Nx,Ny)

    for j in range(M):
        for l in range(M):
            kl = k[l]
            dot_b = 1j * (kl[0] * b[j][0] + kl[1] * b[j][1])  # (Nx,Ny) complex
            dot_b_conj = 1j * (kl[0] * np.conjugate(b[j][0]) + kl[1] * np.conjugate(b[j][1]))

            S1 = avg_factor_tophat(-(w[j] + w[l]), T, t0=t0) if include_sum_terms else 0.0
            S2 = avg_factor_tophat(-(w[j] - w[l]), T, t0=t0)
            S3 = avg_factor_tophat(+(w[j] - w[l]), T, t0=t0)
            S4 = avg_factor_tophat(+(w[j] + w[l]), T, t0=t0) if include_sum_terms else 0.0

            contrib = (
                dot_b * a[l] * S1
                - dot_b * np.conjugate(a[l]) * S2
                + dot_b_conj * a[l] * S3
                - dot_b_conj * np.conjugate(a[l]) * S4
            ) / 4.0

            uS += np.real(contrib)

    # Gauge-constant contribution: xi_const · ∇(A_T[u_fast])
    if use_gauge_constants:
        xi_const = np.zeros_like(a[0], dtype=float)  # (2,Nx,Ny)
        u_mean = np.zeros_like(a[0], dtype=float)
        du_mean_dx = np.zeros_like(a[0], dtype=float)
        du_mean_dy = np.zeros_like(a[0], dtype=float)

        for j in range(M):
            xi_const += -np.real(b[j] * H[j])

        for l in range(M):
            u_mean += np.real(a[l] * H[l])
            du_mean_dx += np.real(1j * k[l][0] * a[l] * H[l])
            du_mean_dy += np.real(1j * k[l][1] * a[l] * H[l])

        uS_const = np.zeros_like(u_mean, dtype=float)
        uS_const[0] = xi_const[0] * du_mean_dx[0] + xi_const[1] * du_mean_dy[0]
        uS_const[1] = xi_const[0] * du_mean_dx[1] + xi_const[1] * du_mean_dy[1]
        uS += uS_const

    return uS


def drift_uS_infinite(
    X: np.ndarray,
    Y: np.ndarray,
    modes: List[Mode],
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Infinite-window / resonance-selected Stokes drift:
      - keep terms only if Δω=0 in the difference-frequency factors,
      - sum-frequency terms vanish,
      - gauge constants ignored (equivalent to H_j=0).
    """
    M = len(modes)
    a = build_spatial_amplitudes(modes, X, Y)
    w = [md.omega for md in modes]
    k = [md.k for md in modes]
    b = [1j * a[j] / w[j] for j in range(M)]

    uS = np.zeros_like(a[0], dtype=float)

    for j in range(M):
        for l in range(M):
            if abs(w[j] - w[l]) > tol:
                continue
            kl = k[l]
            dot_b = 1j * (kl[0] * b[j][0] + kl[1] * b[j][1])
            dot_b_conj = 1j * (kl[0] * np.conjugate(b[j][0]) + kl[1] * np.conjugate(b[j][1]))

            contrib = (
                - dot_b * np.conjugate(a[l]) * 1.0
                + dot_b_conj * a[l] * 1.0
            ) / 4.0
            uS += np.real(contrib)

    return uS


