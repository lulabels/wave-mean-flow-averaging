from __future__ import annotations

import numpy as np

from .solver_pseudospectral import RSWParameters, RSWState, SpectralGrid, fft2, ifft2


def balanced_basis(grid: SpectralGrid, params: RSWParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u0 = -1j * params.g * grid.KY / params.f
    v0 = 1j * params.g * grid.KX / params.f
    eta0 = np.ones_like(grid.KX, dtype=complex)
    zero = (np.abs(grid.KX) < 1.0e-14) & (np.abs(grid.KY) < 1.0e-14)
    u0 = u0.copy()
    v0 = v0.copy()
    eta0 = eta0.copy()
    u0[zero] = 0.0
    v0[zero] = 0.0
    eta0[zero] = 1.0
    return u0, v0, eta0


def slow_project_state(state: RSWState, grid: SpectralGrid, params: RSWParameters) -> RSWState:
    u0, v0, eta0 = balanced_basis(grid, params)
    # Solver states are 2/3 dealiased.  Applying the same mask here
    # avoids ambiguous real-valued Nyquist modes in the discrete geostrophic
    # basis and keeps P0 idempotent on grid fields.
    mask = grid.dealias_mask
    u_hat = fft2(state.u) * mask
    v_hat = fft2(state.v) * mask
    eta_hat = fft2(state.eta) * mask

    denom = (
        params.H * np.conjugate(u0) * u0
        + params.H * np.conjugate(v0) * v0
        + params.g * np.conjugate(eta0) * eta0
    )
    denom = np.where(np.abs(denom) < 1.0e-30, 1.0, denom)
    alpha = (
        params.H * np.conjugate(u0) * u_hat
        + params.H * np.conjugate(v0) * v_hat
        + params.g * np.conjugate(eta0) * eta_hat
    ) / denom

    return RSWState(ifft2(u0 * alpha), ifft2(v0 * alpha), ifft2(eta0 * alpha))


def wave_project_state(state: RSWState, grid: SpectralGrid, params: RSWParameters) -> RSWState:
    slow = slow_project_state(state, grid, params)
    return RSWState(state.u - slow.u, state.v - slow.v, state.eta - slow.eta)


def energy_inner(a: RSWState, b: RSWState, params: RSWParameters) -> float:
    val = params.H * np.mean(a.u * b.u + a.v * b.v) + params.g * np.mean(a.eta * b.eta)
    return float(val)


def projector_sanity(state: RSWState, grid: SpectralGrid, params: RSWParameters) -> dict[str, float]:
    p1 = slow_project_state(state, grid, params)
    p2 = slow_project_state(p1, grid, params)
    wave = RSWState(state.u - p1.u, state.v - p1.v, state.eta - p1.eta)
    idempotence = np.sqrt(np.mean((p2.as_array() - p1.as_array()) ** 2)) / max(
        np.sqrt(np.mean(p1.as_array() ** 2)), 1.0e-14
    )
    orth = abs(energy_inner(p1, wave, params)) / max(abs(energy_inner(state, state, params)), 1.0e-14)
    slow_fraction = energy_inner(p1, p1, params) / max(energy_inner(state, state, params), 1.0e-14)
    return {
        "P0_idempotence_error": float(idempotence),
        "P0_energy_orthogonality": float(orth),
        "P0_slow_energy_fraction": float(slow_fraction),
    }
