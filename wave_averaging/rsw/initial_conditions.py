from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver_pseudospectral import RSWParameters, RSWState, SpectralGrid


@dataclass(frozen=True)
class WaveMode:
    mx: int
    my: int
    phase: float
    amplitude: float
    frequency: float

    @property
    def k_abs(self) -> float:
        return float(np.sqrt(self.mx * self.mx + self.my * self.my))


def geostrophic_jet(grid: SpectralGrid, params: RSWParameters, amplitude: float, meridional_mode: int) -> RSWState:
    """Exact inviscid geostrophic jet: u=U cos(my), eta_y=-(f/g)u."""
    y_phase = 2.0 * np.pi * meridional_mode * grid.Y / grid.Ly
    u = amplitude * np.cos(y_phase)
    v = np.zeros_like(u)
    eta = -(params.f / params.g) * amplitude * grid.Ly / (2.0 * np.pi * meridional_mode) * np.sin(y_phase)
    return RSWState(u, v, eta)


def poincare_frequency(mx: int, my: int, params: RSWParameters, grid: SpectralGrid) -> float:
    kx = 2.0 * np.pi * mx / grid.Lx
    ky = 2.0 * np.pi * my / grid.Ly
    return float(np.sqrt(params.f * params.f + params.g * params.H * (kx * kx + ky * ky)))


def poincare_eigenvector(mx: int, my: int, params: RSWParameters, grid: SpectralGrid) -> np.ndarray:
    """Return q=(u,v,eta) with eta=1 for L q = -i omega q."""
    if mx == 0 and my == 0:
        raise ValueError("Poincare modes require nonzero horizontal wavenumber")
    kx = 2.0 * np.pi * mx / grid.Lx
    ky = 2.0 * np.pi * my / grid.Ly
    omega = poincare_frequency(mx, my, params, grid)
    matrix = np.array([[-1j * omega, -params.f], [params.f, -1j * omega]], dtype=complex)
    rhs = np.array([-1j * params.g * kx, -1j * params.g * ky], dtype=complex)
    u, v = np.linalg.solve(matrix, rhs)
    return np.array([u, v, 1.0 + 0.0j], dtype=complex)


def linear_residual_norm(mx: int, my: int, params: RSWParameters, grid: SpectralGrid) -> float:
    kx = 2.0 * np.pi * mx / grid.Lx
    ky = 2.0 * np.pi * my / grid.Ly
    q = poincare_eigenvector(mx, my, params, grid)
    omega = poincare_frequency(mx, my, params, grid)
    Lq = np.array(
        [
            params.f * q[1] - 1j * params.g * kx * q[2],
            -params.f * q[0] - 1j * params.g * ky * q[2],
            -1j * params.H * (kx * q[0] + ky * q[1]),
        ],
        dtype=complex,
    )
    return float(np.linalg.norm(Lq + 1j * omega * q) / max(np.linalg.norm(q), 1.0e-14))


def add_poincare_modes(
    state: RSWState,
    grid: SpectralGrid,
    params: RSWParameters,
    mode_specs: list[tuple[int, int, float]],
    amplitude: float,
) -> tuple[RSWState, list[WaveMode]]:
    u = state.u.copy()
    v = state.v.copy()
    eta = state.eta.copy()
    modes: list[WaveMode] = []

    for j, (mx, my, phase) in enumerate(mode_specs):
        eig = poincare_eigenvector(mx, my, params, grid)
        theta = 2.0 * np.pi * (mx * grid.X / grid.Lx + my * grid.Y / grid.Ly) + phase
        amp = amplitude / np.sqrt(j + 1.0)
        wave = amp * eig[:, None, None] * np.exp(1j * theta)[None, :, :]
        u += wave[0].real
        v += wave[1].real
        eta += wave[2].real
        modes.append(WaveMode(mx, my, phase, amp, poincare_frequency(mx, my, params, grid)))

    return RSWState(u, v, eta), modes


def make_initial_state(config: dict, grid: SpectralGrid, params: RSWParameters) -> tuple[RSWState, list[WaveMode]]:
    slow = config["slow_flow"]
    state = geostrophic_jet(
        grid,
        params,
        amplitude=float(slow["jet_amplitude"]),
        meridional_mode=int(slow["meridional_mode"]),
    )

    waves = config["waves"]
    mode_specs = [(int(m[0]), int(m[1]), float(m[2])) for m in waves["modes"]]
    return add_poincare_modes(state, grid, params, mode_specs, float(waves["amplitude"]))


def geostrophic_residual(state: RSWState, grid: SpectralGrid, params: RSWParameters) -> float:
    etax = np.fft.ifft2(1j * grid.KX * np.fft.fft2(state.eta)).real
    etay = np.fft.ifft2(1j * grid.KY * np.fft.fft2(state.eta)).real
    rx = -params.f * state.v + params.g * etax
    ry = params.f * state.u + params.g * etay
    denom = np.sqrt(np.mean(state.u * state.u + state.v * state.v + state.eta * state.eta))
    return float(np.sqrt(np.mean(rx * rx + ry * ry)) / max(denom, 1.0e-14))
