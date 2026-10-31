from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RSWParameters:
    f: float = 1.0
    g: float = 1.0
    H: float = 1.0
    nu: float = 0.0


@dataclass(frozen=True)
class SpectralGrid:
    N: int
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        if self.N <= 0 or self.N % 2:
            raise ValueError("N must be a positive even integer")

    @property
    def x(self) -> Array:
        return np.linspace(0.0, self.Lx, self.N, endpoint=False)

    @property
    def y(self) -> Array:
        return np.linspace(0.0, self.Ly, self.N, endpoint=False)

    @property
    def X(self) -> Array:
        return np.meshgrid(self.x, self.y, indexing="ij")[0]

    @property
    def Y(self) -> Array:
        return np.meshgrid(self.x, self.y, indexing="ij")[1]

    @property
    def kx_1d(self) -> Array:
        return 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.Lx / self.N)

    @property
    def ky_1d(self) -> Array:
        return 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.Ly / self.N)

    @property
    def KX(self) -> Array:
        return np.meshgrid(self.kx_1d, self.ky_1d, indexing="ij")[0]

    @property
    def KY(self) -> Array:
        return np.meshgrid(self.kx_1d, self.ky_1d, indexing="ij")[1]

    @property
    def K2(self) -> Array:
        return self.KX * self.KX + self.KY * self.KY

    @property
    def dealias_mask(self) -> Array:
        ix = np.fft.fftfreq(self.N, d=1.0 / self.N)
        iy = np.fft.fftfreq(self.N, d=1.0 / self.N)
        IX, IY = np.meshgrid(ix, iy, indexing="ij")
        # Use a strict 2/3 cutoff.  Including the exact N/3 endpoint when
        # N is divisible by 3 lets two retained boundary modes alias back
        # into the retained set under quadratic products.
        cutoff = self.N / 3.0
        return (np.abs(IX) < cutoff) & (np.abs(IY) < cutoff)


@dataclass
class RSWState:
    u: Array
    v: Array
    eta: Array

    def copy(self) -> "RSWState":
        return RSWState(self.u.copy(), self.v.copy(), self.eta.copy())

    def as_array(self) -> Array:
        return np.stack([self.u, self.v, self.eta], axis=0)

    @classmethod
    def from_array(cls, arr: Array) -> "RSWState":
        return cls(arr[0].copy(), arr[1].copy(), arr[2].copy())


def fft2(a: Array) -> Array:
    return np.fft.fft2(a)


def ifft2(a_hat: Array) -> Array:
    return np.fft.ifft2(a_hat).real


def deriv_x(a: Array, grid: SpectralGrid) -> Array:
    return ifft2(1j * grid.KX * fft2(a))


def deriv_y(a: Array, grid: SpectralGrid) -> Array:
    return ifft2(1j * grid.KY * fft2(a))


def dealias_field(a: Array, grid: SpectralGrid) -> Array:
    return ifft2(fft2(a) * grid.dealias_mask)


def dealias_state(state: RSWState, grid: SpectralGrid) -> RSWState:
    return RSWState(
        dealias_field(state.u, grid),
        dealias_field(state.v, grid),
        dealias_field(state.eta, grid),
    )


class PseudospectralRSWSolver:
    """Periodic pseudospectral 2D rotating shallow-water solver."""

    def __init__(self, grid: SpectralGrid, params: RSWParameters):
        self.grid = grid
        self.params = params

    def rhs(self, state: RSWState) -> RSWState:
        p = self.params
        g = self.grid

        u = state.u
        v = state.v
        eta = state.eta

        ux = deriv_x(u, g)
        uy = deriv_y(u, g)
        vx = deriv_x(v, g)
        vy = deriv_y(v, g)
        etax = deriv_x(eta, g)
        etay = deriv_y(eta, g)

        du = -(u * ux + v * uy) + p.f * v - p.g * etax
        dv = -(u * vx + v * vy) - p.f * u - p.g * etay

        h = p.H + eta
        deta = -(deriv_x(h * u, g) + deriv_y(h * v, g))

        if p.nu:
            lap = -g.K2
            du += p.nu * ifft2(lap * fft2(u))
            dv += p.nu * ifft2(lap * fft2(v))
            deta += p.nu * ifft2(lap * fft2(eta))

        return dealias_state(RSWState(du, dv, deta), g)

    def step_rk4(self, state: RSWState, dt: float) -> RSWState:
        def add_scaled(a: RSWState, b: RSWState, scale: float) -> RSWState:
            return RSWState(a.u + scale * b.u, a.v + scale * b.v, a.eta + scale * b.eta)

        k1 = self.rhs(state)
        k2 = self.rhs(add_scaled(state, k1, 0.5 * dt))
        k3 = self.rhs(add_scaled(state, k2, 0.5 * dt))
        k4 = self.rhs(add_scaled(state, k3, dt))

        out = RSWState(
            state.u + dt * (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u) / 6.0,
            state.v + dt * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v) / 6.0,
            state.eta + dt * (k1.eta + 2.0 * k2.eta + 2.0 * k3.eta + k4.eta) / 6.0,
        )
        return dealias_state(out, self.grid)

    def integrate(
        self,
        state0: RSWState,
        dt: float,
        t_end: float,
        snapshot_stride: int = 1,
        progress: Callable[[int, int, float], None] | None = None,
    ) -> tuple[Array, Array]:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if snapshot_stride <= 0:
            raise ValueError("snapshot_stride must be positive")

        nsteps = int(np.ceil(t_end / dt))
        state = dealias_state(state0.copy(), self.grid)

        times: list[float] = [0.0]
        snapshots: list[Array] = [state.as_array()]

        for step in range(1, nsteps + 1):
            local_dt = min(dt, t_end - (step - 1) * dt)
            state = self.step_rk4(state, local_dt)
            t = min(step * dt, t_end)
            if step % snapshot_stride == 0 or step == nsteps:
                times.append(t)
                snapshots.append(state.as_array())
            if progress is not None and (step == 1 or step == nsteps or step % max(1, nsteps // 10) == 0):
                progress(step, nsteps, t)

        return np.asarray(times), np.asarray(snapshots)


def rms(a: Array) -> float:
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def state_rms(state: RSWState) -> float:
    arr = state.as_array()
    return float(np.sqrt(np.mean(arr * arr)))
