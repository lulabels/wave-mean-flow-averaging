r"""Modal RSW interactions and the optional finite-window evolution API.

ModalSystem and CompactModalSystem expose
    finite_window_rhs(amplitudes, time=t, epsilon=..., window_name=...,
                      window_length=..., window_origin=...)
    rk4_step(amplitudes, dt, "finite", time=t, ...)
Here t is physical time; each RK4 stage evaluates the phase at its own time.
ModalSystem.integrate_rk4 also accepts t_start and the absolute final time t_end.

Use window_origin="centered" for centered compact windows, or "trailing" for
past-facing windows and the causal exponential. The low-level "forward" default
is retained for the Figure 8 initial-state diagnostic. time_unit controls window
length, not the units of the required current time. Weight caches depend on the
origin; interaction tables do not.

This implements the closed averaged equation in section 3 and Appendix C.
Small-grid evolution checks are in tests/test_modal_evolution.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from .filters import is_causal_exponential, transfer_function, trapezoid, window_values
from .solver_pseudospectral import RSWParameters, RSWState, SpectralGrid, fft2


Array = np.ndarray
Branch = Literal[0, 1, -1]
WindowTimeUnit = Literal["fast", "physical"]
WindowOrigin = Literal["forward", "centered", "trailing"]

BRANCHES: tuple[Branch, ...] = (0, 1, -1)
STRICT_RESONANCE_TOL = 1.0e-12


@dataclass(frozen=True)
class FourierMode:
    ix: int
    iy: int
    mx: int
    my: int
    kx: float
    ky: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.mx, self.my)


@dataclass(frozen=True)
class ModalIndex:
    mode: FourierMode
    sigma: Branch
    omega: float
    flat_index: int

    @property
    def key(self) -> tuple[int, int, Branch]:
        return (self.mode.mx, self.mode.my, self.sigma)


@dataclass(frozen=True)
class Interaction:
    c: int
    a: int
    b: int
    coefficient: complex
    delta: float


def energy_weight(params: RSWParameters) -> Array:
    return np.diag([params.H, params.H, params.g]).astype(complex)


def linear_matrix(kx: float, ky: float, params: RSWParameters) -> Array:
    return np.asarray(
        [
            [0.0, params.f, -1j * params.g * kx],
            [-params.f, 0.0, -1j * params.g * ky],
            [-1j * params.H * kx, -1j * params.H * ky, 0.0],
        ],
        dtype=complex,
    )


def branch_frequency(kx: float, ky: float, sigma: Branch, params: RSWParameters) -> float:
    if sigma == 0:
        return 0.0
    k2 = kx * kx + ky * ky
    if k2 < 1.0e-30:
        return float(sigma * params.f)
    return float(sigma * np.sqrt(params.f * params.f + params.g * params.H * k2))


def right_eigenvector(kx: float, ky: float, sigma: Branch, params: RSWParameters) -> Array:
    k2 = kx * kx + ky * ky
    if k2 < 1.0e-30:
        if sigma == 0:
            return np.asarray([0.0, 0.0, 1.0], dtype=complex)
        if sigma == 1:
            return np.asarray([1.0, -1j, 0.0], dtype=complex)
        return np.asarray([1.0, 1j, 0.0], dtype=complex)

    if sigma == 0:
        return np.asarray([-1j * params.g * ky / params.f, 1j * params.g * kx / params.f, 1.0], dtype=complex)

    omega_abs = np.sqrt(params.f * params.f + params.g * params.H * k2)
    return np.asarray(
        [
            (sigma * omega_abs * kx + 1j * params.f * ky) / (params.H * k2),
            (sigma * omega_abs * ky - 1j * params.f * kx) / (params.H * k2),
            1.0,
        ],
        dtype=complex,
    )


def modal_matrices(kx: float, ky: float, params: RSWParameters) -> tuple[Array, Array]:
    r = np.column_stack([right_eigenvector(kx, ky, sigma, params) for sigma in BRANCHES])
    w = energy_weight(params)
    gram = r.conj().T @ w @ r
    lam = np.linalg.solve(gram, r.conj().T @ w)
    return r, lam


def ordered_pair_bilinear(ka: tuple[float, float], ra: Array, kb: tuple[float, float], rb: Array) -> Array:
    kcx = ka[0] + kb[0]
    kcy = ka[1] + kb[1]
    advective_factor = ra[0] * kb[0] + ra[1] * kb[1]
    return np.asarray(
        [
            -1j * advective_factor * rb[0],
            -1j * advective_factor * rb[1],
            -1j * (kcx * ra[2] * rb[0] + kcy * ra[2] * rb[1]),
        ],
        dtype=complex,
    )


@dataclass
class ModalSystem:
    """Modal finite truncation for the 2D RSW nonlinear interaction.

    Amplitudes use mathematical Fourier-series coefficients, not raw NumPy
    FFT coefficients. The retained branch set includes all sigma in {0,+,-}
    at every dealiased wave number, including the k=0 inertial branches.
    """

    grid: SpectralGrid
    params: RSWParameters
    resonance_tol: float = STRICT_RESONANCE_TOL

    @cached_property
    def modes(self) -> tuple[FourierMode, ...]:
        mx_1d = np.rint(np.fft.fftfreq(self.grid.N, d=1.0 / self.grid.N)).astype(int)
        my_1d = np.rint(np.fft.fftfreq(self.grid.N, d=1.0 / self.grid.N)).astype(int)
        out: list[FourierMode] = []
        for ix, mx in enumerate(mx_1d):
            for iy, my in enumerate(my_1d):
                if self.grid.dealias_mask[ix, iy]:
                    out.append(
                        FourierMode(
                            ix=ix,
                            iy=iy,
                            mx=int(mx),
                            my=int(my),
                            kx=float(self.grid.kx_1d[ix]),
                            ky=float(self.grid.ky_1d[iy]),
                        )
                    )
        return tuple(out)

    @cached_property
    def mode_by_key(self) -> dict[tuple[int, int], FourierMode]:
        return {mode.key: mode for mode in self.modes}

    @cached_property
    def modal_indices(self) -> tuple[ModalIndex, ...]:
        out: list[ModalIndex] = []
        for mode in self.modes:
            for sigma in BRANCHES:
                out.append(
                    ModalIndex(
                        mode=mode,
                        sigma=sigma,
                        omega=branch_frequency(mode.kx, mode.ky, sigma, self.params),
                        flat_index=len(out),
                    )
                )
        return tuple(out)

    @cached_property
    def index_by_key(self) -> dict[tuple[int, int, Branch], int]:
        return {idx.key: idx.flat_index for idx in self.modal_indices}

    @cached_property
    def omega(self) -> Array:
        return np.asarray([idx.omega for idx in self.modal_indices], dtype=float)

    @cached_property
    def mean_inertial_indices(self) -> tuple[int, int]:
        return (self.index_by_key[(0, 0, 1)], self.index_by_key[(0, 0, -1)])

    @cached_property
    def r_by_mode(self) -> dict[tuple[int, int], Array]:
        return {mode.key: modal_matrices(mode.kx, mode.ky, self.params)[0] for mode in self.modes}

    @cached_property
    def lambda_by_mode(self) -> dict[tuple[int, int], Array]:
        return {mode.key: modal_matrices(mode.kx, mode.ky, self.params)[1] for mode in self.modes}

    @cached_property
    def interactions(self) -> tuple[Interaction, ...]:
        out: list[Interaction] = []
        for c_idx in self.modal_indices:
            mc = c_idx.mode
            lambda_row = self.lambda_by_mode[mc.key][BRANCHES.index(c_idx.sigma)]
            for a_idx in self.modal_indices:
                ma = a_idx.mode
                mb_key = (mc.mx - ma.mx, mc.my - ma.my)
                mb = self.mode_by_key.get(mb_key)
                if mb is None:
                    continue
                for sigma_b in BRANCHES:
                    b_flat = self.index_by_key[(mb.mx, mb.my, sigma_b)]
                    b_idx = self.modal_indices[b_flat]
                    ra = self.r_by_mode[ma.key][:, BRANCHES.index(a_idx.sigma)]
                    rb = self.r_by_mode[mb.key][:, BRANCHES.index(sigma_b)]
                    vec = ordered_pair_bilinear((ma.kx, ma.ky), ra, (mb.kx, mb.ky), rb)
                    coeff = complex(lambda_row @ vec)
                    delta = float(a_idx.omega + b_idx.omega - c_idx.omega)
                    out.append(Interaction(c=c_idx.flat_index, a=a_idx.flat_index, b=b_flat, coefficient=coeff, delta=delta))
        return tuple(out)

    def zero_amplitudes(self) -> Array:
        return np.zeros(len(self.modal_indices), dtype=complex)

    def modal_to_fourier(self, amplitudes: Array) -> Array:
        amplitudes = np.asarray(amplitudes, dtype=complex)
        if amplitudes.shape != (len(self.modal_indices),):
            raise ValueError(f"expected amplitude shape {(len(self.modal_indices),)}, got {amplitudes.shape}")
        q_hat = np.zeros((3, self.grid.N, self.grid.N), dtype=complex)
        for mode in self.modes:
            local_amp = np.asarray([amplitudes[self.index_by_key[(mode.mx, mode.my, sigma)]] for sigma in BRANCHES])
            q_hat[:, mode.ix, mode.iy] = self.r_by_mode[mode.key] @ local_amp
        return q_hat

    def fourier_to_modal(self, q_hat: Array) -> Array:
        q_hat = np.asarray(q_hat, dtype=complex)
        if q_hat.shape != (3, self.grid.N, self.grid.N):
            raise ValueError(f"expected Fourier shape {(3, self.grid.N, self.grid.N)}, got {q_hat.shape}")
        amplitudes = self.zero_amplitudes()
        for mode in self.modes:
            local_q = q_hat[:, mode.ix, mode.iy]
            local_amp = self.lambda_by_mode[mode.key] @ local_q
            for j, sigma in enumerate(BRANCHES):
                amplitudes[self.index_by_key[(mode.mx, mode.my, sigma)]] = local_amp[j]
        return amplitudes

    def state_to_modal(self, state: RSWState) -> Array:
        scale = float(self.grid.N * self.grid.N)
        q_hat = np.stack([fft2(state.u), fft2(state.v), fft2(state.eta)], axis=0) / scale
        q_hat = q_hat * self.grid.dealias_mask[None, :, :]
        return self.fourier_to_modal(q_hat)

    def modal_to_state(self, amplitudes: Array, *, real: bool = True) -> RSWState:
        q_hat = self.modal_to_fourier(amplitudes)
        scale = float(self.grid.N * self.grid.N)
        arr = np.fft.ifft2(q_hat * scale, axes=(-2, -1))
        if real:
            arr = arr.real
        return RSWState(arr[0].copy(), arr[1].copy(), arr[2].copy())

    def slow_project_fourier(self, q_hat: Array) -> Array:
        q_hat = np.asarray(q_hat, dtype=complex)
        out = np.zeros_like(q_hat, dtype=complex)
        for mode in self.modes:
            amp0 = (self.lambda_by_mode[mode.key] @ q_hat[:, mode.ix, mode.iy])[0]
            out[:, mode.ix, mode.iy] = self.r_by_mode[mode.key][:, 0] * amp0
        return out

    def nonlinear_rhs_modal(self, amplitudes: Array) -> Array:
        amplitudes = np.asarray(amplitudes, dtype=complex)
        rhs = self.zero_amplitudes()
        for interaction in self.interactions:
            rhs[interaction.c] += interaction.coefficient * amplitudes[interaction.a] * amplitudes[interaction.b]
        return rhs

    def exact_resonant_rhs(self, amplitudes: Array, *, epsilon: float = 1.0) -> Array:
        _validate_epsilon(epsilon)
        amplitudes = np.asarray(amplitudes, dtype=complex)
        rhs = self.zero_amplitudes()
        for interaction in self.interactions:
            if abs(interaction.delta) <= self.resonance_tol:
                rhs[interaction.c] += interaction.coefficient * amplitudes[interaction.a] * amplitudes[interaction.b]
        return rhs

    def finite_window_rhs(
        self,
        amplitudes: Array,
        *,
        time: float,
        epsilon: float,
        window_name: str,
        window_length: float,
        time_unit: WindowTimeUnit = "fast",
        nquad: int = 2048,
        window_origin: WindowOrigin = "forward",
    ) -> Array:
        """Current-phase averaged RHS; ``time`` is always physical time.

        The forward origin is retained for historical diagnostic compatibility.
        Prototype drivers explicitly select centered or trailing offsets.
        """
        _validate_epsilon(epsilon)
        if window_length <= 0.0:
            raise ValueError("window_length must be positive")
        amplitudes = np.asarray(amplitudes, dtype=complex)
        phase = current_wave_phase(self.omega, time=time, epsilon=epsilon)
        phased = amplitudes * phase
        rhs = self.zero_amplitudes()
        multiplier_cache: dict[float, complex] = {}
        for interaction in self.interactions:
            if interaction.delta not in multiplier_cache:
                multiplier_cache[interaction.delta] = finite_window_multiplier(
                    window_name,
                    interaction.delta,
                    window_length=window_length,
                    epsilon=epsilon,
                    time_unit=time_unit,
                    nquad=nquad,
                    window_origin=window_origin,
                )
            rhs[interaction.c] += (
                interaction.coefficient
                * multiplier_cache[interaction.delta]
                * phased[interaction.a]
                * phased[interaction.b]
            )
        return rhs * np.conjugate(phase)

    def rk4_step(self, amplitudes: Array, dt: float, rhs_kind: Literal["exact", "finite"], *, time: float, **rhs_kwargs: object) -> Array:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        def f(a: Array, stage_time: float) -> Array:
            if rhs_kind == "exact":
                return self.exact_resonant_rhs(a, **rhs_kwargs)
            if rhs_kind == "finite":
                return self.finite_window_rhs(a, time=stage_time, **rhs_kwargs)
            raise ValueError(f"unknown rhs_kind {rhs_kind!r}")

        k1 = f(amplitudes, time)
        k2 = f(amplitudes + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = f(amplitudes + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = f(amplitudes + dt * k3, time + dt)
        return amplitudes + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def integrate_rk4(
        self,
        amplitudes0: Array,
        *,
        dt: float,
        t_end: float,
        rhs_kind: Literal["exact", "finite"],
        t_start: float = 0.0,
        **rhs_kwargs: object,
    ) -> tuple[Array, Array]:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end < t_start:
            raise ValueError("finite t_end must be at least finite t_start")
        duration = t_end - t_start
        nsteps = int(np.ceil(duration / dt)) if duration else 0
        amplitudes = np.asarray(amplitudes0, dtype=complex).copy()
        times = [t_start]
        snapshots = [amplitudes.copy()]
        for step in range(1, nsteps + 1):
            time = t_start + (step - 1) * dt
            local_dt = min(dt, t_end - time)
            amplitudes = self.rk4_step(amplitudes, local_dt, rhs_kind, time=time, **rhs_kwargs)
            times.append(min(t_start + step * dt, t_end))
            snapshots.append(amplitudes.copy())
        return np.asarray(times), np.asarray(snapshots)

    def conjugate_symmetric_amplitudes(self, rng: np.random.Generator, scale: float = 1.0) -> Array:
        amplitudes = self.zero_amplitudes()
        visited: set[tuple[int, int, Branch]] = set()
        for idx in self.modal_indices:
            key = idx.key
            if key in visited:
                continue
            if idx.sigma == 0:
                partner_key: tuple[int, int, Branch] = (-idx.mode.mx, -idx.mode.my, 0)
            else:
                partner_key = (-idx.mode.mx, -idx.mode.my, -idx.sigma)  # type: ignore[arg-type]
            partner = self.index_by_key.get(partner_key)
            if partner is None:
                continue
            if partner == idx.flat_index:
                value = scale * rng.normal()
            else:
                value = scale * (rng.normal() + 1j * rng.normal())
            amplitudes[idx.flat_index] = value
            amplitudes[partner] = np.conjugate(value)
            visited.add(key)
            visited.add(partner_key)
        return amplitudes

    def find_wave_exact_resonances(self, *, coefficient_tol: float = 1.0e-12) -> list[Interaction]:
        out: list[Interaction] = []
        for interaction in self.interactions:
            if abs(interaction.delta) > self.resonance_tol or abs(interaction.coefficient) <= coefficient_tol:
                continue
            sigmas = (
                self.modal_indices[interaction.c].sigma,
                self.modal_indices[interaction.a].sigma,
                self.modal_indices[interaction.b].sigma,
            )
            if any(sigma != 0 for sigma in sigmas):
                out.append(interaction)
        return out


def finite_window_multiplier(
    window_name: str,
    delta: float,
    *,
    window_length: float,
    epsilon: float,
    time_unit: WindowTimeUnit = "fast",
    nquad: int = 2048,
    window_origin: WindowOrigin = "forward",
) -> complex:
    """Return the multiplier for explicit offsets from the current phase.

    Compact offsets occupy [0,T], [-T/2,T/2], or [-T,0]. A trailing
    exponential uses negative offsets and a positive memory timescale T.
    The default forward convention preserves historical diagnostic weights.
    """

    _validate_epsilon(epsilon)
    validate_window_origin(window_name, window_origin)
    if not np.isfinite(window_length) or window_length <= 0.0:
        raise ValueError("window_length must be positive and finite")
    if time_unit == "fast":
        rate = delta
    elif time_unit == "physical":
        rate = delta / epsilon
    else:
        raise ValueError("time_unit must be 'fast' or 'physical'")
    weight = complex(transfer_function(window_name, -rate, window_length, nquad=nquad))
    if window_origin == "forward":
        return weight
    if is_causal_exponential(window_name):
        return np.conjugate(weight)
    shift = window_length / 2.0 if window_origin == "centered" else window_length
    return weight * np.exp(1j * rate * shift)


def validate_window_origin(window_name: str, window_origin: WindowOrigin) -> None:
    if window_origin not in {"forward", "centered", "trailing"}:
        raise ValueError("window_origin must be 'forward', 'centered' or 'trailing'")
    if is_causal_exponential(window_name) and window_origin == "centered":
        raise ValueError("a causal exponential requires trailing offsets (or historical forward offsets)")


def current_wave_phase(omega: Array, *, time: float, epsilon: float) -> Array:
    _validate_epsilon(epsilon)
    if not np.isfinite(time):
        raise ValueError("time must be finite physical time")
    return np.exp(-1j * omega * (time / epsilon))


def nonlinear_rhs_fourier(q_hat: Array, grid: SpectralGrid) -> Array:
    """Complex pseudospectral quadratic RHS in Fourier-series normalization."""

    q_hat = np.asarray(q_hat, dtype=complex)
    if q_hat.shape != (3, grid.N, grid.N):
        raise ValueError(f"expected Fourier shape {(3, grid.N, grid.N)}, got {q_hat.shape}")
    scale = float(grid.N * grid.N)
    u, v, eta = [np.fft.ifft2(q_hat[j] * scale) for j in range(3)]
    ux = np.fft.ifft2(1j * grid.KX * q_hat[0] * scale)
    uy = np.fft.ifft2(1j * grid.KY * q_hat[0] * scale)
    vx = np.fft.ifft2(1j * grid.KX * q_hat[1] * scale)
    vy = np.fft.ifft2(1j * grid.KY * q_hat[1] * scale)
    n_u = -(u * ux + v * uy)
    n_v = -(u * vx + v * vy)
    n_eta = -(
        np.fft.ifft2(1j * grid.KX * fft2(eta * u))
        + np.fft.ifft2(1j * grid.KY * fft2(eta * v))
    )
    out = np.stack([fft2(n_u), fft2(n_v), fft2(n_eta)], axis=0) / scale
    return out * grid.dealias_mask[None, :, :]


def direct_phase_average_rhs(
    system: ModalSystem,
    amplitudes: Array,
    *,
    s_end: float,
    nquad: int,
) -> Array:
    if s_end <= 0.0:
        raise ValueError("s_end must be positive")
    s = np.linspace(0.0, s_end, nquad)
    vals = np.empty((nquad, len(system.modal_indices)), dtype=complex)
    omega = system.omega
    for j, sj in enumerate(s):
        phased = amplitudes * np.exp(-1j * omega * sj)
        vals[j] = system.nonlinear_rhs_modal(phased) * np.exp(1j * omega * sj)
    return trapezoid(vals, s, axis=0) / s_end


def direct_finite_window_rhs(
    system: ModalSystem,
    amplitudes: Array,
    *,
    time: float,
    epsilon: float,
    window_name: str,
    window_length: float,
    time_unit: WindowTimeUnit = "fast",
    nquad: int = 2048,
    window_origin: WindowOrigin = "forward",
) -> Array:
    _validate_epsilon(epsilon)
    validate_window_origin(window_name, window_origin)
    if not np.isfinite(window_length) or window_length <= 0.0 or nquad < 2:
        raise ValueError("positive finite window_length and at least two quadrature nodes are required")
    current_phase = current_wave_phase(system.omega, time=time, epsilon=epsilon)
    if time_unit == "fast":
        phase_rate = system.omega
    elif time_unit == "physical":
        phase_rate = system.omega / epsilon
    else:
        raise ValueError("time_unit must be 'fast' or 'physical'")

    if is_causal_exponential(window_name):
        tau = np.linspace(0.0, 32.0 * window_length, nquad)
        raw = np.exp(-tau / window_length)
        offsets = -tau if window_origin == "trailing" else tau
    else:
        tau = np.linspace(0.0, window_length, nquad)
        raw = window_values(window_name, tau, window_length)
        shift = {"forward": 0.0, "centered": window_length / 2.0, "trailing": window_length}[window_origin]
        offsets = tau - shift
    weights = raw / trapezoid(raw, tau)
    vals = np.empty((nquad, len(system.modal_indices)), dtype=complex)
    for j, offset in enumerate(offsets):
        phase = current_phase * np.exp(-1j * phase_rate * offset)
        phased = amplitudes * phase
        vals[j] = system.nonlinear_rhs_modal(phased) * np.conjugate(phase)
    return trapezoid(vals * weights[:, None], tau, axis=0)


def _validate_epsilon(epsilon: float) -> None:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
