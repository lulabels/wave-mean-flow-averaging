r"""Two-wave Rossby displacement and drift for Figures 5 and 6.

Setting a1=0 recovers Figure 3's base wave and sampling. sigma contains the
positive phase rates called omega in the paper: k*x+m*z+sigma*t in the
background-following coordinate. omega contains the laboratory rates in
k*x_lab-omega*t.

Drift is evaluated at fixed horizontal reference points. selected_drift
keeps exact self terms and the boxcar-weighted difference term; full_boxcar
also includes sum and double frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


XY_GAIN = 2.0
YZ_GAIN = 1.8
X_STRETCH = 4.0
Z_CENTER = 0.52
DRIFT_GAIN = 1.706773754141205

# Figure 3 sampling, kept here so the model needs no plotting-script import.
REFERENCE_PHASE = np.linspace(0.0, 2.0 * math.pi, 721)
REFERENCE_XY_CENTERS = np.linspace(0.14, 0.86, 6)
REFERENCE_YZ_CENTERS = np.linspace(0.16, 0.84, 6)


@dataclass(frozen=True)
class Parameters:
    """Parameters for the two-wave Rossby model."""

    k0: float = 2.0
    k1: float = 2.5
    ell: float = 1.0
    H: float = 1.4
    m: float = 1.1
    a0: float = 1.0
    a1: float = 0.10
    Ubar: float = 10.0
    f0: float = 1.0
    N2: float = 10.0
    z_ref: float = 0.2

    @property
    def h(self) -> float:
        return 1.0 / (2.0 * self.H)

    @property
    def k(self) -> np.ndarray:
        return np.array([self.k0, self.k1])

    @property
    def amplitude(self) -> np.ndarray:
        return np.array([self.a0, self.a1])

    @property
    def D(self) -> np.ndarray:
        return self.k**2 + self.ell**2 + (self.f0**2 / self.N2) * (
            self.m**2 + self.h**2
        )

    @property
    def beta(self) -> float:
        return self.Ubar * float(self.D[0])

    @property
    def omega(self) -> np.ndarray:
        """Laboratory rates in ``cos(k*x_lab - omega*t)``."""
        return self.k * self.Ubar - self.beta * self.k / self.D

    @property
    def sigma(self) -> np.ndarray:
        """Positive background-following phase rates (paper's omega)."""
        return self.k * self.Ubar - self.omega

    @property
    def delta_sigma(self) -> float:
        return float(self.sigma[1] - self.sigma[0])

    @property
    def duration(self) -> float:
        """One base-wave period in the background-following coordinate."""
        return 2.0 * math.pi / abs(float(self.sigma[0]))

    @property
    def difference_weight(self) -> float:
        value = 0.5 * self.delta_sigma * self.duration
        return float(np.sinc(value / math.pi))

    @property
    def q_ref(self) -> float:
        """Fixed reference position preserving Figure 3's base phase."""
        return -self.m * self.z_ref / self.k0


P = Parameters()


def sinc(value: np.ndarray | float) -> np.ndarray:
    """Return the unnormalised sinc, ``sin(value)/value``."""
    return np.sinc(np.asarray(value, float) / math.pi)


def phases(q, z, time, p: Parameters = P) -> np.ndarray:
    """Return both positive-rate phases in the background-following frame."""
    q, z, time = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(z, float), np.asarray(time, float)
    )
    return q[..., None] * p.k + p.m * z[..., None] + time[..., None] * p.sigma


def streamfunction(q, y, z, time=0.0, p: Parameters = P) -> np.ndarray:
    """Two-wave streamfunction in the background-following coordinate."""
    theta = phases(q, z, time, p)
    return (
        math.exp(z / (2.0 * p.H))
        * np.sin(p.ell * y)
        * np.sum(p.amplitude * np.cos(theta), axis=-1)
    )


def textbook_base_displacement(
    centers, phase, p: Parameters = P
) -> np.ndarray:
    """Operation-for-operation public Figure 3 first-order displacement."""
    s = np.asarray(centers, float)[:, None]
    theta = np.asarray(phase, float)[None, :]
    ly = math.pi * s
    amplitude = p.a0 * math.exp(p.z_ref / (2.0 * p.H))
    xi = -(p.ell * amplitude / (p.k0 * p.Ubar)) * np.sin(theta) * np.cos(ly)
    eta = +(amplitude / p.Ubar) * np.cos(theta) * np.sin(ly)
    zeta = -(p.f0 * amplitude / p.N2) * (
        np.cos(theta) / (2.0 * p.H) - p.m * np.sin(theta)
    ) * np.sin(ly)
    return np.stack([xi, eta, zeta], axis=2)


def second_displacement(
    centers, base_phase, p: Parameters = P
) -> np.ndarray:
    """First-order displacement of the added mode at the same reference."""
    y = math.pi * np.asarray(centers, float)[:, None]
    initial = (p.k1 - p.k0) * p.q_ref
    theta = initial + (p.sigma[1] / p.sigma[0]) * np.asarray(base_phase)[None, :]
    envelope = math.exp(p.z_ref / (2.0 * p.H))
    sy, cy = np.sin(p.ell * y), np.cos(p.ell * y)
    xi = -(p.a1 * p.ell * envelope / p.sigma[1]) * cy * np.sin(theta)
    eta = +(p.a1 * p.k1 * envelope / p.sigma[1]) * sy * np.cos(theta)
    zeta = -(p.f0 * p.a1 * envelope / p.N2) * sy * (
        p.h * np.cos(theta) - p.m * np.sin(theta)
    )
    return np.stack([xi, eta, zeta], axis=2)


def exact_drift(q, y, z, p: Parameters = P) -> np.ndarray:
    """Infinite phase-average of the fixed-reference horizontal diagnostic."""
    q, y = np.broadcast_arrays(np.asarray(q, float), np.asarray(y, float))
    coefficient = np.sum(p.amplitude**2 * p.k / (2.0 * p.sigma))
    ux = -p.ell**2 * math.exp(z / p.H) * np.cos(2.0 * p.ell * y) * coefficient
    return np.stack([ux, np.zeros_like(q)], axis=-1)


def difference_coefficients(
    y, z, p: Parameters = P
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrised cross-pair coefficients at the difference frequency."""
    y = np.asarray(y, float)
    s, c = np.sin(p.ell * y), np.cos(p.ell * y)
    k0, k1 = p.k
    sigma0, sigma1 = p.sigma
    prefactor = p.a0 * p.a1 * math.exp(z / p.H)
    cx = 0.5 * prefactor * p.ell**2 * (
        (-k1 * c**2 + k0 * s**2) / sigma0
        + (-k0 * c**2 + k1 * s**2) / sigma1
    )
    cy = (
        0.5
        * prefactor
        * p.ell
        * s
        * c
        * (k0 + k1)
        * (k0 / sigma1 - k1 / sigma0)
    )
    return cx, cy


def difference_drift(q, y, z, time, p: Parameters = P) -> np.ndarray:
    """Unweighted difference-frequency cross-wave contribution."""
    q, y, time = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(y, float), np.asarray(time, float)
    )
    cx, cy = difference_coefficients(y, z, p)
    delta = (p.k1 - p.k0) * q + p.delta_sigma * time
    return np.stack([cx * np.cos(delta), cy * np.sin(delta)], axis=-1)


def sum_drift(q, y, z, time, p: Parameters = P) -> np.ndarray:
    """Unweighted cross-wave contribution at the sum frequency."""
    q, y, time = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(y, float), np.asarray(time, float)
    )
    s, c = np.sin(p.ell * y), np.cos(p.ell * y)
    k0, k1 = p.k
    sigma0, sigma1 = p.sigma
    prefactor = p.a0 * p.a1 * math.exp(z / p.H)
    cx = 0.5 * prefactor * p.ell**2 * (
        (k1 * c**2 + k0 * s**2) / sigma0
        + (k0 * c**2 + k1 * s**2) / sigma1
    )
    cy = (
        0.5
        * prefactor
        * p.ell
        * s
        * c
        * (k1 - k0)
        * (k1 / sigma0 - k0 / sigma1)
    )
    theta_sum = (k0 + k1) * q + 2.0 * p.m * z + (sigma0 + sigma1) * time
    return np.stack([cx * np.cos(theta_sum), cy * np.sin(theta_sum)], axis=-1)


def double_drift(q, y, z, time, p: Parameters = P) -> np.ndarray:
    """Unweighted self-pair contributions at twice each phase rate."""
    q, y, time = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(y, float), np.asarray(time, float)
    )
    theta = phases(q, z, time, p)
    coefficient = (
        p.amplitude**2
        * p.ell**2
        * math.exp(z / p.H)
        * p.k
        / (2.0 * p.sigma)
    )
    ux = np.sum(coefficient * np.cos(2.0 * theta), axis=-1)
    return np.stack([ux, np.zeros_like(ux)], axis=-1)


def instantaneous_quadratic(q, y, z, time, p: Parameters = P) -> np.ndarray:
    """Directly evaluate ``(xi_h dot grad_h) u_h`` from both modes."""
    q, y, time = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(y, float), np.asarray(time, float)
    )
    theta = phases(q, z, time, p)
    envelope = math.exp(z / (2.0 * p.H))
    s = np.sin(p.ell * y)[..., None]
    c = np.cos(p.ell * y)[..., None]
    xi = -(p.amplitude * p.ell * envelope / p.sigma) * c * np.sin(theta)
    eta = +(p.amplitude * p.k * envelope / p.sigma) * s * np.cos(theta)
    out = np.zeros(q.shape + (2,))
    for j in range(2):
        du_q = (
            p.amplitude[j]
            * p.ell
            * p.k[j]
            * envelope
            * c[..., 0]
            * np.sin(theta[..., j])
        )
        du_y = (
            p.amplitude[j]
            * p.ell**2
            * envelope
            * s[..., 0]
            * np.cos(theta[..., j])
        )
        dv_q = (
            -p.amplitude[j]
            * p.k[j] ** 2
            * envelope
            * s[..., 0]
            * np.cos(theta[..., j])
        )
        dv_y = (
            -p.amplitude[j]
            * p.k[j]
            * p.ell
            * envelope
            * c[..., 0]
            * np.sin(theta[..., j])
        )
        out[..., 0] += np.sum(
            xi * du_q[..., None] + eta * du_y[..., None], axis=-1
        )
        out[..., 1] += np.sum(
            xi * dv_q[..., None] + eta * dv_y[..., None], axis=-1
        )
    return out


def selected_drift(q, y, z, time, p: Parameters = P) -> np.ndarray:
    """Exact self terms plus the centered-boxcar difference term."""
    return exact_drift(q, y, z, p) + p.difference_weight * difference_drift(
        q, y, z, time, p
    )


def full_boxcar(q, y, z, center_time, p: Parameters = P) -> np.ndarray:
    """Complete centered boxcar control over one base-wave period."""
    result = selected_drift(q, y, z, center_time, p)
    result += float(sinc(0.5 * np.sum(p.sigma) * p.duration)) * sum_drift(
        q, y, z, center_time, p
    )
    theta = phases(q, z, center_time, p)
    for j in range(2):
        weight = float(sinc(p.sigma[j] * p.duration))
        coefficient = (
            p.amplitude[j] ** 2
            * p.ell**2
            * math.exp(z / p.H)
            * p.k[j]
            / (2.0 * p.sigma[j])
        )
        result[..., 0] += weight * coefficient * np.cos(2.0 * theta[..., j])
    return result


def drift_displacements(
    centers, phase, p: Parameters = P
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-reference exact and selected horizontal displacements."""
    time = np.asarray(phase, float) / p.sigma[0]
    y = math.pi * np.asarray(centers, float)
    exact_velocity = exact_drift(np.full_like(y, p.q_ref), y, p.z_ref, p)
    exact = exact_velocity[:, None, :] * time[None, :, None]
    cx, cy = difference_coefficients(y, p.z_ref, p)
    initial = (p.k1 - p.k0) * p.q_ref
    final_phase = initial + p.delta_sigma * time
    cross = np.empty_like(exact)
    cross[..., 0] = (
        p.difference_weight
        * cx[:, None]
        * (np.sin(final_phase)[None, :] - math.sin(initial))
        / p.delta_sigma
    )
    cross[..., 1] = (
        p.difference_weight
        * cy[:, None]
        * (math.cos(initial) - np.cos(final_phase)[None, :])
        / p.delta_sigma
    )
    return exact, exact + cross


def build_arrays(
    p: Parameters = P,
    *,
    phase: np.ndarray | None = None,
    xy_centers: np.ndarray | None = None,
    yz_centers: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build every physical and displayed array used by public Figure 5."""
    q = np.linspace(0.0, 2.0 * math.pi, 360)
    y = np.linspace(0.0, math.pi, 180)
    q_grid, y_grid = np.meshgrid(q, y, indexing="xy")
    fields = {
        "field_q": q,
        "field_y": y,
        "streamfunction": streamfunction(q_grid, y_grid, p.z_ref, 0.0, p),
        "exact_drift": exact_drift(q_grid, y_grid, p.z_ref, p),
        "selected_drift": selected_drift(q_grid, y_grid, p.z_ref, 0.0, p),
        "full_boxcar_drift": full_boxcar(q_grid, y_grid, p.z_ref, 0.0, p),
    }
    fields["selected_minus_exact"] = fields["selected_drift"] - fields["exact_drift"]
    fields["full_boxcar_minus_selected"] = (
        fields["full_boxcar_drift"] - fields["selected_drift"]
    )

    phase = np.array(REFERENCE_PHASE if phase is None else phase, copy=True)
    xy = np.array(REFERENCE_XY_CENTERS if xy_centers is None else xy_centers, copy=True)
    yz = np.array(REFERENCE_YZ_CENTERS if yz_centers is None else yz_centers, copy=True)
    time = phase / p.sigma[0]
    xy_base = textbook_base_displacement(xy, phase, p)
    yz_base = textbook_base_displacement(yz, phase, p)
    xy_two = xy_base + second_displacement(xy, phase, p)
    yz_two = yz_base + second_displacement(yz, phase, p)
    xy_exact, xy_selected = drift_displacements(xy, phase, p)
    yz_exact, yz_selected = drift_displacements(yz, phase, p)
    orbits = {
        "orbit_phase": phase,
        "orbit_time": time,
        "xy_centers": xy,
        "yz_centers": yz,
        "xy_base_displacement": xy_base,
        "yz_base_displacement": yz_base,
        "xy_two_wave_displacement": xy_two,
        "yz_two_wave_displacement": yz_two,
        "xy_exact_drift_displacement": xy_exact,
        "xy_selected_drift_displacement": xy_selected,
        "yz_exact_drift_displacement": yz_exact,
        "yz_selected_drift_displacement": yz_selected,
    }
    orbits.update(
        {
            "b_exact_x": X_STRETCH
            * XY_GAIN
            * (xy_two[..., 0] + DRIFT_GAIN * xy_exact[..., 0])
            / math.pi,
            "b_exact_y": xy[:, None]
            + XY_GAIN
            * (xy_two[..., 1] + DRIFT_GAIN * xy_exact[..., 1])
            / math.pi,
            "b_selected_x": X_STRETCH
            * XY_GAIN
            * (xy_two[..., 0] + DRIFT_GAIN * xy_selected[..., 0])
            / math.pi,
            "b_selected_y": xy[:, None]
            + XY_GAIN
            * (xy_two[..., 1] + DRIFT_GAIN * xy_selected[..., 1])
            / math.pi,
            "c_exact_y": yz[:, None]
            + YZ_GAIN
            * (yz_two[..., 1] + DRIFT_GAIN * yz_exact[..., 1])
            / math.pi,
            "c_selected_y": yz[:, None]
            + YZ_GAIN
            * (yz_two[..., 1] + DRIFT_GAIN * yz_selected[..., 1])
            / math.pi,
            "c_reference_z": Z_CENTER + YZ_GAIN * yz_two[..., 2] / math.pi,
            "pure_b_base_x": X_STRETCH * XY_GAIN * xy_base[..., 0] / math.pi,
            "pure_b_base_y": xy[:, None] + XY_GAIN * xy_base[..., 1] / math.pi,
            "pure_b_two_x": X_STRETCH * XY_GAIN * xy_two[..., 0] / math.pi,
            "pure_b_two_y": xy[:, None] + XY_GAIN * xy_two[..., 1] / math.pi,
            "pure_c_base_y": yz[:, None] + YZ_GAIN * yz_base[..., 1] / math.pi,
            "pure_c_base_z": Z_CENTER + YZ_GAIN * yz_base[..., 2] / math.pi,
            "pure_c_two_y": yz[:, None] + YZ_GAIN * yz_two[..., 1] / math.pi,
            "pure_c_two_z": Z_CENTER + YZ_GAIN * yz_two[..., 2] / math.pi,
        }
    )
    return fields, orbits


def array_payload(
    fields: dict[str, np.ndarray],
    orbits: dict[str, np.ndarray],
    p: Parameters = P,
) -> dict[str, np.ndarray]:
    """Return the Figure 5 fields, orbits and model parameters for saving."""
    payload = {**fields, **orbits}
    payload.update(
        {
            "parameter_k": p.k,
            "parameter_amplitude": p.amplitude,
            "parameter_D": p.D,
            "parameter_omega": p.omega,
            "parameter_sigma": p.sigma,
            "parameter_beta": np.array(p.beta),
            "parameter_delta_sigma": np.array(p.delta_sigma),
            "parameter_duration": np.array(p.duration),
            "parameter_difference_weight": np.array(p.difference_weight),
            "parameter_q_ref": np.array(p.q_ref),
            "parameter_z_ref": np.array(p.z_ref),
            "parameter_Ubar": np.array(p.Ubar),
            "parameter_H": np.array(p.H),
            "parameter_m": np.array(p.m),
            "parameter_ell": np.array(p.ell),
            "parameter_f0": np.array(p.f0),
            "parameter_N2": np.array(p.N2),
        }
    )
    return payload


def _reference_display_paths(
    p: Parameters = P,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the public Figure 3 paths without importing its renderer."""
    phase = REFERENCE_PHASE[None, :]
    amplitude = p.a0 * math.exp(p.z_ref / (2.0 * p.H))
    xy = REFERENCE_XY_CENTERS[:, None]
    yz = REFERENCE_YZ_CENTERS[:, None]
    xy_x = (
        -X_STRETCH
        * XY_GAIN
        * p.ell
        * amplitude
        * np.sin(phase)
        * np.cos(math.pi * xy)
        / (p.k0 * p.Ubar * math.pi)
    )
    xy_y = xy + (
        XY_GAIN
        * amplitude
        * np.cos(phase)
        * np.sin(math.pi * xy)
        / (p.Ubar * math.pi)
    )
    yz_y = yz + (
        YZ_GAIN
        * amplitude
        * np.cos(phase)
        * np.sin(math.pi * yz)
        / (p.Ubar * math.pi)
    )
    yz_z = Z_CENTER - (
        YZ_GAIN
        * p.f0
        * amplitude
        * (np.cos(phase) / (2.0 * p.H) - p.m * np.sin(phase))
        * np.sin(math.pi * yz)
        / (p.N2 * math.pi)
    )
    return xy_x, xy_y, yz_y, yz_z


def validate_arrays(
    fields: dict[str, np.ndarray],
    orbits: dict[str, np.ndarray],
    p: Parameters = P,
) -> dict[str, float | bool]:
    """Run one-wave, quadratic, quadrature, integration and bounds checks."""
    accepted_xy_x, accepted_xy_y, accepted_yz_y, accepted_yz_z = (
        _reference_display_paths(p)
    )
    base_xy_x = X_STRETCH * XY_GAIN * orbits["xy_base_displacement"][..., 0] / math.pi
    base_xy_y = orbits["xy_centers"][:, None] + XY_GAIN * orbits[
        "xy_base_displacement"
    ][..., 1] / math.pi
    base_yz_y = orbits["yz_centers"][:, None] + YZ_GAIN * orbits[
        "yz_base_displacement"
    ][..., 1] / math.pi
    base_yz_z = Z_CENTER + YZ_GAIN * orbits["yz_base_displacement"][..., 2] / math.pi
    xy_res = max(
        float(np.max(np.abs(base_xy_x - accepted_xy_x))),
        float(np.max(np.abs(base_xy_y - accepted_xy_y))),
    )
    yz_res = max(
        float(np.max(np.abs(base_yz_y - accepted_yz_y))),
        float(np.max(np.abs(base_yz_z - accepted_yz_z))),
    )

    p0 = Parameters(a1=0.0)
    q_grid, y_grid = np.meshgrid(fields["field_q"], fields["field_y"], indexing="xy")
    expected_snapshot = (
        p.a0
        * math.exp(p.z_ref / (2.0 * p.H))
        * np.cos(p.k0 * q_grid + p.m * p.z_ref)
        * np.sin(p.ell * y_grid)
    )
    snapshot_res = float(
        np.max(
            np.abs(
                streamfunction(q_grid, y_grid, p.z_ref, 0.0, p0)
                - expected_snapshot
            )
        )
    )
    y = np.linspace(0.07, math.pi - 0.07, 51)
    calculated = exact_drift(np.full_like(y, p.q_ref), y, p.z_ref, p0)[:, 0]
    expected = (
        -p.ell**2
        * p.a0**2
        * math.exp(p.z_ref / p.H)
        * np.cos(2.0 * p.ell * y)
        / (2.0 * p.Ubar)
    )
    drift_res = float(np.max(np.abs(calculated - expected)))

    rng = np.random.default_rng(20261026)
    q = rng.uniform(0.0, 2.0 * math.pi, 97)
    y = rng.uniform(0.04, math.pi - 0.04, 97)
    time = rng.uniform(-p.duration, p.duration, 97)
    direct = instantaneous_quadratic(q, y, p.z_ref, time, p)
    pieces = (
        exact_drift(q, y, p.z_ref, p)
        + difference_drift(q, y, p.z_ref, time, p)
        + sum_drift(q, y, p.z_ref, time, p)
        + double_drift(q, y, p.z_ref, time, p)
    )
    decomposition_res = float(np.max(np.abs(direct - pieces)))

    quadrature_q = np.array([0.13, 1.17, 4.81])
    quadrature_y = np.array([0.31, 1.22, 2.43])
    quadrature_time = np.linspace(-p.duration / 2.0, p.duration / 2.0, 40001)
    numerical_boxcar = []
    for q_value, y_value in zip(quadrature_q, quadrature_y):
        history = instantaneous_quadratic(
            np.full_like(quadrature_time, q_value),
            np.full_like(quadrature_time, y_value),
            p.z_ref,
            quadrature_time,
            p,
        )
        numerical_boxcar.append(
            np.trapezoid(history, quadrature_time, axis=0) / p.duration
        )
    boxcar_res = float(
        np.max(
            np.abs(
                np.asarray(numerical_boxcar)
                - full_boxcar(quadrature_q, quadrature_y, p.z_ref, 0.0, p)
            )
        )
    )

    integration_time = orbits["orbit_time"]
    centers = orbits["xy_centers"]
    physical_y = math.pi * centers
    history = selected_drift(
        np.full((len(physical_y), len(integration_time)), p.q_ref),
        np.broadcast_to(
            physical_y[:, None], (len(physical_y), len(integration_time))
        ),
        p.z_ref,
        np.broadcast_to(
            integration_time[None, :], (len(physical_y), len(integration_time))
        ),
        p,
    )
    numeric = np.zeros_like(history)
    numeric[:, 1:] = np.cumsum(
        0.5
        * (history[:, 1:] + history[:, :-1])
        * np.diff(integration_time)[None, :, None],
        axis=1,
    )
    integration_res = float(
        np.max(np.abs(numeric - orbits["xy_selected_drift_displacement"]))
    )

    inside = bool(
        np.min(orbits["b_exact_x"]) >= -0.26
        and np.min(orbits["b_selected_x"]) >= -0.26
        and np.max(orbits["b_exact_x"]) <= 0.26
        and np.max(orbits["b_selected_x"]) <= 0.26
        and np.min(orbits["b_exact_y"]) >= 0.0
        and np.min(orbits["b_selected_y"]) >= 0.0
        and np.max(orbits["b_exact_y"]) <= 1.0
        and np.max(orbits["b_selected_y"]) <= 1.0
        and np.min(orbits["c_exact_y"]) >= 0.03
        and np.min(orbits["c_selected_y"]) >= 0.03
        and np.max(orbits["c_exact_y"]) <= 0.97
        and np.max(orbits["c_selected_y"]) <= 0.97
        and np.min(orbits["c_reference_z"]) >= 0.31
        and np.max(orbits["c_reference_z"]) <= 0.73
    )
    return {
        "one_wave_xy_display_max_abs_residual": xy_res,
        "one_wave_yz_display_max_abs_residual": yz_res,
        "one_wave_snapshot_max_abs_residual": snapshot_res,
        "one_wave_AHL_4p5p25a_max_abs_residual": drift_res,
        "instantaneous_decomposition_max_abs_residual": decomposition_res,
        "full_boxcar_quadrature_max_abs_residual": boxcar_res,
        "selected_drift_integration_max_abs_residual": integration_res,
        "full_boxcar_minus_selected_max_abs": float(
            np.max(np.abs(fields["full_boxcar_minus_selected"]))
        ),
        "all_curves_inside_fixed_axes": inside,
    }
