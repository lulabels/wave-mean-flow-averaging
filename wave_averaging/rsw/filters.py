from __future__ import annotations

import numpy as np

CAUSAL_EXPONENTIAL_NAMES = {"causal_exponential", "exponential", "exponential_memory"}


def trapezoid(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy-version-compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def is_causal_exponential(name: str) -> bool:
    return name in CAUSAL_EXPONENTIAL_NAMES


def window_values(name: str, tau: np.ndarray, T: float) -> np.ndarray:
    """Unnormalized compact window on tau in [0,T]."""
    if is_causal_exponential(name):
        raise ValueError(
            f"{name!r} is an infinite-support memory kernel; use normalized_window or transfer_function"
        )
    x = np.asarray(tau, dtype=float) / T
    if name == "boxcar":
        w = np.ones_like(x)
    elif name == "hann":
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * x)
    elif name == "bump":
        s = 2.0 * x - 1.0
        inside = np.abs(s) < 1.0
        w = np.zeros_like(x)
        w[inside] = np.exp(-1.0 / (1.0 - s[inside] * s[inside]))
    elif name == "gaussian":
        sigma = T / 6.0
        w = np.exp(-0.5 * ((tau - 0.5 * T) / sigma) ** 2)
    elif name in {"triangle", "autocorrelation"}:
        w = np.maximum(1.0 - np.abs(2.0 * x - 1.0), 0.0)
    else:
        raise ValueError(f"unknown filter {name!r}")
    return np.where((x >= 0.0) & (x <= 1.0), w, 0.0)


def normalized_window(name: str, times: np.ndarray, t_end: float, T: float) -> tuple[np.ndarray, np.ndarray]:
    if T <= 0.0:
        raise ValueError("T must be positive")
    times = np.asarray(times, dtype=float)
    if is_causal_exponential(name):
        mask = times <= t_end + 1.0e-12
        if np.count_nonzero(mask) < 2:
            raise ValueError("window contains fewer than two snapshots")
        local_times = times[mask]
        lag = t_end - local_times
        raw = np.exp(-lag / T)
        dt = np.gradient(local_times)
        weights = raw * dt
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError(f"filter {name!r} has zero discrete mass")
        return mask, weights / total

    mask = (times >= t_end - T - 1.0e-12) & (times <= t_end + 1.0e-12)
    if np.count_nonzero(mask) < 2:
        raise ValueError("window contains fewer than two snapshots")

    local_times = times[mask]
    tau = local_times - local_times[0]
    actual_T = local_times[-1] - local_times[0]
    raw = window_values(name, tau, max(actual_T, 1.0e-14))
    dt = np.gradient(local_times)
    weights = raw * dt
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"filter {name!r} has zero discrete mass")
    return mask, weights / total


def discrete_window_span(times: np.ndarray, t_end: float, T: float, name: str | None = None) -> tuple[float, int]:
    """Actual span and snapshot count used by the discrete averaging weights."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    times = np.asarray(times, dtype=float)

    if name is not None and is_causal_exponential(name):
        mask = times <= t_end + 1.0e-12
        if np.count_nonzero(mask) < 2:
            raise ValueError("window contains fewer than two snapshots")
        local_times = times[mask]
        return float(local_times[-1] - local_times[0]), int(local_times.size)

    mask = (times >= t_end - T - 1.0e-12) & (times <= t_end + 1.0e-12)
    if np.count_nonzero(mask) < 2:
        raise ValueError("window contains fewer than two snapshots")
    local_times = times[mask]
    return float(local_times[-1] - local_times[0]), int(local_times.size)


def apply_window(samples: np.ndarray, times: np.ndarray, name: str, T: float, t_end: float | None = None) -> np.ndarray:
    if t_end is None:
        t_end = float(times[-1])
    mask, weights = normalized_window(name, times, t_end, T)
    return np.tensordot(weights, samples[mask], axes=(0, 0))


def transfer_function(name: str, detuning: np.ndarray | float, T: float, nquad: int = 2048) -> np.ndarray:
    detuning_arr = np.asarray(detuning, dtype=float)
    if is_causal_exponential(name):
        return 1.0 / (1.0 - 1j * detuning_arr * T)
    tau = np.linspace(0.0, T, nquad)
    raw = window_values(name, tau, T)
    weights = raw / trapezoid(raw, tau)
    phase = np.exp(1j * detuning_arr[..., None] * tau)
    return trapezoid(phase * weights, tau, axis=-1)


def verify_filter_normalization(names: list[str], T: float = 1.0) -> dict[str, float]:
    tau = np.linspace(0.0, T, 4097)
    out: dict[str, float] = {}
    for name in names:
        if is_causal_exponential(name):
            out[f"{name}_mass_error"] = 0.0
            continue
        raw = window_values(name, tau, T)
        mass = trapezoid(raw / trapezoid(raw, tau), tau)
        out[f"{name}_mass_error"] = float(abs(mass - 1.0))
    return out
