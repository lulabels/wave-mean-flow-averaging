from __future__ import annotations

import numpy as np

from .filters import apply_window
from .projectors import slow_project_state
from .solver_pseudospectral import RSWParameters, RSWState, SpectralGrid


def relative_state_error(a: RSWState, b: RSWState) -> float:
    diff = a.as_array() - b.as_array()
    denom = np.sqrt(np.mean(b.as_array() ** 2))
    return float(np.sqrt(np.mean(diff * diff)) / max(denom, 1.0e-14))


def state_from_filtered_samples(samples: np.ndarray, times: np.ndarray, name: str, T: float) -> RSWState:
    return RSWState.from_array(apply_window(samples, times, name, T))


def long_window_references(
    samples: np.ndarray, times: np.ndarray, T: float, grid: SpectralGrid, params: RSWParameters
) -> tuple[RSWState, RSWState]:
    mean_state = RSWState.from_array(apply_window(samples, times, "hann", T))
    return mean_state, slow_project_state(mean_state, grid, params)


def long_window_reference(samples: np.ndarray, times: np.ndarray, T: float, grid: SpectralGrid, params: RSWParameters) -> RSWState:
    _, slow_reference = long_window_references(samples, times, T, grid, params)
    return slow_reference


def state_diagnostic_row(
    filter_name: str,
    T: float,
    samples: np.ndarray,
    times: np.ndarray,
    grid: SpectralGrid,
    params: RSWParameters,
    reference: RSWState | tuple[RSWState, RSWState],
) -> tuple[RSWState, RSWState, dict[str, float]]:
    filtered = state_from_filtered_samples(samples, times, filter_name, T)
    filtered_slow = slow_project_state(filtered, grid, params)
    instantaneous = RSWState.from_array(samples[-1])
    instantaneous_slow = slow_project_state(instantaneous, grid, params)
    if isinstance(reference, tuple):
        reference_full, reference_slow = reference
    else:
        reference_full = reference
        reference_slow = reference
    full_rms = float(np.sqrt(np.mean(filtered.as_array() ** 2)))
    slow_rms = float(np.sqrt(np.mean(filtered_slow.as_array() ** 2)))
    wave_rms = float(np.sqrt(np.mean((filtered.as_array() - filtered_slow.as_array()) ** 2)))
    row = {
        "full_state_error_vs_long_filtered": relative_state_error(filtered, reference_full),
        "full_state_error_vs_instant": relative_state_error(filtered, instantaneous),
        "state_error_vs_long_slow": relative_state_error(filtered_slow, reference_slow),
        "state_error_vs_P0_instant": relative_state_error(filtered_slow, instantaneous_slow),
        "filtered_full_rms": full_rms,
        "filtered_slow_rms": slow_rms,
        "filtered_wave_rms": wave_rms,
        "filtered_wave_fraction": wave_rms / max(full_rms, 1.0e-14),
    }
    return filtered, filtered_slow, row
