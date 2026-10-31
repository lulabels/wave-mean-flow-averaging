"""Independent invariants for the portable dense field-history workflow."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generate_data"))
import run_refined_fig78 as refined
from fast_fig78_solver import FastRSWSolver
from wave_averaging.rsw.initial_conditions import make_initial_state
from wave_averaging.rsw.solver_pseudospectral import (
    PseudospectralRSWSolver, RSWParameters, RSWState, dealias_state,
)


class RefinedHistoryInvariants(unittest.TestCase):
    def test_grid_comparison_counts_unresolved_scales(self):
        x = np.arange(8) * 2 * np.pi / 8
        xx = np.arange(16) * 2 * np.pi / 16
        low = np.zeros((3, 8, 8))
        high = np.zeros((3, 16, 16))
        low[0] = np.sin(x)[:, None]
        high[0] = (np.sin(xx) + 0.3 * np.sin(5 * xx))[:, None]
        result = refined.study.field_error(low, high)
        self.assertLess(result["relative_rms_restricted"], 2e-15)
        self.assertAlmostEqual(result["relative_rms_including_unresolved_modes"],
                               0.3 / np.sqrt(1 + 0.3**2), places=14)

    def test_centered_weights_preserve_constant_and_linear_histories(self):
        times = np.arange(161) * 0.00625
        for kernel in ("boxcar", "bump"):
            weights = refined.window_weights(times, 0.5, 0.5, kernel, "trapezoid")
            np.testing.assert_allclose(weights.sum(), 1, atol=2e-15, rtol=0)
            np.testing.assert_allclose(weights @ times, 0.5, atol=2e-15, rtol=0)
            np.testing.assert_allclose(weights, weights[::-1], atol=2e-15, rtol=0)

    def test_aligned_endpoints_are_required(self):
        with self.assertRaises(ValueError):
            refined.window_weights(np.arange(11) * 0.1, 0.5, 0.5, "boxcar", "trapezoid")

    def test_fast_rk4_matches_canonical(self):
        grid = refined.CachedGrid(16)
        params = RSWParameters(nu=0)
        initial, _ = make_initial_state(refined.INITIAL, grid, params)
        a = dealias_state(initial, grid)
        b = a.copy()
        canonical = PseudospectralRSWSolver(grid, params)
        fast = FastRSWSolver(grid, params, workers=1)
        for _ in range(12):
            a = canonical.step_rk4(a, 0.00078125)
            b = fast.step_rk4(b, 0.00078125)
        np.testing.assert_allclose(a.as_array(), b.as_array(), atol=2e-15, rtol=0)
        with self.assertRaises(TypeError):
            fast.rhs(RSWState.from_array(b.as_array().astype(np.float32)))

    def test_wave_frame_preserves_a_freely_propagating_wave(self):
        grid = refined.CachedGrid(8)
        params = RSWParameters(nu=0)
        system = refined.study.BatchedModalMap(grid, params)
        initial, _ = make_initial_state(refined.INITIAL, grid, params)
        initial = dealias_state(initial, grid)
        amplitudes = system.state_to_modal(initial)
        times = np.arange(81) * 0.0125
        samples = np.asarray([system.modal_to_state(amplitudes * np.exp(-1j * system.omega * t)).as_array()
                              for t in times])
        results = refined.average_arrays(times, samples, grid, params,
                                         [(0.5, 0.5, k) for k in ("boxcar", "bump")], "trapezoid")
        target = system.modal_to_state(amplitudes * np.exp(-1j * system.omega * 0.5)).as_array()
        for _, wave in results:
            np.testing.assert_allclose(wave.as_array(), target, atol=2e-15, rtol=0)


if __name__ == "__main__":
    unittest.main()
