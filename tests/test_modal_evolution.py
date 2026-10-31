"""Small checks of the optional averaged-equation API, separate from figure data."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wave_averaging.rsw.modal_backend_compact import CompactModalSystem
from wave_averaging.rsw.modal_reduced import ModalSystem, direct_finite_window_rhs, finite_window_multiplier
from wave_averaging.rsw.solver_pseudospectral import RSWParameters, SpectralGrid


class OptionalModalEvolution(unittest.TestCase):
    def test_nonzero_time_rhs_matches_phase_quadrature(self):
        reference = ModalSystem(SpectralGrid(4), RSWParameters())
        compact = CompactModalSystem(SpectralGrid(4), RSWParameters())
        a = reference.conjugate_symmetric_amplitudes(np.random.default_rng(820), scale=0.03)
        kwargs = dict(time=0.37, epsilon=0.3, window_name="hann", window_length=0.8,
                      window_origin="centered", nquad=129)
        expected = direct_finite_window_rhs(reference, a, **kwargs)
        for system in (reference, compact):
            np.testing.assert_allclose(system.finite_window_rhs(a, **kwargs), expected,
                                       atol=2e-14, rtol=2e-12)

    def test_causal_exponential_uses_past_offsets(self):
        actual = finite_window_multiplier("causal_exponential", 1.3, epsilon=0.4,
                                          window_length=0.2, time_unit="physical", window_origin="trailing")
        self.assertLess(abs(actual - 1 / (1 - 1j * 1.3 * 0.2 / 0.4)), 1e-15)

    def test_rk4_evaluates_each_stage_at_its_physical_time(self):
        for backend in (ModalSystem, CompactModalSystem):
            system = backend(SpectralGrid(4), RSWParameters())
            seen = []

            def rhs(a, *, time, **kwargs):
                seen.append(time)
                return (1 + time) * a

            with patch.object(system, "finite_window_rhs", rhs):
                system.rk4_step(np.ones(1), 0.1, "finite", time=0.37)
            np.testing.assert_allclose(seen, [0.37, 0.42, 0.42, 0.47], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
