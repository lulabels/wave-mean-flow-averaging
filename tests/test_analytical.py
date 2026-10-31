from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wave_averaging.andrews_drift import (
    drift_uS_infinite,
    drift_uS_tophat,
    modes_for_streamfunction_cos_sin,
    sinc,
)


class AnalyticalCheckpoints(unittest.TestCase):
    def test_figure4_boxcar_weights(self) -> None:
        T = 10.0
        detunings = np.array([0.0, 3.0 + 4.75 - 8.0, 3.0 + 2.4 - 8.0])
        actual = np.abs(sinc(detunings * T / 2.0))
        np.testing.assert_allclose(actual, [1.0, 0.759187695484469, 0.03232054129435687], rtol=1e-13)

    def test_boxcar_weight_at_pi_over_six(self) -> None:
        self.assertAlmostEqual(sinc(np.pi / 6.0), 3.0 / np.pi, places=14)
        self.assertAlmostEqual(sinc(np.pi / 6.0), 0.954929658551372, places=14)

    def test_common_period_recovers_exact_drift(self) -> None:
        x = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        y = np.linspace(0.0, np.pi, 16, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        modes = (
            modes_for_streamfunction_cos_sin(kx=12.0, l=1.0, omega=12.0, psi0=1.0)
            + modes_for_streamfunction_cos_sin(kx=13.0, l=1.0, omega=13.0, psi0=1.0)
        )
        exact = drift_uS_infinite(X, Y, modes)
        finite = drift_uS_tophat(
            X,
            Y,
            modes,
            T=12.0 * 2.0 * np.pi / 12.0,
            include_sum_terms=False,
            use_gauge_constants=False,
        )
        self.assertLess(float(np.max(np.abs(finite - exact))), 2.0e-14)


if __name__ == "__main__":
    unittest.main()
