from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wave_averaging import andrews_textbook_two_wave as model


class TextbookAnchoredTwoWaveChecks(unittest.TestCase):
    def test_dispersion_frame_and_one_period_weight(self) -> None:
        p = model.P
        np.testing.assert_allclose(
            p.D,
            [5.133755102040816, 7.383755102040816],
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertAlmostEqual(p.beta, 51.33755102040816, places=13)
        np.testing.assert_allclose(
            p.omega,
            [0.0, 7.61807497982333],
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            p.sigma,
            [20.0, 17.38192502017667],
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertAlmostEqual(p.delta_sigma, -2.61807497982333, places=13)
        self.assertAlmostEqual(p.duration, math.pi / 10.0, places=14)
        self.assertAlmostEqual(p.difference_weight, 0.9720501515207912, places=14)

    def test_instantaneous_quadratic_closes_into_four_harmonic_groups(self) -> None:
        p = model.P
        rng = np.random.default_rng(20261027)
        q = rng.uniform(0.0, 2.0 * math.pi, 41)
        y = rng.uniform(0.05, math.pi - 0.05, 41)
        t = rng.uniform(-p.duration, p.duration, 41)
        direct = model.instantaneous_quadratic(q, y, p.z_ref, t, p)
        harmonic_sum = (
            model.exact_drift(q, y, p.z_ref, p)
            + model.difference_drift(q, y, p.z_ref, t, p)
            + model.sum_drift(q, y, p.z_ref, t, p)
            + model.double_drift(q, y, p.z_ref, t, p)
        )
        np.testing.assert_allclose(direct, harmonic_sum, rtol=0.0, atol=2.0e-16)

    def test_full_boxcar_matches_direct_centered_quadrature(self) -> None:
        p = model.P
        time = np.linspace(-p.duration / 2.0, p.duration / 2.0, 20001)
        q = np.full_like(time, 1.17)
        y = np.full_like(time, 1.22)
        history = model.instantaneous_quadratic(q, y, p.z_ref, time, p)
        numerical = np.trapezoid(history, time, axis=0) / p.duration
        analytic = model.full_boxcar(1.17, 1.22, p.z_ref, 0.0, p)
        np.testing.assert_allclose(numerical, analytic, rtol=0.0, atol=3.0e-11)

    def test_one_wave_limit_recovers_textbook_snapshot_and_zonal_drift(self) -> None:
        p = model.Parameters(a1=0.0)
        q = np.linspace(0.0, 2.0 * math.pi, 53)
        y = np.linspace(0.0, math.pi, 31)
        q_grid, y_grid = np.meshgrid(q, y, indexing="xy")
        expected_psi = (
            p.a0
            * math.exp(p.z_ref / (2.0 * p.H))
            * np.sin(p.ell * y_grid)
            * np.cos(p.k0 * q_grid + p.m * p.z_ref)
        )
        np.testing.assert_allclose(
            model.streamfunction(q_grid, y_grid, p.z_ref, 0.0, p),
            expected_psi,
            rtol=0.0,
            atol=3.0e-16,
        )
        expected_drift = (
            -p.ell**2
            * p.a0**2
            * math.exp(p.z_ref / p.H)
            * np.cos(2.0 * p.ell * y_grid)
            / (2.0 * p.Ubar)
        )
        np.testing.assert_allclose(
            model.exact_drift(q_grid, y_grid, p.z_ref, p)[..., 0],
            expected_drift,
            rtol=0.0,
            atol=2.0e-17,
        )


if __name__ == "__main__":
    unittest.main()
