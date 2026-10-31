"""Symbolic checks for the near-inertial and surface-wave examples in §§1 and 4.

Check near-inertial averages, surface-wave displacement and Stokes drift,
and the first-order Taylor coefficient of displaced velocity.

The near-inertial averages hold U and A fixed over one period. Surface-wave
checks use the paper's linear deep-water fields with real constant a and
positive k and omega (physically omega**2 = g*k, z <= 0). Taylor checks assume
differentiable fields and displacement.

Requires SymPy. Run: python verification/check_classical_examples.py
"""

import unittest

import sympy as sp


class ClassicalExampleChecks(unittest.TestCase):
    def assert_zero(self, expression):
        residual = sp.simplify(expression)
        self.assertEqual(residual, 0, f"Nonzero symbolic residual: {residual}")

    def setUp(self):
        self.x, self.z, self.t, self.t0, self.a = sp.symbols("x z t t0 a", real=True)
        self.k, self.omega = sp.symbols("k omega", positive=True)
        phase = self.k * self.x - self.omega * self.t
        envelope = self.a * sp.exp(self.k * self.z)
        # (4.1) and (4.2).
        self.velocity = self.omega * envelope * sp.Matrix([sp.cos(phase), sp.sin(phase)])
        self.displacement = envelope * sp.Matrix([-sp.sin(phase), sp.cos(phase)])

    def period_mean(self, expression):
        period = 2 * sp.pi / self.omega
        return sp.integrate(expression, (self.t, self.t0, self.t0 + period)) / period

    def test_frozen_near_inertial_averages_for_both_frequency_signs(self):
        """(1.2)–(1.4): direct and mapped averages with frozen slow amplitudes."""
        s, t0 = sp.symbols("s t0", real=True)
        magnitude = sp.Symbol("f_abs", positive=True)
        background, amplitude = sp.symbols("U A")  # arbitrary complex constants
        period = 2 * sp.pi / magnitude
        for sign in (-1, 1):
            with self.subTest(frequency_sign=sign):
                frequency = sign * magnitude
                velocity = background + amplitude * sp.exp(-sp.I * frequency * s)
                mapped = sp.exp(sp.I * frequency * s) * velocity
                expected_mapped = amplitude + background * sp.exp(sp.I * frequency * s)
                self.assert_zero(mapped - expected_mapped)
                self.assert_zero(sp.exp(-sp.I * frequency * s) * mapped - velocity)
                direct_mean = sp.integrate(velocity, (s, t0, t0 + period)) / period
                mapped_mean = sp.integrate(sp.expand(mapped), (s, t0, t0 + period)) / period
                self.assert_zero(direct_mean - background)
                self.assert_zero(mapped_mean - amplitude)

    def test_deep_water_displacement_and_zero_period_means(self):
        """(4.1)–(4.3): integrate and differentiate at fixed x,z."""
        for velocity, displacement in zip(self.velocity, self.displacement):
            with self.subTest(component=str(velocity)):
                self.assert_zero(sp.diff(displacement, self.t) - velocity)
                primitive = sp.integrate(velocity, self.t)
                # The zero-period-mean condition fixes the integration constant.
                zero_mean_primitive = primitive - self.period_mean(primitive)
                self.assert_zero(zero_mean_primitive - displacement)
                self.assert_zero(self.period_mean(displacement))
                self.assert_zero(self.period_mean(velocity))

    def test_displaced_velocity_first_taylor_coefficient(self):
        """(4.4): d/dh of u(x+h*xi,z+h*zeta,t) at h=0."""
        h = sp.Symbol("h", real=True)
        xi, zeta = self.displacement
        u = self.velocity[0]
        shifted = u.subs({self.x: self.x + h * xi, self.z: self.z + h * zeta},
                         simultaneous=True)
        coefficient = sp.diff(shifted, h).subs(h, 0)
        directional_derivative = xi * sp.diff(u, self.x) + zeta * sp.diff(u, self.z)
        self.assert_zero(shifted.subs(h, 0) - u)
        self.assert_zero(coefficient - directional_derivative)
        self.assert_zero(coefficient - self.k * self.a**2 * self.omega
                         * sp.exp(2 * self.k * self.z))
        # Independent amplitude expansion: displacement and velocity both scale as a.
        quadratic_coefficient = sp.diff(shifted.subs(h, 1), self.a, 2).subs(self.a, 0) / 2
        self.assert_zero(quadratic_coefficient - self.k * self.omega
                         * sp.exp(2 * self.k * self.z))

    def test_quadratic_stokes_drift_period_average(self):
        """(4.5)–(4.6): average each spatial-gradient pairing."""
        xi, zeta = self.displacement
        u = self.velocity[0]
        horizontal_pair = xi * sp.diff(u, self.x)
        vertical_pair = zeta * sp.diff(u, self.z)
        expected = self.k * self.a**2 * self.omega * sp.exp(2 * self.k * self.z)
        self.assert_zero(self.period_mean(horizontal_pair) - expected / 2)
        self.assert_zero(self.period_mean(vertical_pair) - expected / 2)
        eulerian_mean = self.period_mean(u)
        first_order_lagrangian_mean = self.period_mean(u + horizontal_pair + vertical_pair)
        self.assert_zero(first_order_lagrangian_mean - eulerian_mean - expected)
        self.assert_zero(sp.diff(horizontal_pair + vertical_pair, self.t))

    def test_generic_vector_first_order_taylor_ingredient(self):
        """(4.16): differentiate a shifted slow+fast field."""
        x, y, z, tau, t, h = sp.symbols("x y z tau t h", real=True)
        coordinates = (x, y, z)
        slow = sp.Matrix([sp.Function(name)(x, y, z, t) for name in ("U", "V", "W")])
        fast = sp.Matrix([sp.Function(name)(x, y, z, tau, t) for name in ("u", "v", "w")])
        xi = sp.Matrix([sp.Function(name)(x, y, z, tau, t)
                        for name in ("xi", "eta", "zeta")])
        # xi is evaluated at the base point and is independent of h.
        shifted = (slow + fast).subs(
            {q: q + h * displacement for q, displacement in zip(coordinates, xi)},
            simultaneous=True,
        )
        coefficient = shifted.diff(h).subs(h, 0)
        expected = slow.jacobian(coordinates) * xi + fast.jacobian(coordinates) * xi
        for residual in coefficient - expected:
            self.assert_zero(residual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
