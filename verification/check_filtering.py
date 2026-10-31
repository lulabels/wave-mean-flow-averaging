"""Symbolic checks for finite-window averaging in §3 and Appendix B.

Check boxcar responses, repeated averaging, linearity, the common-period
case, the modal exact-average limit and the material-trajectory chain rule.

Window length T is positive; evaluation times and signed frequencies are real.
Zero frequency is treated separately. Modal checks apply componentwise and
to finite sums. The chain rule assumes smooth scalar fields and trajectories.

Requires SymPy. Run: python verification/check_filtering.py
"""

import unittest

import sympy as sp


def boxcar(expression, variable, center, width):
    """Integrate the expression over a centered window, as in (B1)."""
    return sp.integrate(
        expression, (variable, center - width / 2, center + width / 2)
    ) / width


class FilteringChecks(unittest.TestCase):
    def assert_zero(self, expression):
        # Exponential expansion avoids costly multivariate trigonometric factoring.
        residual = sp.simplify(sp.expand(expression.rewrite(sp.exp)))
        self.assertEqual(residual, 0, f"Nonzero symbolic residual: {residual}")

    def test_boxcar_response_for_signed_frequency(self):
        """(3.5), (B2): derive the response for either sign of omega."""
        t, t0, s = sp.symbols("t t0 s", real=True)
        width = sp.Symbol("T", positive=True)
        omega = sp.Symbol("omega", real=True, nonzero=True)
        expected = sp.exp(-sp.I * omega * t0) * (
            sp.sin(omega * width / 2) / (omega * width / 2)
        )
        absolute_time = boxcar(sp.exp(-sp.I * omega * t), t, t0, width)
        offset_time = boxcar(sp.exp(-sp.I * omega * (t0 + s)), s, 0, width)
        self.assert_zero(absolute_time - expected)
        self.assert_zero(offset_time - expected)

    def test_zero_frequency_and_removable_limit(self):
        """(B2) at omega=0: constants are preserved, including the removable limit."""
        t, t0, omega = sp.symbols("t t0 omega", real=True)
        width = sp.Symbol("T", positive=True)
        constant = sp.Symbol("c")
        once = boxcar(constant, t, t0, width)
        twice = boxcar(once, t0, t, width)
        self.assert_zero(once - constant)
        self.assert_zero(twice - constant)
        response = sp.exp(-sp.I * omega * t0) * (
            sp.sin(omega * width / 2) / (omega * width / 2)
        )
        self.assert_zero(sp.limit(response, omega, 0, dir="+") - 1)
        self.assert_zero(sp.limit(response, omega, 0, dir="-") - 1)

    def test_repeated_average_and_general_nonprojection(self):
        """Section 3, after (3.5): two integrations and a nonzero modal defect."""
        t, s, t0 = sp.symbols("t s t0", real=True)
        width = sp.Symbol("T", positive=True)
        omega = sp.Symbol("omega", real=True, nonzero=True)
        once = boxcar(sp.exp(-sp.I * omega * s), s, t, width)
        twice = boxcar(once, t, t0, width)
        multiplier = sp.sin(omega * width / 2) / (omega * width / 2)
        self.assert_zero(twice - multiplier**2 * sp.exp(-sp.I * omega * t0))
        defect = twice - once.subs(t, t0)
        self.assert_zero(defect - multiplier * (multiplier - 1)
                         * sp.exp(-sp.I * omega * t0))
        # For every T > 0, this nonzero coefficient rules out idempotence on
        # arbitrary real-frequency signals. Zero/full-period modes are separate
        # exceptions; this does not assert that every modal defect is nonzero.
        coefficient = sp.limit(defect / omega**2, omega, 0)
        self.assert_zero(coefficient + width**2 / 24)
        self.assertIs(coefficient.is_negative, True)

    def test_linearity_for_arbitrary_integrable_functions(self):
        """(3.7): integral linearity extends componentwise identities to finite sums."""
        t, t0 = sp.symbols("t t0", real=True)
        width = sp.Symbol("T", positive=True)
        a, b = sp.symbols("a b")
        f, g = sp.Function("f"), sp.Function("g")
        # Existence of these integrals is assumed; finite Fourier sums satisfy it.
        combined = boxcar(a * f(t) + b * g(t), t, t0, width)
        separate = a * boxcar(f(t), t, t0, width) + b * boxcar(g(t), t, t0, width)
        self.assert_zero(combined - separate)

    def test_common_period_projection_exception(self):
        """§3 qualification: a full-period boxcar projects a finite harmonic sum."""
        t, t0 = sp.symbols("t t0", real=True)
        fundamental = sp.Symbol("Omega", positive=True)
        harmonic = sp.Symbol("n", integer=True, nonzero=True)
        period = 2 * sp.pi / fundamental
        # Every signed, nonzero integer harmonic is removed by direct integration.
        self.assert_zero(
            boxcar(sp.exp(-sp.I * harmonic * fundamental * t), t, t0, period)
        )
        c0, coefficient = sp.symbols("c0 c_n")
        signal = c0 + coefficient * sp.exp(-sp.I * harmonic * fundamental * t)
        once = boxcar(signal, t, t0, period)
        twice = boxcar(once, t0, t, period)
        self.assert_zero(once - c0)
        self.assert_zero(twice - c0)

    def test_general_fourier_component_and_exact_selector(self):
        """(3.7)–(3.8): arbitrary coefficients and signed frequencies, mode by mode."""
        t, t0 = sp.symbols("t t0", real=True)
        width = sp.Symbol("T", positive=True)
        magnitude = sp.Symbol("Omega", positive=True)
        c0, coefficient = sp.symbols("c0 c")
        for sign in (-1, 1):
            with self.subTest(frequency_sign=sign):
                omega = sign * magnitude
                signal = c0 + coefficient * sp.exp(-sp.I * omega * t)
                averaged = boxcar(signal, t, t0, width)
                expected = c0 + coefficient * sp.exp(-sp.I * omega * t0) * (
                    sp.sin(omega * width / 2) / (omega * width / 2)
                )
                self.assert_zero(averaged - expected)
                # The verified sine form exposes the bounded oscillations.
                self.assert_zero(sp.limit(expected, width, sp.oo) - c0)
        # Integral linearity and the preceding arbitrary-mode calculation give
        # (3.7)–(3.8) for every finite sum, with no exchange of an infinite sum.
        # Applying the exact selector to its constant result leaves it unchanged.
        self.assert_zero(sp.limit(boxcar(c0, t, t0, width), width, sp.oo) - c0)

    def test_material_trajectory_chain_rule(self):
        """(B4), assuming dX/ds = u(X,t+s)."""
        s, t = sp.symbols("s t", real=True)
        x, y, z, q = sp.symbols("x y z q", real=True)
        coordinates = (x, y, z)
        trajectory = [sp.Function(name)(s) for name in ("X", "Y", "Z")]
        velocity = [sp.Function(name)(x, y, z, q) for name in ("u", "v", "w")]
        theta = sp.Function("theta")(x, y, z, q)
        pullback = dict(zip((*coordinates, q), (*trajectory, t + s)))
        direct = sp.diff(theta.subs(pullback, simultaneous=True), s)
        trajectory_ode = {
            sp.diff(position, s): component.subs(pullback, simultaneous=True)
            for position, component in zip(trajectory, velocity)
        }
        direct = direct.subs(trajectory_ode, simultaneous=True)
        material_derivative = sp.diff(theta, q) + sum(
            component * sp.diff(theta, coordinate)
            for component, coordinate in zip(velocity, coordinates)
        )
        expected = material_derivative.subs(pullback, simultaneous=True)
        self.assert_zero(direct - expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
