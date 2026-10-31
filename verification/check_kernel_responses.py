"""Symbolic checks for averaging-kernel responses in §3.

Check kernel scaling and normalization, frequency and detuning responses,
and the Gaussian and causal-exponential formulas.

T is positive; frequencies and evaluation times are real. The change of
variable assumes integrable K; preservation of constants also requires
integral(K)=1. The Fourier convention is integral K(s) exp(-i omega s) ds.
The causal kernel samples past offsets s <= 0 in an average of f(t0+s).

Requires SymPy. Run: python verification/check_kernel_responses.py
"""

import unittest

import sympy as sp


class KernelResponseChecks(unittest.TestCase):
    def assert_zero(self, expression):
        residual = sp.simplify(sp.expand(expression.rewrite(sp.exp)))
        self.assertEqual(residual, 0, f"Nonzero or unresolved residual: {residual}")

    def test_general_kernel_scaling_and_normalization(self):
        """(3.1): the Jacobian cancels T^-1; normalization is inherited."""
        s, r = sp.symbols("s r", real=True)
        T = sp.Symbol("T", positive=True)
        kernel = sp.Function("K")
        original = sp.Integral(kernel(s / T) / T, (s, -sp.oo, sp.oo))
        transformed = original.transform(s, (T * r, r))
        self.assertEqual(transformed, sp.Integral(kernel(r), (r, -sp.oo, sp.oo)))

    def test_general_frequency_and_detuning_response(self):
        """(3.4) and (3.9), for integrable K."""
        s, r, t0, tau = sp.symbols("s r t0 tau", real=True)
        omega, wa, wb, wc = sp.symbols("omega omega_a omega_b omega_c", real=True)
        T = sp.Symbol("T", positive=True)
        kernel = sp.Function("K")
        for frequency, center in ((omega, t0), (wa + wb - wc, tau)):
            with self.subTest(frequency=frequency):
                original = sp.Integral(
                    kernel(s / T) * sp.exp(-sp.I * frequency * (center + s)) / T,
                    (s, -sp.oo, sp.oo),
                )
                transformed = original.transform(s, (T * r, r))
                # This remaining integral is K-hat(T*frequency) in the paper's
                # Fourier convention. The evaluation-time phase stays outside it.
                target_integrand = (
                    sp.exp(-sp.I * frequency * center)
                    * kernel(r) * sp.exp(-sp.I * T * frequency * r)
                )
                self.assertEqual(transformed.limits, ((r, -sp.oo, sp.oo),))
                self.assert_zero(transformed.function - target_integrand)

    def test_gaussian_response_by_integration(self):
        """Section 3: the normalized Gaussian has response exp(-(T*omega)^2/2)."""
        s, omega = sp.symbols("s omega", real=True)
        T = sp.Symbol("T", positive=True)
        kernel = sp.exp(-s**2 / (2 * T**2)) / (sp.sqrt(2 * sp.pi) * T)
        self.assert_zero(sp.integrate(kernel, (s, -sp.oo, sp.oo)) - 1)
        actual = sp.integrate(
            kernel * sp.exp(-sp.I * omega * s), (s, -sp.oo, sp.oo)
        )
        target = sp.exp(-(T * omega)**2 / 2)
        self.assert_zero(actual - target)
        self.assert_zero(actual.subs(omega, 0) - 1)
        self.assert_zero(actual * sp.conjugate(actual) - sp.exp(-(T * omega)**2))

    def test_causal_exponential_response_for_both_signs(self):
        """Section 3: past-offset response 1/(1-i*T*omega), including its phase."""
        s = sp.Symbol("s", real=True)
        T, magnitude = sp.symbols("T Omega", positive=True)
        kernel = sp.exp(s / T) / T  # Restricted to s <= 0 by the integral bounds.
        self.assert_zero(sp.integrate(kernel, (s, -sp.oo, 0)) - 1)
        for sign in (-1, 1):
            with self.subTest(frequency_sign=sign):
                omega = sign * magnitude
                actual = sp.integrate(
                    kernel * sp.exp(-sp.I * omega * s), (s, -sp.oo, 0)
                )
                target = 1 / (1 - sp.I * T * omega)
                self.assert_zero(actual - target)
                self.assert_zero(actual * sp.conjugate(actual)
                                 - 1 / (1 + (T * omega)**2))
                self.assert_zero(sp.limit(actual, magnitude, 0, dir="+") - 1)

    def test_response_magnitudes_and_decay(self):
        """Section 3: monotone Gaussian/exponential magnitudes and their endpoints."""
        q = sp.Symbol("q", positive=True)  # q = T*abs(omega) > 0.
        gaussian = sp.exp(-q**2 / 2)
        exponential = 1 / sp.sqrt(1 + q**2)
        self.assert_zero(sp.diff(gaussian, q) + q * sp.exp(-q**2 / 2))
        self.assert_zero(sp.diff(exponential, q) + q / (1 + q**2)**sp.Rational(3, 2))
        for response in (gaussian, exponential):
            self.assertIs(sp.diff(response, q).is_negative, True)
            self.assert_zero(sp.limit(response, q, 0, dir="+") - 1)
            self.assert_zero(sp.limit(response, q, sp.oo))


if __name__ == "__main__":
    unittest.main(verbosity=2)
