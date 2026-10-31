"""Symbolic checks for the wave reference frame in §§2–3 and Appendix C.

Check cancellation of the fast term, quadratic interaction phases and the
centered-boxcar detuning weight at the current fast time.

The fast generator is time independent with a complete finite eigenbasis and
signed real frequencies, including zero. Slow, diffusive and bilinear
coefficients are arbitrary; commutation with the fast generator is not assumed.
Epsilon and T are positive, and amplitudes are held fixed during fast-phase
integration.

Requires SymPy. Run: python verification/check_wave_frame.py
"""

import unittest

import sympy as sp


class WaveFrameChecks(unittest.TestCase):
    def assert_zero(self, expression):
        entries = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
        for entry in entries:
            residual = sp.simplify(entry.rewrite(sp.exp))
            self.assertEqual(residual, 0, msg=f"Nonzero residual: {residual}")

    def test_fast_term_cancels_after_differentiation(self):
        """(2.3), (2.9), (C1): arbitrary finite modal system and bilinear tensor."""
        t = sp.symbols("t", real=True)
        epsilon = sp.symbols("epsilon", positive=True)
        count = sp.Symbol("M", integer=True, positive=True)
        output, left, right = sp.symbols("c a b", integer=True)
        omega = sp.IndexedBase("omega", real=True)
        slow, diffusion, bilinear = (sp.IndexedBase(name) for name in ("S", "D", "B"))
        w = sp.Function("w")
        amplitude = sp.IndexedBase("v")
        linear_term = sp.Sum(slow[output, left] * w(left, t), (left, 0, count - 1))
        diffusive_term = sp.Sum(diffusion[output, left] * w(left, t), (left, 0, count - 1))
        quadratic_term = sp.Sum(
            bilinear[output, left, right] * w(left, t) * w(right, t),
            (left, 0, count - 1), (right, 0, count - 1),
        )
        forward = sp.exp(sp.I * omega[output] * t / epsilon)
        original_rhs = -sp.I * omega[output] * w(output, t) / epsilon - linear_term - quadratic_term + diffusive_term
        actual = sp.diff(forward * w(output, t), t).subs(sp.diff(w(output, t), t), original_rhs)
        # Replace each physical component by its inverse-mapped amplitude only
        # after differentiation and substitution of the physical dynamics.
        actual = actual.replace(
            lambda expression: expression.is_Function and expression.func == w,
            lambda expression: sp.exp(-sp.I * omega[expression.args[0]] * expression.args[1] / epsilon) * amplitude[expression.args[0]],
        )
        linear_phase = sp.exp(sp.I * (omega[output] - omega[left]) * t / epsilon)
        quadratic_phase = sp.exp(-sp.I * (omega[left] + omega[right] - omega[output]) * t / epsilon)
        expected = -sp.Sum(slow[output, left] * amplitude[left] * linear_phase, (left, 0, count - 1)) - sp.Sum(
            bilinear[output, left, right] * amplitude[left] * amplitude[right] * quadratic_phase,
            (left, 0, count - 1), (right, 0, count - 1),
        ) + sp.Sum(diffusion[output, left] * amplitude[left] * linear_phase, (left, 0, count - 1))
        self.assert_zero(forward * sp.exp(-sp.I * omega[output] * t / epsilon) - 1)
        self.assert_zero(actual - expected)

    def test_quadratic_phase_from_input_and_output_modes(self):
        """(C5): two input phases and one output phase."""
        tau, s = sp.symbols("tau s", real=True)
        omega_a, omega_b, omega_c = sp.symbols("omega_a omega_b omega_c", real=True)
        amplitude_a, amplitude_b, coefficient = sp.symbols("A B coefficient")
        phase = tau + s
        # L_F r_j = i omega_j r_j. B(r_a, r_b) has coefficient in r_c.
        input_a = amplitude_a * sp.exp(-sp.I * omega_a * phase)
        input_b = amplitude_b * sp.exp(-sp.I * omega_b * phase)
        output_map = sp.exp(sp.I * omega_c * phase)
        actual = output_map * coefficient * input_a * input_b
        detuning = omega_a + omega_b - omega_c
        expected = coefficient * amplitude_a * amplitude_b * sp.exp(-sp.I * detuning * phase)
        self.assert_zero(actual - expected)
        self.assert_zero(
            actual.subs(omega_c, omega_a + omega_b) - coefficient * amplitude_a * amplitude_b
        )

    def test_centered_boxcar_detuning_weight(self):
        """(3.10) and (C6); T > 0."""
        tau, s = sp.symbols("tau s", real=True)
        T = sp.symbols("T", positive=True)
        detuning = sp.symbols("detuning", real=True, nonzero=True)
        signal = sp.exp(-sp.I * detuning * (tau + s))
        actual = sp.integrate(signal, (s, -T / 2, T / 2)) / T
        sinc_weight = sp.sin(T * detuning / 2) / (T * detuning / 2)
        expected = sp.exp(-sp.I * tau * detuning) * sinc_weight
        self.assert_zero(actual - expected)
        # Exact resonance is integrated separately, without dividing by detuning.
        resonant = sp.integrate(signal.subs(detuning, 0), (s, -T / 2, T / 2)) / T
        self.assert_zero(resonant - 1)
        self.assert_zero(sp.limit(expected, detuning, 0) - 1)

    def test_current_phase_differs_from_zero_centered_window(self):
        """(C2) uses tau+s, retaining the current phase as in (3.11)."""
        tau, s = sp.symbols("tau s", real=True)
        T = sp.symbols("T", positive=True)
        detuning = sp.symbols("detuning", real=True, nonzero=True)
        current = sp.integrate(sp.exp(-sp.I * detuning * (tau + s)), (s, -T / 2, T / 2)) / T
        at_zero = sp.integrate(sp.exp(-sp.I * detuning * s), (s, -T / 2, T / 2)) / T
        expected_difference = (sp.exp(-sp.I * tau * detuning) - 1) * (
            sp.sin(T * detuning / 2) / (T * detuning / 2)
        )
        self.assert_zero(current - at_zero - expected_difference)
        self.assert_zero(current.subs(tau, 0) - at_zero)
        self.assert_zero(sp.diff(current, tau) + sp.I * detuning * current)
        self.assert_zero(sp.diff(at_zero, tau))
        self.assert_zero(current.subs(detuning, -detuning) - sp.conjugate(current))
        # For every nonzero signed detuning, the phase dependence is not the
        # zero function of (tau,T). This allows special equalities at sinc zeros
        # or phase multiples, unlike a claim of pointwise inequality everywhere.
        phase_slope = sp.limit(sp.diff(current - at_zero, tau).subs(tau, 0), T, 0, dir="+")
        self.assert_zero(phase_slope + sp.I * detuning)
        self.assertIs(phase_slope.is_zero, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
