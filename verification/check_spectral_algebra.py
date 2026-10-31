"""Symbolic checks for pressure projection and modal averaging in Appendix A.

Check pressure recovery, Fourier projection, Fourier-symbol skewness on
divergence-free velocities, and the one-sided Cesaro response and modal limit.

Fourier checks use real, nonzero wavevectors (k, l, m), with integer components
on the 2*pi torus; individual components may vanish. The zero mode is treated
separately: the inverse Laplacian annihilates it and the Leray projection
preserves constant velocities. F and modal frequencies are real, with T > 0.

Requires SymPy. Run: python verification/check_spectral_algebra.py
"""

import unittest

import sympy as sp


X, Y, Z = sp.symbols("x y z", real=True)
COORDINATES = (X, Y, Z)
K, L, M, F = sp.symbols("k l m F", real=True)
WAVEVECTOR = sp.Matrix([K, L, M])
SQUARED_NORM = WAVEVECTOR.dot(WAVEVECTOR)
VERTICAL = sp.Matrix([0, 0, 1])


def gradient(scalar):
    return sp.Matrix([sp.diff(scalar, coordinate) for coordinate in COORDINATES])


def divergence(vector):
    return sum(sp.diff(vector[j], COORDINATES[j]) for j in range(3))


def curl(vector):
    return sp.Matrix([sp.diff(vector[2], Y) - sp.diff(vector[1], Z),
                      sp.diff(vector[0], Z) - sp.diff(vector[2], X),
                      sp.diff(vector[1], X) - sp.diff(vector[0], Y)])


class SpectralAlgebraChecks(unittest.TestCase):
    def assert_zero(self, residual, context):
        entries = list(residual) if isinstance(residual, sp.MatrixBase) else [residual]
        for index, entry in enumerate(entries):
            reduced = sp.simplify(sp.expand(entry.rewrite(sp.exp)))
            self.assertEqual(reduced, 0, f"{context}; entry {index}: {reduced}")

    def test_differential_divergence_and_curl(self):
        """(A4a), valid before imposing incompressibility."""
        u, v, w, density = [sp.Function(name)(*COORDINATES)
                             for name in ("u", "v", "w", "rho")]
        velocity = sp.Matrix([u, v, w])
        forcing = F * VERTICAL.cross(velocity) + density * VERTICAL
        target = -F * VERTICAL.dot(curl(velocity)) + sp.diff(density, Z)
        self.assert_zero(divergence(forcing) - target, "Coriolis/density divergence")
        pressure = sp.Function("p")(*COORDINATES)
        self.assert_zero(curl(gradient(pressure)), "pressure gradient has zero curl")

    def test_fourier_pressure_recovery(self):
        """(A2) and (A4b) recover (2.4a) for a nonzero wavevector."""
        u, v, w, density = sp.symbols("u_hat v_hat w_hat rho_hat")
        amplitude = sp.Matrix([u, v, w])
        phase = sp.exp(sp.I * WAVEVECTOR.dot(sp.Matrix(COORDINATES)))
        velocity, rho = amplitude * phase, density * phase
        forcing = F * VERTICAL.cross(velocity) + rho * VERTICAL
        # Delta has multiplier -|k|^2. This inverse is used only for |k|^2 > 0.
        inverse_divergence = -divergence(forcing) / SQUARED_NORM
        self.assert_zero(divergence(gradient(inverse_divergence)) - divergence(forcing),
                         "inverse Laplacian from differentiated plane wave")
        projected = forcing - gradient(inverse_divergence)
        leray = sp.eye(3) - WAVEVECTOR * WAVEVECTOR.T / SQUARED_NORM
        target = leray * (F * VERTICAL.cross(amplitude) + density * VERTICAL) * phase
        self.assert_zero(projected - target, "independent Fourier projection target")
        self.assert_zero(divergence(projected), "pressure removes divergence")
        source = F * VERTICAL.dot(curl(velocity)) - sp.diff(rho, Z)
        qg_velocity = forcing + gradient(-source / SQUARED_NORM)
        self.assert_zero(qg_velocity.col_join(sp.Matrix([-velocity[2]]))
                         - target.col_join(sp.Matrix([-w * phase])),
                         "full QG fast operator, including density sign")

    def test_fourier_projection_and_restricted_skewness(self):
        """Finite Fourier-symbol algebra for (A2) and (A6)."""
        leray = sp.eye(3) - WAVEVECTOR * WAVEVECTOR.T / SQUARED_NORM
        self.assert_zero(leray**2 - leray, "Leray idempotence")
        self.assert_zero(leray.H - leray, "Leray Hermitian symmetry")
        self.assert_zero(WAVEVECTOR.T * leray, "Leray range is divergence free")
        self.assert_zero(leray * WAVEVECTOR, "gradient amplitudes are removed")
        state_projection = sp.diag(leray, 1)
        unprojected = sp.Matrix([[0, -F, 0, 0], [F, 0, 0, 0],
                                [0, 0, 0, 1], [0, 0, -1, 0]])
        fast = state_projection * unprojected
        # Both test states lie in ran(state_projection); fast need not be
        # skew-Hermitian on the unrestricted four-component ambient space.
        self.assert_zero(state_projection * (fast.H + fast) * state_projection,
                         "skewness on the divergence-free state subspace")

    def test_zero_mode_convention(self):
        """At k=0 the Leray symbol is I, not the singular nonzero-mode formula."""
        u, v, w, density = sp.symbols("u_0 v_0 w_0 rho_0")
        velocity = sp.Matrix([u, v, w])
        forcing = F * VERTICAL.cross(velocity) + density * VERTICAL
        self.assert_zero(divergence(forcing), "constant forcing has zero divergence")
        self.assert_zero(F * VERTICAL.dot(curl(velocity)) - sp.diff(density, Z),
                         "zero-mode pressure source")
        # The zero-mean periodic convention sets Delta^{-1}(0) = 0 here.
        projected = forcing - gradient(sp.Integer(0))
        self.assert_zero(projected - forcing, "constant vectors survive Leray projection")
        actual = projected.col_join(sp.Matrix([-w]))
        target = sp.Matrix([-F * v, F * u, density, -w])
        self.assert_zero(actual - target, "constant-state fast operator")
        self.assert_zero(actual.jacobian((u, v, w, density)).H
                         + actual.jacobian((u, v, w, density)),
                         "zero-mode skewness")

    def test_one_sided_cesaro_integral(self):
        """(A9) and (A11); signed real omega != 0."""
        frequency = sp.Symbol("omega", real=True, nonzero=True)
        window = sp.Symbol("T", positive=True)
        time = sp.Symbol("s", real=True)
        actual = sp.integrate(sp.exp(-sp.I * time * frequency), (time, 0, window)) / window
        target = (1 - sp.exp(-sp.I * window * frequency)) / (sp.I * window * frequency)
        self.assert_zero(actual - target, "one-sided integral and exponent sign")
        centered = sp.sin(window * frequency / 2) / (window * frequency / 2)
        self.assert_zero(actual - sp.exp(-sp.I * window * frequency / 2) * centered,
                         "one-sided interval retains its midpoint phase")
        self.assert_zero(actual.subs(frequency, -frequency) - sp.conjugate(actual),
                         "opposite signed frequencies are conjugate")
        real_part, imaginary_part = sp.expand_complex(actual).as_real_imag()
        self.assert_zero(real_part - sp.sin(window * frequency) / (window * frequency),
                         "general real part of the one-sided response")
        self.assert_zero(imaginary_part - (sp.cos(window * frequency) - 1)
                         / (window * frequency),
                         "general imaginary part of the one-sided response")
        # The nonzero local coefficient proves the response is not generally
        # real, for every T > 0, without selecting a particular frequency/window.
        # It does not exclude isolated real responses, e.g. complete periods.
        phase_coefficient = sp.limit(imaginary_part / frequency, frequency, 0)
        self.assert_zero(phase_coefficient + window / 2,
                         "one-sided phase near zero frequency")
        self.assertIs(phase_coefficient.is_negative, True)

    def test_cesaro_zero_and_nonzero_limits(self):
        """Scalar content of (A12)–(A13) only."""
        magnitude, window = sp.symbols("a T", positive=True)
        time, frequency = sp.symbols("s omega", real=True)
        zero_response = sp.integrate(sp.Integer(1), (time, 0, window)) / window
        self.assertEqual(zero_response, 1)
        self.assertEqual(sp.limit(zero_response, window, sp.oo), 1)
        response = (1 - sp.exp(-sp.I * window * frequency)) / (sp.I * window * frequency)
        self.assertEqual(sp.limit(response, frequency, 0), 1)
        for sign in (-1, 1):
            # Separate signs avoid assuming that a generic nonzero real omega is positive.
            actual = sp.integrate(sp.exp(-sp.I * sign * magnitude * time),
                                   (time, 0, window)) / window
            real_part, imaginary_part = sp.expand_complex(actual).as_real_imag()
            self.assert_zero(real_part - sp.sin(magnitude * window) / (magnitude * window),
                             "real part from integration")
            self.assert_zero(imaginary_part - sign * (sp.cos(magnitude * window) - 1)
                             / (magnitude * window), "imaginary part from integration")
            self.assertEqual(sp.limit(real_part, window, sp.oo), 0)
            self.assertEqual(sp.limit(imaginary_part, window, sp.oo), 0)
            self.assert_zero(actual * sp.conjugate(actual)
                             - 4 * sp.sin(magnitude * window / 2)**2 / (magnitude * window)**2,
                             "exact squared magnitude")
        # With |sin| <= 1, the preceding identity gives the scalar envelope 2/(a T).
        self.assertEqual(sp.limit(2 / (magnitude * window), window, sp.oo), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
