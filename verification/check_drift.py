"""Symbolic checks for wave displacement and Stokes drift in §4.

Check hydrostatic wave projections, displacement, and exact and finite-window
drift. Complex seeds generate the signed wave eigenspaces; pairwise checks
use arbitrary reconstructed velocity amplitudes.

The hydrostatic state is (u, v, rho), with real wavevectors and rotation and
m != 0. Wave inverses and signed eigenspaces require omega_squared > 0 and
real nonzero omega satisfying omega**2 = omega_squared. Zero frequency is
treated in the balanced subspace. Slow amplitudes are held fixed.

The physical displacement includes epsilon and has zero exact fast mean.
Both drift averages use this displacement without subtracting its finite-window mean.

Requires SymPy. Run: python verification/check_drift.py
"""

import unittest

import sympy as sp


X, Y, Z, TAU, TIME, SHIFT = sp.symbols("x y z tau t s", real=True)
EPS, WINDOW = sp.symbols("epsilon T", positive=True)
POSITION = sp.Matrix([X, Y, Z])
COORDINATES = (X, Y, Z)


def symbol(k, l, m, rotation):
    """Density convention in (4.9), with m != 0."""
    return sp.Matrix([[0, -rotation, -k / m],
                      [rotation, 0, -l / m], [k / m, l / m, 0]])


def reconstruction(k, l, m):
    """Velocity reconstruction (4.12), with m != 0."""
    return sp.Matrix([[1, 0, 0], [0, 1, 0], [-k / m, -l / m, 0]])


def wavevector(label):
    k, l = sp.symbols(f"k_{label} l_{label}", real=True)
    m = sp.Symbol(f"m_{label}", real=True, nonzero=True)
    return sp.Matrix([k, l, m])


def velocity_amplitude(label, vector):
    """Arbitrary complex horizontal amplitudes; incompressibility supplies w."""
    u, v, density = sp.symbols(f"u_{label} v_{label} rho_{label}")
    return reconstruction(*vector) * sp.Matrix([u, v, density])


def phase(vector, frequency):
    return sp.exp(sp.I * vector.dot(POSITION) - sp.I * frequency * TAU)


def directional(vector, field):
    """Compute (vector dot grad) field by actual spatial differentiation."""
    return field.jacobian(COORDINATES) * vector


def centered_mean(expression, window=WINDOW):
    """Direct centered integral (3.2); no sinc formula is assumed."""
    return sp.integrate(expression.subs(TAU, TAU + SHIFT),
                        (SHIFT, -window / 2, window / 2)) / window


class DriftChecks(unittest.TestCase):
    def assert_zero(self, residual, context, dispersion=None):
        """Optionally reduce rational coefficients modulo omega**2-Omega**2."""
        entries = list(residual) if isinstance(residual, sp.MatrixBase) else [residual]
        for index, entry in enumerate(entries):
            reduced = sp.cancel(sp.expand_power_exp(sp.expand(entry.rewrite(sp.exp))))
            if dispersion is not None:
                omega, omega_squared = dispersion
                numerator = sp.fraction(sp.cancel(reduced))[0]
                reduced = sp.rem(numerator, omega**2 - omega_squared, omega)
                reduced = sp.simplify(reduced)
            self.assertEqual(reduced, 0, f"{context}; entry {index}: {reduced}")

    def test_density_symbol_and_dispersion(self):
        """(4.8b) gives the Fourier symbol (4.9) and dispersion relation (4.10)."""
        k, l, m = wavevector("a")
        rotation = sp.Symbol("F", real=True)
        u, v, density, eigenvalue = sp.symbols("u v rho lambda")
        spatial_phase = phase(sp.Matrix([k, l, m]), 0)
        # On a plane wave, partial_z^{-1} multiplies by 1/(i m).
        action = sp.Matrix([
            -rotation * v * spatial_phase
            - sp.diff(density * spatial_phase, X) / (sp.I * m),
            rotation * u * spatial_phase
            - sp.diff(density * spatial_phase, Y) / (sp.I * m),
            (sp.diff(u * spatial_phase, X) + sp.diff(v * spatial_phase, Y)) / (sp.I * m),
        ])
        fast = symbol(k, l, m, rotation)
        omega_squared = rotation**2 + (k**2 + l**2) / m**2
        self.assert_zero(action.jacobian((u, v, density)) / spatial_phase - fast,
                         "Fourier symbol from differential action")
        self.assert_zero(fast.T + fast, "real skew symmetry")
        self.assert_zero(fast.charpoly(eigenvalue).as_expr()
                         - eigenvalue * (eigenvalue**2 + omega_squared),
                         "eigenvalues 0 and +/- i omega")

    def test_balanced_projector_and_wave_inverse(self):
        """(4.11) and (4.15): inverse identities only where Omega**2 > 0."""
        k, l, m = wavevector("a")
        rotation = sp.Symbol("F", real=True)
        fast = symbol(k, l, m, rotation)
        omega_squared = rotation**2 + (k**2 + l**2) / m**2
        balanced_vector = sp.Matrix([l, -k, rotation * m])
        balanced = balanced_vector * balanced_vector.T / balanced_vector.dot(balanced_vector)
        wave = sp.eye(3) - balanced
        inverse = -fast / omega_squared
        self.assert_zero(fast * balanced_vector, "balanced kernel")
        self.assert_zero(balanced**2 - balanced, "balanced projection")
        self.assert_zero(balanced.T - balanced, "orthogonal balanced projection")
        self.assert_zero(fast**2 + omega_squared * wave, "wave projection")
        self.assert_zero(fast * inverse - wave, "left inverse on wave space")
        self.assert_zero(inverse * fast - wave, "right inverse on wave space")
        self.assert_zero(inverse * balanced, "zero extension on balanced space")
        self.assert_zero(sp.Matrix([[k, l, m]]) * reconstruction(k, l, m),
                         "incompressible reconstructed velocity")
        self.assert_zero(reconstruction(k, l, m)[:2, :] - sp.eye(3)[:2, :],
                         "reconstruction preserves both horizontal components")
        self.assertEqual(fast.det(), 0, "there is no full-state inverse")
        # The excluded Omega=0 case is evaluated without division: all of L is
        # zero, so its kernel is the entire state space and no wave space remains.
        self.assert_zero(fast.subs({k: 0, l: 0, rotation: 0}), "degenerate zero symbol")

    def test_arbitrary_signed_eigenspaces(self):
        """(4.9)–(4.11): arbitrary complex seeds cover either signed wave branch."""
        k, l, m = wavevector("a")
        rotation = sp.Symbol("F", real=True)
        omega = sp.Symbol("omega", real=True, nonzero=True)
        fast = symbol(k, l, m, rotation)
        omega_squared = rotation**2 + (k**2 + l**2) / m**2
        wave = -fast**2 / omega_squared
        signed = (wave + fast / (sp.I * omega)) / 2
        opposite = (wave - fast / (sp.I * omega)) / 2
        seed = sp.Matrix(sp.symbols("a_u a_v a_rho"))
        branch = signed * seed
        relation = (omega, omega_squared)
        self.assert_zero(fast * branch - sp.I * omega * branch,
                         "arbitrary seed projected to signed eigenmode", relation)
        self.assert_zero(signed**2 - signed, "signed eigenspace projection", relation)
        self.assert_zero(signed * opposite, "opposite branches are disjoint", relation)
        self.assert_zero(signed + opposite - wave, "both branches span wave space")
        self.assert_zero(sp.trace(signed) - 1, "signed projector has rank one")
        self.assert_zero(-fast * branch / omega_squared - branch / (sp.I * omega),
                         "wave inverse has multiplier 1/(i omega)", relation)
        # omega is signed: replacing it by -omega gives the other branch.
        self.assert_zero(signed.subs(omega, -omega) - opposite, "signed branch reversal")
        self.assert_zero(sp.conjugate(signed) - opposite, "conjugate branch reversal")
        self.assert_zero(symbol(-k, -l, -m, rotation) - fast,
                         "spatial conjugacy preserves the hydrostatic symbol")

    def test_physical_displacement_and_exact_mean(self):
        """(4.14)–(4.15), (4.18): epsilon scaling and the exact mean condition."""
        vector = wavevector("a")
        omega = sp.Symbol("omega", real=True, nonzero=True)
        amplitude = velocity_amplitude("a", vector)
        velocity = amplitude * phase(vector, omega)
        displacement = -EPS * velocity / (sp.I * omega)
        self.assert_zero(displacement.diff(TAU) - EPS * velocity,
                         "fast derivative of physical displacement")
        self.assert_zero(displacement.subs(TAU, TIME / EPS).diff(TIME)
                         - velocity.subs(TAU, TIME / EPS),
                         "physical-time derivative with slow amplitude frozen")
        # This is a direct integral, evaluated for either sign of nonzero omega.
        mean_phase = sp.simplify(centered_mean(sp.exp(-sp.I * omega * TAU)).subs(TAU, 0)
                                 .rewrite(sp.sin))
        self.assert_zero(sp.limit(mean_phase, WINDOW, sp.oo), "zero exact fast mean")
        # The finite mean is not identically zero; it tends to one as T -> 0.
        self.assert_zero(sp.limit(mean_phase, WINDOW, 0) - 1,
                         "a finite displacement mean must not be subtracted")

    def test_conjugate_fields_and_pair_coefficients(self):
        """(4.18)–(4.20): signed conjugate partners give real fields and drift."""
        vector_a, vector_b = wavevector("a"), wavevector("b")
        omega_a, omega_b = sp.symbols("omega_a omega_b", real=True, nonzero=True)
        amplitude_a = velocity_amplitude("a", vector_a)
        amplitude_b = velocity_amplitude("b", vector_b)
        velocity_a = amplitude_a * phase(vector_a, omega_a)
        velocity_b = amplitude_b * phase(vector_b, omega_b)
        displacement_a = -EPS * velocity_a / (sp.I * omega_a)
        partner_velocity = sp.conjugate(amplitude_a) * phase(-vector_a, -omega_a)
        partner_displacement = -EPS * partner_velocity / (sp.I * (-omega_a))
        self.assert_zero(partner_velocity - sp.conjugate(velocity_a), "velocity conjugate")
        self.assert_zero(partner_displacement - sp.conjugate(displacement_a),
                         "displacement conjugate, including signed denominator")
        pair = directional(displacement_a, velocity_b)
        partner_pair = directional(partner_displacement,
                                   sp.conjugate(amplitude_b) * phase(-vector_b, -omega_b))
        self.assert_zero(partner_pair - sp.conjugate(pair), "conjugate drift pairs")
        real_velocity = velocity_a + partner_velocity
        real_displacement = displacement_a + partner_displacement
        self.assert_zero(real_velocity - sp.conjugate(real_velocity), "real velocity")
        self.assert_zero(real_displacement - sp.conjugate(real_displacement), "real displacement")

    def test_pairing_from_spatial_derivatives(self):
        """(4.19)–(4.20): arbitrary wavevectors, amplitudes and signed frequencies."""
        vector_a, vector_b = wavevector("a"), wavevector("b")
        omega_a, omega_b = sp.symbols("omega_a omega_b", real=True, nonzero=True)
        amplitude_a = velocity_amplitude("a", vector_a)
        amplitude_b = velocity_amplitude("b", vector_b)
        displacement_a = -EPS * amplitude_a * phase(vector_a, omega_a) / (sp.I * omega_a)
        velocity_b = amplitude_b * phase(vector_b, omega_b)
        actual = directional(displacement_a, velocity_b)
        # The target is written independently of directional(): gradient supplies
        # i*k_b and the displacement supplies -epsilon/(i*omega_a).
        coefficient = -EPS * amplitude_a.dot(vector_b) * amplitude_b / omega_a
        expected = coefficient * phase(vector_a + vector_b, omega_a + omega_b)
        self.assert_zero(actual - expected, "ordered vector pair from spatial derivatives")
        self.assertNotEqual(sp.expand(coefficient[0]), 0,
                            "the arbitrary ordered pair is not identically zero")
        self.assert_zero(actual.diff(EPS) * EPS - actual, "physical epsilon scaling of drift")
        self.assert_zero(vector_a.dot(amplitude_a), "general amplitude is divergence free")
        # Same-wave and conjugate-wave pairings vanish for these homogeneous
        # incompressible plane waves. No nonzero single-wave drift is presumed.
        self.assert_zero(directional(displacement_a, amplitude_a * phase(vector_a, omega_a)),
                         "homogeneous plane-wave self-pair")
        self.assert_zero(directional(displacement_a,
                                     sp.conjugate(amplitude_a) * phase(-vector_a, -omega_a)),
                         "homogeneous plane-wave conjugate-pair")

    def test_exact_and_finite_pair_drift(self):
        """(4.21)–(4.23): integrate differentiated pairs; delta=0 is separate."""
        vector_a, vector_b = wavevector("a"), wavevector("b")
        omega_a = sp.Symbol("omega_a", real=True, nonzero=True)
        delta = sp.Symbol("delta", real=True, nonzero=True)
        amplitude_a = velocity_amplitude("a", vector_a)
        amplitude_b = velocity_amplitude("b", vector_b)
        displacement_a = -EPS * amplitude_a * phase(vector_a, omega_a) / (sp.I * omega_a)
        # delta - omega_a is the second signed wave frequency, and must also be
        # nonzero for an admissible wave pair. The ordered pair never divides by it.
        for mismatch in (sp.Integer(0), delta):
            with self.subTest(mismatch=mismatch):
                pair = directional(displacement_a,
                                   amplitude_b * phase(vector_b, mismatch - omega_a))
                coefficient = pair.subs(TAU, 0)
                self.assert_zero(pair - coefficient * sp.exp(-sp.I * mismatch * TAU),
                                 "combined temporal phase from actual differentiation")
                # The checked separation above lets each differentiated spatial
                # coefficient leave the time integral, without dividing by it.
                # Integrate the remaining actual temporal factor, not a supplied
                # sinc target. The subsequent identities apply to every component.
                integral = centered_mean(sp.exp(-sp.I * mismatch * TAU))
                exact_weight = sp.limit(sp.simplify(integral.subs(TAU, 0).rewrite(sp.sin)),
                                        WINDOW, sp.oo)
                finite = coefficient * integral
                exact = coefficient * exact_weight
                if mismatch == 0:
                    self.assert_zero(finite - coefficient, "resonant finite drift")
                    self.assert_zero(exact - coefficient, "exact resonance selector")
                    self.assert_zero(finite - exact, "resonant terms cancel in finite-minus-exact")
                else:
                    weight = sp.sin(delta * WINDOW / 2) / (delta * WINDOW / 2)
                    target = coefficient * sp.exp(-sp.I * delta * TAU) * weight
                    self.assert_zero(finite - target, "nonresonant finite drift")
                    self.assert_zero(exact, "nonzero mismatch removed by exact averaging")
                    self.assert_zero(finite - exact - target, "nonresonant finite-minus-exact")
                    self.assert_zero(sp.limit(weight, delta, 0) - 1,
                                     "continuous finite response at zero mismatch")
                    self.assert_zero(weight.subs(delta, -delta) - weight,
                                     "real even weight preserves conjugate-pair reality")
                    self.assert_zero(sp.conjugate(weight) - weight, "real finite-window weight")


if __name__ == "__main__":
    unittest.main(verbosity=2)
