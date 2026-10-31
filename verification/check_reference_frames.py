"""Symbolic checks for reference-frame identities in §5 and Appendix B.

Check modal averaging kernels, generator identities for steady shear and
wave evolution, and frequency shifts under uniform advection.

Times, positions and frequencies are real, with boxcar length T > 0. Modal
checks use a time-independent generator with a complete finite eigenbasis,
including repeated and zero frequencies. Histories are smooth where
differentiated and integrable over the window. The shear is U(y); the Doppler
check uses one nonzero Fourier mode and prescribed uniform velocity at linear order.

Requires SymPy. Run: python verification/check_reference_frames.py
"""

import unittest

import sympy as sp


def boxcar(expression, s, width):
    return sp.integrate(expression, (s, -width / 2, width / 2)) / width


class ReferenceFrameChecks(unittest.TestCase):
    def assert_zero(self, expression):
        entries = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
        for entry in entries:
            residual = sp.simplify(sp.expand(entry.rewrite(sp.exp)))
            self.assertEqual(residual, 0, f"Nonzero symbolic residual: {residual}")

    def test_inner_and_combined_modal_kernels(self):
        """(5.8a,b): arbitrary modal histories and scalar kernels."""
        t, s = sp.symbols("t s", real=True)
        width = sp.Symbol("T", positive=True)
        omega = sp.Symbol("omega", real=True)
        outer = sp.exp(-sp.I * omega * t)
        inner = sp.exp(sp.I * omega * (t + s))
        combined = sp.exp(sp.I * omega * s)
        kernel = sp.Function("K_T")(s)
        history = sp.Function("w")(t + s)
        # Equality before integration covers every integrable modal history.
        self.assert_zero(outer * kernel * inner * history - kernel * combined * history)
        envelope = sp.Function("a")(t + s)
        modulated_wave = sp.exp(-sp.I * omega * (t + s)) * envelope
        inner_samples = sp.simplify(inner * modulated_wave)
        combined_samples = sp.simplify(combined * modulated_wave)
        # An arbitrary envelope is left under its integral, without assuming a
        # closed form or a convergent Taylor expansion for that envelope.
        envelope_mean = sp.Integral(envelope, (s, -width / 2, width / 2)) / width
        mapped_average = outer * sp.Integral(inner_samples, (s, -width / 2, width / 2)) / width
        self.assert_zero(mapped_average - outer * envelope_mean)
        self.assert_zero(combined_samples - outer * envelope)
        amplitude = sp.Symbol("A")
        self.assert_zero(boxcar(combined_samples.subs(envelope, amplitude), s, width) - outer * amplitude)
        self.assert_zero((outer * inner - combined).subs(omega, 0))

    def test_scalar_generators_on_a_smooth_shear_history(self):
        """(5.11), (5.15a,b): arbitrary smooth scalar and steady shear."""
        x, y, t, s, r = sp.symbols("x y t s r", real=True)
        theta = sp.Function("theta")(x, y, t)
        shear = sp.Function("U")(y)
        for name, advector in (("Eulerian", 0), ("Lagrangian", shear)):
            with self.subTest(frame=name):
                trajectory = sp.Matrix([x + advector * s, y])
                velocity = sp.Matrix([advector, 0])
                sample_point = {x: x + advector * s, t: t + s}
                self.assert_zero(trajectory.subs(s, 0) - sp.Matrix([x, y]))
                self.assert_zero(sp.diff(trajectory, s) - velocity.subs(sample_point, simultaneous=True))
                history = theta.subs(sample_point, simultaneous=True)
                generator = sp.diff(theta, t) + advector * sp.diff(theta, x)
                self.assert_zero(
                    sp.diff(history, s) - generator.subs(sample_point, simultaneous=True)
                )
                self.assert_zero(history.subs(s, 0) - theta)
                # Exact composition identifies this trajectory pullback for all
                # offsets in its domain, without a formal exponential series.
                twice = history.subs({x: x + advector * r, t: t + r}, simultaneous=True)
                self.assert_zero(twice - history.subs(s, s + r))

    def test_wave_generator_on_an_arbitrary_smooth_history(self):
        """(5.14), (5.15c): arbitrary eigenfrequency and smooth history."""
        t, s = sp.symbols("t s", real=True)
        omega = sp.Symbol("omega", real=True)
        history = sp.Function("w")(t)
        evolved = sp.exp(sp.I * omega * s) * history.subs(t, t + s)
        # D_W = partial_t + L_F; time independence of L_F is essential here.
        self.assert_zero(sp.diff(evolved, s) - (sp.diff(evolved, t) + sp.I * omega * evolved))
        self.assert_zero(evolved.subs(s, 0) - history)
        self.assert_zero(evolved.subs(omega, 0) - history.subs(t, t + s))

    def test_uniform_flow_doppler_relation_from_the_pde(self):
        """(5.16): differentiate the plane wave to obtain the Doppler relation."""
        x, y, t, s = sp.symbols("x y t s", real=True)
        k, ell, U, V, omega_e, omega_f = sp.symbols("k ell U V omega_E omega_F", real=True)
        amplitude = sp.Symbol("A", nonzero=True)
        mode = amplitude * sp.exp(sp.I * (k * x + ell * y - omega_e * t))
        # Work in one eigenspace, where L_F acts as multiplication by i omega_F.
        residual = sp.diff(mode, t) + U * sp.diff(mode, x) + V * sp.diff(mode, y) + sp.I * omega_f * mode
        frequency_equation = sp.simplify(residual / mode)
        roots = sp.solve(frequency_equation, omega_e)
        self.assertEqual(roots, [k * U + ell * V + omega_f])
        solution = mode.subs(omega_e, roots[0])
        self.assert_zero(residual.subs(omega_e, roots[0]))
        trajectory = sp.Matrix([x + U * s, y + V * s])
        self.assert_zero(sp.diff(trajectory, s) - sp.Matrix([U, V]))
        self.assert_zero(trajectory.subs(s, 0) - sp.Matrix([x, y]))
        lagrangian = solution.subs({x: trajectory[0], y: trajectory[1], t: t + s}, simultaneous=True)
        eulerian = solution.subs(t, t + s)
        self.assert_zero(lagrangian - solution * sp.exp(-sp.I * omega_f * s))
        self.assert_zero(eulerian - solution * sp.exp(-sp.I * roots[0] * s))

    def test_frame_frequency_response_for_signed_and_zero_frequencies(self):
        """(B7), specialized to a centered boxcar."""
        s = sp.Symbol("s", real=True)
        frequency = sp.Symbol("omega_R", real=True, nonzero=True)
        width = sp.Symbol("T", positive=True)
        amplitude = sp.Symbol("a_R")
        # The same integral applies to either frame's record; omega_R is signed.
        history = amplitude * sp.exp(-sp.I * frequency * s)
        actual = boxcar(history, s, width)
        target = amplitude * sp.sin(frequency * width / 2) / (frequency * width / 2)
        self.assert_zero(actual - target)
        self.assert_zero(boxcar(history.subs(frequency, 0), s, width) - amplitude)
        self.assert_zero(actual.subs(frequency, -frequency) - actual)
        self.assert_zero(sp.limit(target, frequency, 0) - amplitude)

    def test_balanced_mode_advected_past_a_fixed_observer(self):
        """(5.4), (5.16), (B7): all f != 0, g,H > 0 and uniform U,V."""
        x, y, t, s = sp.symbols("x y t s", real=True)
        f = sp.Symbol("f", real=True, nonzero=True)
        g, depth = sp.symbols("g H", positive=True)
        k, ell, U, V = sp.symbols("k ell U V", real=True)
        amplitude = sp.Symbol("A")
        width = sp.Symbol("T", positive=True)
        fast = sp.Matrix([[0, -f, sp.I * g * k], [f, 0, sp.I * g * ell], [sp.I * depth * k, sp.I * depth * ell, 0]])
        # Scaling by f avoids division and keeps the k=ell=0 balanced mode.
        balanced = sp.Matrix([-sp.I * g * ell, sp.I * g * k, f])
        self.assert_zero(fast * balanced)
        sweeping = k * U + ell * V
        field = amplitude * balanced * sp.exp(sp.I * (k * x + ell * y - sweeping * t))
        self.assert_zero(sp.diff(field, t) + U * sp.diff(field, x) + V * sp.diff(field, y) + fast * field)
        eulerian = field.subs(t, t + s)
        lagrangian = field.subs({x: x + U * s, y: y + V * s, t: t + s}, simultaneous=True)
        self.assert_zero(lagrangian - field)
        self.assert_zero(eulerian - field * sp.exp(-sp.I * sweeping * s))
        # Integrate an independent real frequency first, so the zero-sweeping
        # branch is retained even when k*U+ell*V vanishes by cancellation.
        frequency = sp.Symbol("omega_E", real=True)
        response = boxcar(sp.exp(-sp.I * frequency * s), s, width)
        target = sp.Piecewise(
            (sp.sin(frequency * width / 2) / (frequency * width / 2), sp.Ne(frequency, 0)),
            (1, True),
        )
        self.assert_zero(response - target)
        eulerian_mean = field * response.subs(frequency, sweeping)
        self.assert_zero(eulerian_mean - field * target.subs(frequency, sweeping))
        lagrangian_mean = lagrangian.applyfunc(lambda q: boxcar(q, s, width))
        self.assert_zero(lagrangian_mean - field)
        self.assert_zero(response.subs(frequency, 0) - 1)
        magnitude = sp.Symbol("Omega", positive=True)
        cycles = sp.Symbol("n", integer=True, positive=True)
        for sign in (-1, 1):
            with self.subTest(sweeping_sign=sign):
                self.assert_zero(response.subs({frequency: sign * magnitude, width: 2 * sp.pi * cycles / magnitude}))
        # Certify the closed-form fast propagator by its ODE and initial data;
        # no generic symbolic matrix exponential is needed.
        omega = sp.sqrt(f**2 + g * depth * (k**2 + ell**2))
        orbit = sp.eye(3) - sp.sin(omega * s) * fast / omega + (1 - sp.cos(omega * s)) * fast**2 / omega**2
        self.assert_zero(sp.diff(orbit, s) + fast * orbit)
        self.assert_zero(orbit.subs(s, 0) - sp.eye(3))
        self.assert_zero(orbit * balanced - balanced)


if __name__ == "__main__":
    unittest.main(verbosity=2)
