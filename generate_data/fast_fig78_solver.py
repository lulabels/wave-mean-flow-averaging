"""Threaded real-FFT RSW solver.

Uses the same equations, strict two-thirds mask and RK4 stages as
PseudospectralRSWSolver, with batched real FFTs and a final projection.
"""
import numpy as np
from scipy import fft

from wave_averaging.rsw.solver_pseudospectral import PseudospectralRSWSolver, RSWState


class FastRSWSolver(PseudospectralRSWSolver):
    def __init__(self, grid, params, workers=1):
        super().__init__(grid, params)
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.shape = (grid.N, grid.N)
        self.mask = grid.dealias_mask[:, :grid.N // 2 + 1]
        self.dx = 1j * grid.KX[:, :grid.N // 2 + 1].copy()
        self.dy = 1j * grid.KY[:, :grid.N // 2 + 1].copy()
        # A real collocation derivative has zero Nyquist contribution, as in
        # canonical ifft2(1j*k*fft2(a)).real. The retained mask also kills it.
        self.dx[grid.N // 2, :] = 0
        self.dy[:, grid.N // 2] = 0
        self.k2 = grid.K2[:, :grid.N // 2 + 1]

    def forward(self, x):
        return fft.rfft2(x, workers=self.workers)

    def inverse(self, x):
        return fft.irfft2(x, s=self.shape, workers=self.workers)

    def project(self, state):
        return RSWState.from_array(self.inverse(self.forward(state.as_array()) * self.mask))

    def rhs(self, state):
        p = self.params
        u, v, eta = state.u, state.v, state.eta
        if any(a.dtype != np.float64 for a in (u, v, eta)):
            raise TypeError("fast backend requires float64 evolution states")
        sh = self.forward(state.as_array())
        gradients = self.inverse(np.stack((self.dx * sh[0], self.dy * sh[0],
                                           self.dx * sh[1], self.dy * sh[1])))
        ux, uy, vx, vy = gradients
        h = p.H + eta
        nonlinear = self.forward(np.stack((-(u * ux + v * uy),
                                           -(u * vx + v * vy), h * u, h * v)))
        rh = np.stack((nonlinear[0] + p.f * sh[1] - p.g * self.dx * sh[2],
                       nonlinear[1] - p.f * sh[0] - p.g * self.dy * sh[2],
                       -self.dx * nonlinear[2] - self.dy * nonlinear[3]))
        if p.nu:
            rh -= p.nu * self.k2 * sh
        return RSWState.from_array(self.inverse(rh * self.mask))

    def step_rk4(self, state, dt):
        def add_scaled(a, b, scale):
            return RSWState(a.u + scale * b.u, a.v + scale * b.v, a.eta + scale * b.eta)

        k1 = self.rhs(state)
        k2 = self.rhs(add_scaled(state, k1, 0.5 * dt))
        k3 = self.rhs(add_scaled(state, k2, 0.5 * dt))
        k4 = self.rhs(add_scaled(state, k3, dt))
        out = RSWState(
            state.u + dt * (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u) / 6.0,
            state.v + dt * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v) / 6.0,
            state.eta + dt * (k1.eta + 2.0 * k2.eta + 2.0 * k3.eta + k4.eta) / 6.0)
        return self.project(out)
