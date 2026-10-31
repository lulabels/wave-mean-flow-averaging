"""Modal transforms and field comparisons for the dense RSW runs."""
from pathlib import Path
import hashlib
import sys
import numpy as np
PKG = REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "figures"))
from wave_averaging.rsw.modal_reduced import modal_matrices
import figures7_8 as fig

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class BatchedModalMap:
    """Batched transforms using the modal_matrices basis.

    No interaction table is needed. run_refined_fig78.py self-test compares
    these transforms with the modal backend.
    """

    def __init__(self, grid, params):
        self.grid = grid
        self.ix, self.iy = np.nonzero(grid.dealias_mask)
        matrices = [modal_matrices(grid.kx_1d[i], grid.ky_1d[j], params)
                    for i, j in zip(self.ix, self.iy)]
        self.r = np.asarray([r for r, _ in matrices])
        self.lam = np.asarray([lam for _, lam in matrices])
        positive = np.sqrt(params.f ** 2 + params.g * params.H *
                           (grid.kx_1d[self.ix] ** 2 + grid.ky_1d[self.iy] ** 2))
        self.omega = np.stack((np.zeros_like(positive), positive, -positive), axis=1).ravel()
        self.modal_indices = range(self.omega.size)

    def state_to_modal(self, state):
        q = np.fft.fft2(state.as_array(), axes=(-2, -1)) / self.grid.N ** 2
        return np.einsum("mij,mj->mi", self.lam, q[:, self.ix, self.iy].T).ravel()

    def modal_to_state(self, amplitudes):
        q = np.zeros((3, self.grid.N, self.grid.N), dtype=complex)
        q[:, self.ix, self.iy] = np.einsum(
            "mij,mj->mi", self.r, np.asarray(amplitudes).reshape(-1, 3)
        ).T
        arr = np.fft.ifft2(q * self.grid.N ** 2, axes=(-2, -1)).real
        return fig.RSWState.from_array(arr)


def field_error(low, high):
    """Spectral restriction plus an explicit unresolved-tail contribution.

    The restricted error follows the earlier convergence norm. The total error
    also includes high-reference modes absent from the low two-thirds mask, so
    spectral restriction cannot hide missing fine scales. Parseval supplies both.
    """
    nl, nh = low.shape[-1], high.shape[-1]
    low_hat = np.fft.fft2(low, axes=(-2, -1)) / nl ** 2
    high_hat = np.fft.fft2(high, axes=(-2, -1)) / nh ** 2
    modes = np.rint(np.fft.fftfreq(nl, d=1.0 / nl)).astype(int)
    indices = modes % nh
    restricted = high_hat[:, indices[:, None], indices[None, :]]
    mask = fig.SpectralGrid(nl).dealias_mask
    target = restricted * mask
    common_sq = float(np.sum(abs(target) ** 2))
    high_sq = float(np.sum(abs(high_hat) ** 2))
    error_sq = float(np.sum(abs(low_hat - target) ** 2))
    high_common_mask = np.zeros((nh, nh), dtype=bool)
    high_common_mask[indices[:, None], indices[None, :]] = mask
    tail_sq = float(np.sum(abs(high_hat[:, ~high_common_mask]) ** 2))
    return {
        "relative_rms_restricted": float(np.sqrt(error_sq / max(common_sq, 1e-30))),
        "relative_rms_including_unresolved_modes": float(np.sqrt((error_sq + tail_sq) / max(high_sq, 1e-30))),
        "reference_unresolved_rms_fraction": float(np.sqrt(tail_sq / max(high_sq, 1e-30))),
    }

def compare(low, high):
    lc, lf, lo = low
    hc, hf, ho = high
    for key in ("Lx", "Ly", "params", "initial_condition"):
        if lc[key] != hc[key]:
            raise ValueError(f"unmatched setup: {key}")
    errors = {key: field_error(lf[key], hf[key]) for key in lf}
    changes = {key: abs(lo[key] - ho[key]) for key in lo}
    return {
        "fields": errors, "observable_absolute_changes": changes,
        "max_field_relative_rms": max(v["relative_rms_including_unresolved_modes"] for v in errors.values()),
        "max_observable_absolute_change": max(changes.values()),
    }
