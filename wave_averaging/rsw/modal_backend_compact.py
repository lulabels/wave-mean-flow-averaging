r"""Compact-array implementation of the modal RSW system.

The finite_window_rhs and RK4 calling conventions, time units, window origins
and optional-model scope are documented at the top of modal_reduced.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
from pathlib import Path
import time
from sys import getsizeof
from typing import Any

import numpy as np

from .modal_reduced import (
    BRANCHES,
    STRICT_RESONANCE_TOL,
    Branch,
    ModalSystem,
    WindowTimeUnit,
    WindowOrigin,
    current_wave_phase,
    validate_window_origin,
    finite_window_multiplier,
    ordered_pair_bilinear,
)
from .solver_pseudospectral import RSWParameters, SpectralGrid


Array = np.ndarray
BRANCH_TO_POSITION: dict[Branch, int] = {sigma: j for j, sigma in enumerate(BRANCHES)}
CACHE_FORMAT_VERSION = 1
STRICT_DEALIAS_MASK_CONVENTION = "strict_2_3_tensor_product_excludes_abs_m_eq_N_over_3"
BRANCH_POLICY = "branches=(0,+1,-1); retain_k0_inertial_branches=True"
DELTA_CONVENTION = "delta=omega_a+omega_b-omega_c"
FINITE_TRANSFER_CONVENTION = "fast_time_detuning=-delta; physical_time_detuning=-delta/epsilon"


@dataclass(frozen=True)
class CompactInteractionArrays:
    """Flat NumPy storage for ordered modal interactions."""

    c_idx: Array
    a_idx: Array
    b_idx: Array
    coefficient: Array
    delta: Array

    @property
    def count(self) -> int:
        return int(self.c_idx.size)

    @property
    def nbytes(self) -> int:
        return int(
            self.c_idx.nbytes
            + self.a_idx.nbytes
            + self.b_idx.nbytes
            + self.coefficient.nbytes
            + self.delta.nbytes
        )

    def save_npz(self, path: str | Path, metadata: dict[str, Any]) -> None:
        payload_metadata = _metadata_with_kind(metadata, "compact_interaction_arrays")
        np.savez_compressed(
            path,
            c_idx=self.c_idx,
            a_idx=self.a_idx,
            b_idx=self.b_idx,
            coefficient=self.coefficient,
            delta=self.delta,
            metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(cls, path: str | Path, expected_metadata: dict[str, Any] | None = None) -> "CompactInteractionArrays":
        with np.load(path, allow_pickle=False) as data:
            metadata = _load_metadata(data)
            _validate_cache_kind(metadata, "compact_interaction_arrays")
            if expected_metadata is not None:
                _validate_metadata(metadata, _metadata_with_kind(expected_metadata, "compact_interaction_arrays"))
            arrays = cls(
                c_idx=np.asarray(data["c_idx"], dtype=np.int32),
                a_idx=np.asarray(data["a_idx"], dtype=np.int32),
                b_idx=np.asarray(data["b_idx"], dtype=np.int32),
                coefficient=np.asarray(data["coefficient"], dtype=np.complex128),
                delta=np.asarray(data["delta"], dtype=np.float64),
            )
        _validate_interaction_arrays_shapes(arrays)
        return arrays


@dataclass(frozen=True)
class CompactDeltaLookup:
    """Unique detunings and inverse map for interaction-weight expansion."""

    unique_delta: Array
    inverse: Array

    @property
    def nbytes(self) -> int:
        return int(self.unique_delta.nbytes + self.inverse.nbytes)


FiniteWindowWeightKey = tuple[str, float, float, WindowTimeUnit, int, WindowOrigin]


@dataclass(frozen=True)
class CompactFiniteWindowWeights:
    """Precomputed finite-window multipliers aligned with compact interactions."""

    key: FiniteWindowWeightKey
    weights: Array
    unique_delta: Array
    unique_weights: Array
    source_id: int

    @property
    def count(self) -> int:
        return int(self.weights.size)

    @property
    def unique_count(self) -> int:
        return int(self.unique_delta.size)

    @property
    def nbytes(self) -> int:
        return int(self.weights.nbytes + self.unique_delta.nbytes + self.unique_weights.nbytes)

    def save_npz(self, path: str | Path, metadata: dict[str, Any]) -> None:
        payload_metadata = _metadata_with_kind(metadata, "compact_finite_window_weights")
        np.savez_compressed(
            path,
            weights=self.weights,
            unique_delta=self.unique_delta,
            unique_weights=self.unique_weights,
            key_json=np.asarray(json.dumps(_finite_window_key_to_jsonable(self.key))),
            metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True)),
        )

    @classmethod
    def load_npz(
        cls,
        path: str | Path,
        expected_metadata: dict[str, Any] | None = None,
        *,
        source_id: int,
    ) -> "CompactFiniteWindowWeights":
        with np.load(path, allow_pickle=False) as data:
            metadata = _load_metadata(data)
            _validate_cache_kind(metadata, "compact_finite_window_weights")
            if expected_metadata is not None:
                _validate_metadata(metadata, _metadata_with_kind(expected_metadata, "compact_finite_window_weights"))
            key = _finite_window_key_from_jsonable(json.loads(str(data["key_json"].tolist())))
            weights = cls(
                key=key,
                weights=np.asarray(data["weights"], dtype=np.complex128),
                unique_delta=np.asarray(data["unique_delta"], dtype=np.float64),
                unique_weights=np.asarray(data["unique_weights"], dtype=np.complex128),
                source_id=source_id,
            )
        _validate_finite_window_weight_shapes(weights)
        return weights


@dataclass
class CompactModalSystem:
    """Array-backed modal RSW reduced model.

    This class preserves the small-grid reference algebra in
    :class:`rswfe.modal_reduced.ModalSystem`, but stores ordered interactions in
    contiguous NumPy arrays instead of one Python dataclass object per triad.
    RHS evaluation is chunked and uses vectorized products with ``np.add.at``.
    """

    grid: SpectralGrid
    params: RSWParameters
    resonance_tol: float = STRICT_RESONANCE_TOL
    chunk_size: int = 1_000_000
    _finite_window_weight_cache: dict[FiniteWindowWeightKey, CompactFiniteWindowWeights] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    @cached_property
    def reference(self) -> ModalSystem:
        return ModalSystem(self.grid, self.params, resonance_tol=self.resonance_tol)

    @property
    def modes(self):
        return self.reference.modes

    @property
    def mode_by_key(self):
        return self.reference.mode_by_key

    @property
    def modal_indices(self):
        return self.reference.modal_indices

    @property
    def index_by_key(self):
        return self.reference.index_by_key

    @property
    def omega(self) -> Array:
        return self.reference.omega

    @property
    def mean_inertial_indices(self) -> tuple[int, int]:
        return self.reference.mean_inertial_indices

    @property
    def r_by_mode(self):
        return self.reference.r_by_mode

    @property
    def lambda_by_mode(self):
        return self.reference.lambda_by_mode

    @cached_property
    def interaction_arrays(self) -> CompactInteractionArrays:
        build_start = time.perf_counter()
        modal_count = len(self.modal_indices)
        _progress_log(
            "building compact interactions",
            N=self.grid.N,
            modal_dof=modal_count,
            chunk_size=self.chunk_size,
        )
        c_idx: list[int] = []
        a_idx: list[int] = []
        b_idx: list[int] = []
        coefficients: list[complex] = []
        deltas: list[float] = []

        next_progress_time = build_start + 60.0
        progress_stride = max(1, modal_count // 20)
        for c_counter, c_modal in enumerate(self.modal_indices, start=1):
            mc = c_modal.mode
            lambda_row = self.lambda_by_mode[mc.key][BRANCH_TO_POSITION[c_modal.sigma]]
            for a_modal in self.modal_indices:
                ma = a_modal.mode
                mb_key = (mc.mx - ma.mx, mc.my - ma.my)
                mb = self.mode_by_key.get(mb_key)
                if mb is None:
                    continue
                ra = self.r_by_mode[ma.key][:, BRANCH_TO_POSITION[a_modal.sigma]]
                for sigma_b in BRANCHES:
                    b_flat = self.index_by_key[(mb.mx, mb.my, sigma_b)]
                    b_modal = self.modal_indices[b_flat]
                    rb = self.r_by_mode[mb.key][:, BRANCH_TO_POSITION[sigma_b]]
                    vec = ordered_pair_bilinear((ma.kx, ma.ky), ra, (mb.kx, mb.ky), rb)
                    c_idx.append(c_modal.flat_index)
                    a_idx.append(a_modal.flat_index)
                    b_idx.append(b_flat)
                    coefficients.append(complex(lambda_row @ vec))
                    deltas.append(float(a_modal.omega + b_modal.omega - c_modal.omega))
            now = time.perf_counter()
            if c_counter == modal_count or c_counter % progress_stride == 0 or now >= next_progress_time:
                _progress_log(
                    "building compact interactions",
                    N=self.grid.N,
                    completed_c_modes=c_counter,
                    total_c_modes=modal_count,
                    interactions_so_far=len(c_idx),
                    elapsed_s=f"{now - build_start:.1f}",
                    maxrss=_maxrss_human(),
                )
                next_progress_time = now + 60.0

        arrays = CompactInteractionArrays(
            c_idx=np.asarray(c_idx, dtype=np.int32),
            a_idx=np.asarray(a_idx, dtype=np.int32),
            b_idx=np.asarray(b_idx, dtype=np.int32),
            coefficient=np.asarray(coefficients, dtype=np.complex128),
            delta=np.asarray(deltas, dtype=np.float64),
        )
        _progress_log(
            "built compact interactions",
            N=self.grid.N,
            interactions=arrays.count,
            storage=_nbytes_human(arrays.nbytes),
            elapsed_s=f"{time.perf_counter() - build_start:.1f}",
            maxrss=_maxrss_human(),
        )
        return arrays

    @property
    def interaction_count(self) -> int:
        return self.interaction_arrays.count

    @cached_property
    def exact_resonant_indices(self) -> Array:
        return np.flatnonzero(np.abs(self.interaction_arrays.delta) <= self.resonance_tol).astype(np.int64)

    def interaction_storage_nbytes(self) -> int:
        return self.interaction_arrays.nbytes

    def interaction_cache_metadata(self) -> dict[str, Any]:
        return _system_cache_metadata(self.grid, self.params, self.resonance_tol)

    def save_interaction_arrays(self, path: str | Path) -> None:
        _progress_log(
            "saving compact interaction cache",
            path=str(path),
            interactions=self.interaction_arrays.count,
            storage=_nbytes_human(self.interaction_arrays.nbytes),
            maxrss=_maxrss_human(),
        )
        start = time.perf_counter()
        self.interaction_arrays.save_npz(path, self.interaction_cache_metadata())
        _progress_log(
            "saved compact interaction cache",
            path=str(path),
            elapsed_s=f"{time.perf_counter() - start:.1f}",
            file_size=_file_size_human(path),
            maxrss=_maxrss_human(),
        )

    def load_interaction_arrays(self, path: str | Path) -> CompactInteractionArrays:
        _progress_log("loading compact interaction cache", path=str(path), file_size=_file_size_human(path))
        start = time.perf_counter()
        arrays = CompactInteractionArrays.load_npz(path, self.interaction_cache_metadata())
        self.__dict__["interaction_arrays"] = arrays
        self.__dict__.pop("exact_resonant_indices", None)
        self.__dict__.pop("finite_window_delta_lookup", None)
        self.clear_finite_window_weight_cache()
        _progress_log(
            "loaded compact interaction cache",
            path=str(path),
            interactions=arrays.count,
            storage=_nbytes_human(arrays.nbytes),
            elapsed_s=f"{time.perf_counter() - start:.1f}",
            maxrss=_maxrss_human(),
        )
        return arrays

    @cached_property
    def finite_window_delta_lookup(self) -> CompactDeltaLookup:
        _progress_log(
            "building finite-window delta lookup",
            interactions=self.interaction_arrays.count,
            maxrss=_maxrss_human(),
        )
        start = time.perf_counter()
        unique_delta, inverse = np.unique(self.interaction_arrays.delta, return_inverse=True)
        lookup = CompactDeltaLookup(
            unique_delta=unique_delta.astype(np.float64, copy=False),
            inverse=inverse.astype(np.int64, copy=False),
        )
        _progress_log(
            "built finite-window delta lookup",
            unique_delta=lookup.unique_delta.size,
            storage=_nbytes_human(lookup.nbytes),
            elapsed_s=f"{time.perf_counter() - start:.1f}",
            maxrss=_maxrss_human(),
        )
        return lookup

    def finite_window_weight_cache_size(self) -> int:
        return len(self._finite_window_weight_cache)

    def clear_finite_window_weight_cache(self) -> None:
        self._finite_window_weight_cache.clear()

    def estimated_reference_interaction_nbytes(self) -> int:
        """Rough lower-bound memory estimate for ModalSystem Interaction objects."""

        object_size = getsizeof(self.reference_interaction_object_sample())
        int_size = getsizeof(0)
        complex_size = getsizeof(0.0 + 0.0j)
        float_size = getsizeof(0.0)
        per_interaction = object_size + 3 * int_size + complex_size + float_size
        return int(self.interaction_count * per_interaction)

    def reference_interaction_object_sample(self):
        from .modal_reduced import Interaction

        return Interaction(c=0, a=0, b=0, coefficient=0.0 + 0.0j, delta=0.0)

    def zero_amplitudes(self) -> Array:
        return self.reference.zero_amplitudes()

    def modal_to_fourier(self, amplitudes: Array) -> Array:
        return self.reference.modal_to_fourier(amplitudes)

    def fourier_to_modal(self, q_hat: Array) -> Array:
        return self.reference.fourier_to_modal(q_hat)

    def state_to_modal(self, state) -> Array:
        return self.reference.state_to_modal(state)

    def modal_to_state(self, amplitudes: Array, *, real: bool = True):
        return self.reference.modal_to_state(amplitudes, real=real)

    def conjugate_symmetric_amplitudes(self, rng: np.random.Generator, scale: float = 1.0) -> Array:
        return self.reference.conjugate_symmetric_amplitudes(rng, scale=scale)

    def nonlinear_rhs_modal(self, amplitudes: Array) -> Array:
        return self._weighted_rhs(amplitudes)

    def exact_resonant_rhs(self, amplitudes: Array, *, epsilon: float = 1.0) -> Array:
        _validate_epsilon(epsilon)
        return self._weighted_rhs(amplitudes, selected=self.exact_resonant_indices)

    def finite_window_rhs(
        self,
        amplitudes: Array,
        *,
        time: float,
        epsilon: float,
        window_name: str,
        window_length: float,
        time_unit: WindowTimeUnit = "fast",
        nquad: int = 2048,
        window_origin: WindowOrigin = "forward",
        precomputed_weights: CompactFiniteWindowWeights | None = None,
    ) -> Array:
        _validate_epsilon(epsilon)
        if window_length <= 0.0:
            raise ValueError("window_length must be positive")
        key = _finite_window_weight_key(
            window_name=window_name,
            window_length=window_length,
            epsilon=epsilon,
            time_unit=time_unit,
            nquad=nquad,
            window_origin=window_origin,
        )
        if precomputed_weights is None:
            precomputed_weights = self.precompute_finite_window_weights(
                epsilon=epsilon,
                window_name=window_name,
                window_length=window_length,
                time_unit=time_unit,
                nquad=nquad,
                window_origin=window_origin,
            )
        else:
            self._validate_precomputed_finite_window_weights(precomputed_weights, key)
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        if amplitudes.shape != (len(self.modal_indices),):
            raise ValueError("amplitudes must have one entry per modal index")
        phase = current_wave_phase(self.omega, time=time, epsilon=epsilon)
        return self._weighted_rhs(amplitudes * phase, precomputed_weights.weights) * np.conjugate(phase)

    def precompute_finite_window_weights(
        self,
        *,
        epsilon: float,
        window_name: str,
        window_length: float,
        time_unit: WindowTimeUnit = "fast",
        nquad: int = 2048,
        window_origin: WindowOrigin = "forward",
    ) -> CompactFiniteWindowWeights:
        """Return cached finite-window multipliers for the compact interaction table.

        The detuning convention remains centralized in
        :func:`finite_window_multiplier`: ``delta = omega_a + omega_b - omega_c``
        is passed as a signed value, and that helper sends ``-delta`` in fast
        time or ``-delta/epsilon`` in physical time to the transfer function.
        """

        _validate_epsilon(epsilon)
        if window_length <= 0.0:
            raise ValueError("window_length must be positive")
        key = _finite_window_weight_key(
            window_name=window_name,
            window_length=window_length,
            epsilon=epsilon,
            time_unit=time_unit,
            nquad=nquad,
            window_origin=window_origin,
        )
        cached = self._finite_window_weight_cache.get(key)
        if cached is not None:
            return cached

        lookup = self.finite_window_delta_lookup
        _progress_log(
            "building finite-window weights",
            window=window_name,
            window_length=window_length,
            time_unit=time_unit,
            nquad=nquad,
            window_origin=window_origin,
            unique_delta=lookup.unique_delta.size,
            interactions=self.interaction_arrays.count,
            maxrss=_maxrss_human(),
        )
        start = time.perf_counter()
        unique_weights = np.empty(lookup.unique_delta.size, dtype=np.complex128)
        next_progress_time = start + 60.0
        progress_stride = max(1, lookup.unique_delta.size // 10)
        for j, delta in enumerate(lookup.unique_delta):
            unique_weights[j] = finite_window_multiplier(
                window_name,
                float(delta),
                window_length=window_length,
                epsilon=epsilon,
                time_unit=time_unit,
                nquad=nquad,
                window_origin=window_origin,
            )
            now = time.perf_counter()
            if j + 1 == lookup.unique_delta.size or (j + 1) % progress_stride == 0 or now >= next_progress_time:
                _progress_log(
                    "building finite-window weights",
                    completed_unique_delta=j + 1,
                    total_unique_delta=lookup.unique_delta.size,
                    elapsed_s=f"{now - start:.1f}",
                    maxrss=_maxrss_human(),
                )
                next_progress_time = now + 60.0
        weights = np.asarray(unique_weights[lookup.inverse], dtype=np.complex128)
        cached = CompactFiniteWindowWeights(
            key=key,
            weights=weights,
            unique_delta=lookup.unique_delta,
            unique_weights=unique_weights,
            source_id=id(self),
        )
        self._finite_window_weight_cache[key] = cached
        _progress_log(
            "built finite-window weights",
            weights=cached.count,
            unique_delta=cached.unique_count,
            storage=_nbytes_human(cached.nbytes),
            elapsed_s=f"{time.perf_counter() - start:.1f}",
            maxrss=_maxrss_human(),
        )
        return cached

    def finite_window_weight_cache_metadata(
        self,
        *,
        epsilon: float,
        window_name: str,
        window_length: float,
        time_unit: WindowTimeUnit = "fast",
        nquad: int = 2048,
        window_origin: WindowOrigin = "forward",
    ) -> dict[str, Any]:
        key = _finite_window_weight_key(
            window_name=window_name,
            window_length=window_length,
            epsilon=epsilon,
            time_unit=time_unit,
            nquad=nquad,
            window_origin=window_origin,
        )
        lookup = self.finite_window_delta_lookup
        metadata = self.interaction_cache_metadata()
        metadata.update(
            {
                "window_name": key[0],
                "window_length": key[1],
                "epsilon": key[2],
                "time_unit": key[3],
                "nquad": key[4],
                "interaction_count": self.interaction_arrays.count,
                "unique_delta_count": int(lookup.unique_delta.size),
                "delta_sha256": _array_sha256(self.interaction_arrays.delta),
                "delta_lookup_inverse_sha256": _array_sha256(lookup.inverse),
                "finite_transfer_convention": FINITE_TRANSFER_CONVENTION,
            }
        )
        if window_origin != "forward":
            metadata["window_origin"] = window_origin
        return metadata

    def save_finite_window_weights(
        self,
        path: str | Path,
        weights: CompactFiniteWindowWeights,
        *,
        epsilon: float,
        window_name: str,
        window_length: float,
        time_unit: WindowTimeUnit = "fast",
        nquad: int = 2048,
        window_origin: WindowOrigin = "forward",
    ) -> None:
        key = _finite_window_weight_key(
            window_name=window_name,
            window_length=window_length,
            epsilon=epsilon,
            time_unit=time_unit,
            nquad=nquad,
            window_origin=window_origin,
        )
        self._validate_precomputed_finite_window_weights(weights, key)
        _progress_log(
            "saving finite-window weight cache",
            path=str(path),
            weights=weights.count,
            unique_delta=weights.unique_count,
            storage=_nbytes_human(weights.nbytes),
            maxrss=_maxrss_human(),
        )
        start = time.perf_counter()
        weights.save_npz(
            path,
            self.finite_window_weight_cache_metadata(
                epsilon=epsilon,
                window_name=window_name,
                window_length=window_length,
                time_unit=time_unit,
                nquad=nquad,
                window_origin=window_origin,
            ),
        )
        _progress_log(
            "saved finite-window weight cache",
            path=str(path),
            elapsed_s=f"{time.perf_counter() - start:.1f}",
            file_size=_file_size_human(path),
            maxrss=_maxrss_human(),
        )

    def load_finite_window_weights(
        self,
        path: str | Path,
        *,
        epsilon: float,
        window_name: str,
        window_length: float,
        time_unit: WindowTimeUnit = "fast",
        nquad: int = 2048,
        window_origin: WindowOrigin = "forward",
    ) -> CompactFiniteWindowWeights:
        key = _finite_window_weight_key(
            window_name=window_name,
            window_length=window_length,
            epsilon=epsilon,
            time_unit=time_unit,
            nquad=nquad,
            window_origin=window_origin,
        )
        _progress_log("loading finite-window weight cache", path=str(path), file_size=_file_size_human(path))
        start = time.perf_counter()
        weights = CompactFiniteWindowWeights.load_npz(
            path,
            self.finite_window_weight_cache_metadata(
                epsilon=epsilon,
                window_name=window_name,
                window_length=window_length,
                time_unit=time_unit,
                nquad=nquad,
                window_origin=window_origin,
            ),
            source_id=id(self),
        )
        self._validate_precomputed_finite_window_weights(weights, key)
        self._finite_window_weight_cache[key] = weights
        _progress_log(
            "loaded finite-window weight cache",
            path=str(path),
            weights=weights.count,
            unique_delta=weights.unique_count,
            storage=_nbytes_human(weights.nbytes),
            elapsed_s=f"{time.perf_counter() - start:.1f}",
            maxrss=_maxrss_human(),
        )
        return weights

    def rhs_timing_payload(self, rng: np.random.Generator, *, scale: float = 0.02) -> Array:
        return scale * (rng.normal(size=len(self.modal_indices)) + 1j * rng.normal(size=len(self.modal_indices)))

    def rk4_step(self, amplitudes: Array, dt: float, rhs_kind: str, *, time: float, **rhs_kwargs: object) -> Array:
        if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(time):
            raise ValueError("dt must be positive and finite, and time must be finite")

        def f(a: Array, stage_time: float) -> Array:
            if rhs_kind == "exact":
                return self.exact_resonant_rhs(a, **rhs_kwargs)
            if rhs_kind == "finite":
                return self.finite_window_rhs(a, time=stage_time, **rhs_kwargs)
            raise ValueError(f"unknown rhs_kind {rhs_kind!r}")

        k1 = f(amplitudes, time)
        k2 = f(amplitudes + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = f(amplitudes + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = f(amplitudes + dt * k3, time + dt)
        return amplitudes + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _validate_precomputed_finite_window_weights(
        self, precomputed_weights: CompactFiniteWindowWeights, key: FiniteWindowWeightKey
    ) -> None:
        if precomputed_weights.key != key:
            raise ValueError("precomputed finite-window weights do not match requested window parameters")
        if precomputed_weights.source_id != id(self):
            raise ValueError("precomputed finite-window weights belong to a different CompactModalSystem")
        expected_shape = (self.interaction_arrays.count,)
        if precomputed_weights.weights.shape != expected_shape:
            raise ValueError(f"expected precomputed weight shape {expected_shape}, got {precomputed_weights.weights.shape}")

    def _weighted_rhs(self, amplitudes: Array, weights: Array | None = None, selected: Array | None = None) -> Array:
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        expected_shape = (len(self.modal_indices),)
        if amplitudes.shape != expected_shape:
            raise ValueError(f"expected amplitude shape {expected_shape}, got {amplitudes.shape}")

        interactions = self.interaction_arrays
        if weights is not None and weights.shape != (interactions.count,):
            raise ValueError(f"expected weight shape {(interactions.count,)}, got {weights.shape}")

        rhs = self.zero_amplitudes()
        if selected is None:
            for start in range(0, interactions.count, self.chunk_size):
                stop = min(start + self.chunk_size, interactions.count)
                values = (
                    interactions.coefficient[start:stop]
                    * amplitudes[interactions.a_idx[start:stop]]
                    * amplitudes[interactions.b_idx[start:stop]]
                )
                if weights is not None:
                    values = values * weights[start:stop]
                np.add.at(rhs, interactions.c_idx[start:stop], values)
        else:
            for start in range(0, selected.size, self.chunk_size):
                local = selected[start : start + self.chunk_size]
                values = (
                    interactions.coefficient[local]
                    * amplitudes[interactions.a_idx[local]]
                    * amplitudes[interactions.b_idx[local]]
                )
                if weights is not None:
                    values = values * weights[local]
                np.add.at(rhs, interactions.c_idx[local], values)
        return rhs


def _validate_epsilon(epsilon: float) -> None:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")


def _progress_log(message: str, **fields: object) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    if details:
        print(f"[rswfe-progress {stamp}] {message} {details}", flush=True)
    else:
        print(f"[rswfe-progress {stamp}] {message}", flush=True)


def _nbytes_human(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _file_size_human(path: str | Path) -> str:
    try:
        return _nbytes_human(Path(path).stat().st_size)
    except OSError:
        return "missing"


def _maxrss_human() -> str:
    try:
        import resource

        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return "unknown"
    # Linux reports ru_maxrss in KiB; macOS reports bytes. The cluster is Linux,
    # but this keeps local smoke output readable too.
    if maxrss > 1024.0 * 1024.0 * 1024.0:
        return _nbytes_human(int(maxrss))
    return _nbytes_human(int(maxrss * 1024.0))


def _finite_window_weight_key(
    *,
    window_name: str,
    window_length: float,
    epsilon: float,
    time_unit: WindowTimeUnit,
    nquad: int,
    window_origin: WindowOrigin = "forward",
) -> FiniteWindowWeightKey:
    if time_unit not in {"fast", "physical"}:
        raise ValueError("time_unit must be 'fast' or 'physical'")
    if nquad < 2:
        raise ValueError("nquad must be at least 2")
    validate_window_origin(window_name, window_origin)
    return (window_name, float(window_length), float(epsilon), time_unit, int(nquad), window_origin)


def _system_cache_metadata(grid: SpectralGrid, params: RSWParameters, resonance_tol: float) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "N": int(grid.N),
        "Lx": float(grid.Lx),
        "Ly": float(grid.Ly),
        "f": float(params.f),
        "g": float(params.g),
        "H": float(params.H),
        "nu": float(params.nu),
        "resonance_tol": float(resonance_tol),
        "branches": list(BRANCHES),
        "branch_policy": BRANCH_POLICY,
        "dealias_mask_convention": STRICT_DEALIAS_MASK_CONVENTION,
        "delta_convention": DELTA_CONVENTION,
    }


def _metadata_with_kind(metadata: dict[str, Any], kind: str) -> dict[str, Any]:
    out = dict(metadata)
    out["cache_kind"] = kind
    return out


def _load_metadata(data: Any) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError("cache file is missing metadata_json")
    raw = data["metadata_json"].tolist()
    if not isinstance(raw, str):
        raw = str(raw)
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("cache metadata is not a JSON object")
    return metadata


def _validate_cache_kind(metadata: dict[str, Any], expected_kind: str) -> None:
    actual = metadata.get("cache_kind")
    if actual != expected_kind:
        raise ValueError(f"incompatible cache kind: expected {expected_kind!r}, got {actual!r}")


def _validate_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key, None)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    if mismatches:
        details = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            details += f"; ... {len(mismatches) - 8} more"
        raise ValueError(f"incompatible compact modal cache metadata ({details})")


def _validate_interaction_arrays_shapes(arrays: CompactInteractionArrays) -> None:
    shape = arrays.c_idx.shape
    if arrays.a_idx.shape != shape or arrays.b_idx.shape != shape or arrays.coefficient.shape != shape or arrays.delta.shape != shape:
        raise ValueError("interaction cache arrays do not share a common one-dimensional shape")
    if len(shape) != 1:
        raise ValueError("interaction cache arrays must be one-dimensional")


def _validate_finite_window_weight_shapes(weights: CompactFiniteWindowWeights) -> None:
    if weights.weights.ndim != 1:
        raise ValueError("finite-window weights must be one-dimensional")
    if weights.unique_delta.shape != weights.unique_weights.shape or weights.unique_delta.ndim != 1:
        raise ValueError("finite-window unique delta and unique weights must share a one-dimensional shape")


def _array_sha256(array: Array) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _finite_window_key_to_jsonable(key: FiniteWindowWeightKey) -> list[Any]:
    values = [key[0], key[1], key[2], key[3], key[4]]
    return values if key[5] == "forward" else values + [key[5]]


def _finite_window_key_from_jsonable(raw: list[Any]) -> FiniteWindowWeightKey:
    if len(raw) not in (5, 6):
        raise ValueError("finite-window key must have five legacy entries or six explicit-origin entries")
    return _finite_window_weight_key(
        window_name=str(raw[0]),
        window_length=float(raw[1]),
        epsilon=float(raw[2]),
        time_unit=raw[3],
        nquad=int(raw[4]),
        window_origin=raw[5] if len(raw) == 6 else "forward",
    )
