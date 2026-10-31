# Verification of calculus and algebra

These scripts use SymPy to check the identities listed below from “A unifying view of averaging for wave–mean flow decomposition”. Each script describes its calculation and assumptions, treating zero frequency and exact resonance separately where needed. They run independently of the figure code and data.

The numerical tests in [`tests/`](../tests/) check the implemented solvers, projectors, averaging routines and saved inputs.

## Run

With Python 3.13.5, from the code repository's root directory:

```bash
python -m pip install -r verification/requirements.txt
python -m unittest discover -s verification -p 'check_*.py' -v
```

SymPy is an optional dependency; the figure-reproduction environment does not require it. After installation, the checks work offline. Each script can also run separately, for example:

```bash
python verification/check_drift.py
```

A failed identity or an expression that cannot be reduced to the stated target causes a nonzero exit status.

## Calculations

| Script | Equations | Calculation and extent |
|---|---|---|
| [check_classical_examples.py](check_classical_examples.py) | (1.2)–(1.4), (4.1)–(4.6), (4.16) | Frozen-amplitude near-inertial averages; displacement, Taylor coefficients and Stokes drift using the stated linear surface-wave fields; generic first-order vector Taylor coefficient. |
| [check_filtering.py](check_filtering.py) | (3.5)–(3.8), (B1)–(B2), (B4) | Signed-frequency boxcar integration, zero-frequency limit, repeated averaging, complete-period exception, finite Fourier-mode linearity and material-trajectory chain rule. |
| [check_kernel_responses.py](check_kernel_responses.py) | (3.1), (3.4), (3.9); Gaussian/exponential responses in §3 | General kernel change of variable under stated integrability assumptions; directly integrated Gaussian and causal-exponential responses, magnitudes and scalar limits. |
| [check_drift.py](check_drift.py) | (4.8b)–(4.15), (4.18)–(4.23) | Hydrostatic symbol/projectors, wave-space inverse and velocity reconstruction; symbolic mode displacement, reality, spatial pair coefficients and resonant/nonresonant drift weights. |
| [check_wave_frame.py](check_wave_frame.py) | (2.3), (2.9), (3.10)–(3.11), (C1)–(C2), (C4)–(C6) | Fast-term cancellation for arbitrary finite modal systems; general scalar triad phase and boxcar detuning weight retaining the current phase. |
| [check_reference_frames.py](check_reference_frames.py) | (5.4), (5.8a,b), (5.11), (5.14)–(5.16), (B7) | Modal kernel/generator identities, arbitrary smooth scalars under steady shear, uniform-flow Doppler relation and balanced-mode frequency weights with symbolic parameters. |
| [check_spectral_algebra.py](check_spectral_algebra.py) | (2.4a), (A2)–(A4), (A6), (A9), (A11)–(A13) | Pressure/Leray identities, Fourier projection and restricted finite-symbol skewness; scalar one-sided averaging multiplier and its zero/nonzero-frequency limits. |

The table and script comments refer to equations in the accompanying paper. A reference may identify an input or definition as well as a checked identity.

The conventions match the paper: `sinc(x) = sin(x)/x`, with value one at zero; the fast phase is `tau = t/epsilon`; displacement has zero exact fast mean and includes the physical factor `epsilon`; and finite-window triad weights retain the phase at the current `tau`. The hydrostatic calculation uses the density-variable state `(u, v, rho)` and excludes zero vertical wavenumber. Further assumptions are stated in each script.

## Scope

The checks cover the listed symbolic identities under their stated assumptions. Modal and pairwise identities extend to finite sums. The exact-average limit is checked mode by mode for the boxcar; kernel substitutions assume integrability, with normalization required to preserve constants.
