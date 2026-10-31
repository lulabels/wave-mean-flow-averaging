"""Construct the fields and render the graphical abstract.

The starting height is an analytic rolled ribbon with an offset arm. Its
geostrophic velocities define a balanced RSW state b. A separately prescribed
height packet, with zero seed velocities, is projected onto the wave subspace
to give q. The initial state at reference time zero is b + q.

For the linear flow from this state, a centered boxcar of length T=4*pi gives
    lab mean:        b + sinc(omega*T/2) q, applied mode by mode;
    wave-frame mean: b + q, after removing the linear evolution before averaging.
Here sinc(z)=sin(z)/z and omega=sqrt(f**2 + g*H*|k|**2). The Fourier calculation
uses no time stepping. configuration(768) gives the active 9.6 x 8 periodic
domain with f=g=H=1; the ribbon and wave constructors prescribe the geometry.

regenerate_fields() takes a central 6 x 5 crop of each mean's height and stores
the two float32 arrays with one scale. compose() resizes them to 1080 x 900 and
selects a complete top-to-bottom zero contour of frame_height - lab_height.
The lab mean supplies the left side, the wave-frame mean the right. colourize()
uses the same scale and palette throughout. The scalar values agree on the
curve without blending. The joined image compares the lab-frame and wave-frame
height averages of the prescribed linear flow.

build_inputs() calculates the fields and adds the reference metadata/profile.
load_fields() is the fast route through the included compact NPZ. Both feed
the same compose(), colourize() and render() functions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import base64
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .rsw.projectors import wave_project_state
from .rsw.solver_pseudospectral import (
    RSWParameters, RSWState, SpectralGrid, dealias_state, deriv_x, deriv_y,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/graphical_abstract_fields.npz"
DEFAULT_REFERENCE = ROOT / "data/graphical_abstract_reference.json"
PREVIEW_SIZE = (1080, 900)
MASTER_SIZE = (3600, 3000)
SCALE = 0.19407985027468144
STOPS = np.array([-1.0, -0.58, -0.18, 0.0, 0.25, 0.55, 0.80, 1.0])
BLUE_COLOURS = [
    [91,193,222], [18,91,143], [4,29,59], [2,11,26],
    [9,53,99], [51,132,187], [202,235,235], [255,158,112],
]

@dataclass(frozen=True)
class Configuration:
    N: int = 192
    Lx: float = 8.0
    Ly: float = 20.0 / 3.0
    f: float = 1.0
    g: float = 1.0
    H: float = 1.0
    centre_x: float = 4.0
    centre_y: float = 3.12
    envelope_radius: float = 2.35
    balanced_height: float = 0.115
    wave_height: float = 0.050
    window_length: float = 2.0 * np.pi
    render_width: float = 6.0
    render_height: float = 5.0
    output_width: int = 1800
    output_height: int = 1500


def periodic_displacement(values: np.ndarray, centre: float, length: float) -> np.ndarray:
    return (values - centre + 0.5 * length) % length - 0.5 * length


def balanced_state_from_height(
    eta: np.ndarray,
    grid: SpectralGrid,
    params: RSWParameters,
) -> RSWState:
    """Set u=-(g/f)*eta_y and v=(g/f)*eta_x for the prescribed height."""
    eta = eta - float(np.mean(eta))
    return dealias_state(
        RSWState(
            -(params.g / params.f) * deriv_y(eta, grid),
            +(params.g / params.f) * deriv_x(eta, grid),
            eta,
        ),
        grid,
    )


def split_field(
    left: np.ndarray,
    right: np.ndarray,
    grid: SpectralGrid,
    cfg: Configuration,
) -> np.ndarray:
    combined = np.where(grid.X < cfg.centre_x, left, right)
    dx = periodic_displacement(grid.x, cfg.centre_x, cfg.Lx)
    dy = periodic_displacement(grid.y, cfg.centre_y, cfg.Ly)
    ix = np.flatnonzero(np.abs(dx) <= 0.5 * cfg.render_width + 1.0e-12)
    iy = np.flatnonzero(np.abs(dy) <= 0.5 * cfg.render_height + 1.0e-12)
    return combined[np.ix_(ix, iy)].T


@dataclass(frozen=True)
class SketchParameters:
    ribbon_width: float = 0.48
    ribbon_slope: float = 0.05
    vortex_radius: float = 1.22
    vortex_power: float = 4.0
    central_turn: float = 7.35
    ribbon_height: float = 0.115
    wave_height: float = 0.100


def ribbon_height(
    grid: SpectralGrid,
    cfg: Configuration,
    sketch: SketchParameters,
) -> np.ndarray:
    """Return the prescribed compact rolled-ribbon eta geometry."""
    x = periodic_displacement(grid.X, cfg.centre_x, cfg.Lx)
    y = periodic_displacement(grid.Y, cfg.centre_y, cfg.Ly)
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    angular_displacement = sketch.central_turn / (
        1.0 + (radius / sketch.vortex_radius) ** sketch.vortex_power
    )
    theta_material = theta - angular_displacement
    xi = radius * np.cos(theta_material)
    eta_material = radius * np.sin(theta_material)

    # A broad core and weaker shoulder form the ribbon. The radial gate
    # localizes it inside the periodic domain.
    transverse = (eta_material - sketch.ribbon_slope * xi) / sketch.ribbon_width
    core = np.exp(-0.5 * transverse**2)
    shoulder = np.exp(-0.5 * (transverse / 1.85) ** 2)
    ribbon = 1.58 * core + 0.34 * shoulder
    gate = np.exp(-((radius / 3.55) ** 12))
    return sketch.ribbon_height * gate * ribbon


def wave_seed_height(
    name: str,
    grid: SpectralGrid,
    cfg: Configuration,
    sketch: SketchParameters,
    sinuous_strength: float = 2.50,
    sinuous_frequency: float = 1.75,
) -> np.ndarray:
    """Prescribe an independent, unwound wave phase for branch projection."""
    x = periodic_displacement(grid.X, cfg.centre_x, cfg.Lx)
    y = periodic_displacement(grid.Y, cfg.centre_y, cfg.Ly)
    radius = np.hypot(x, y)

    if name != "sinuous_fronts":
        raise ValueError(f"unsupported wave seed {name!r}")
    angle = -0.62
    along = np.cos(angle) * x + np.sin(angle) * y
    across = -np.sin(angle) * x + np.cos(angle) * y
    phase = 5.85 * along + sinuous_strength * np.sin(
        sinuous_frequency * across - 0.30
    )
    envelope = np.exp(-((radius / 3.25) ** 10))

    return sketch.wave_height * envelope * np.cos(phase)


def material_coordinates(grid, cfg, sketch):
    """Unwind the prescribed spiral to evaluate its ribbon profiles."""
    x = periodic_displacement(grid.X, cfg.centre_x, cfg.Lx)
    y = periodic_displacement(grid.Y, cfg.centre_y, cfg.Ly)
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    displacement = sketch.central_turn / (
        1.0 + (radius / sketch.vortex_radius) ** sketch.vortex_power
    )
    theta_material = theta - displacement
    xi = radius * np.cos(theta_material)
    eta_material = radius * np.sin(theta_material)
    return x, y, radius, xi, eta_material


def secondary_arm_height(grid, cfg, sketch, amplitude):
    """A narrow, offset, finite material-ribbon segment."""
    _, _, radius, xi, eta_material = material_coordinates(grid, cfg, sketch)
    centreline = sketch.ribbon_slope * xi + 1.65
    transverse = (eta_material - centreline) / 0.23
    profile = np.exp(-0.5 * transverse**2)
    profile += 0.14 * np.exp(-0.5 * (transverse / 1.65) ** 2)

    # The segment is chosen in material coordinates, so differential rotation
    # makes one asymmetric arm rather than a complete second spiral.  The inner
    # cutoff protects the original core as the sole focal point.
    along_gate = np.exp(-0.5 * ((xi - 0.90) / 0.55) ** 4)
    inner_cutoff = 1.0 - np.exp(-((radius / 1.20) ** 8))
    outer_gate = np.exp(-((radius / 3.55) ** 12))
    return amplitude * profile * along_gate * inner_cutoff * outer_gate


def refined_balanced_height(grid, cfg, sketch, arm_amplitude):
    main = ribbon_height(grid, cfg, sketch)
    arm = secondary_arm_height(grid, cfg, sketch, arm_amplitude)
    return main + arm, arm


def localized_fine_seed(grid, cfg, strength):
    """Prescribe a localized packet with finer, curved wave crests."""
    x = periodic_displacement(grid.X, cfg.centre_x, cfg.Lx)
    y = periodic_displacement(grid.Y, cfg.centre_y, cfg.Ly)
    angle = -0.62
    along = np.cos(angle) * x + np.sin(angle) * y
    across = -np.sin(angle) * x + np.cos(angle) * y
    phase = 16.5 * along + 4.2 * np.sin(2.15 * across - 0.3)
    phase += 0.85 * np.sin(4.3 * across + 0.6)

    radial = np.exp(-((np.hypot(x, y) / 3.25) ** 10))
    packet = np.exp(
        -0.5 * (((x - 0.85) / 1.65) ** 2 + ((y - 0.35) / 2.5) ** 2)
    )
    # A broad focus in the phase-normal direction leaves continuous weak wave
    # support but makes only a central group of crests strongly luminous.
    crest_focus = 0.34 + 0.66 * np.exp(-0.5 * ((along - 0.10) / 1.35) ** 2)
    return strength * radial * packet * crest_focus * np.cos(phase)


def wave_state(grid, cfg, params, sketch):
    """Combine broad and fine height packets, then remove their balanced part."""
    broad = wave_seed_height(
        "sinuous_fronts", grid, cfg, replace(sketch, wave_height=0.045)
    )
    seed_eta = broad + localized_fine_seed(grid, cfg, 0.100)
    seed = dealias_state(
        RSWState(
            np.zeros_like(seed_eta), np.zeros_like(seed_eta), seed_eta
        ),
        grid,
    )
    wave = dealias_state(
        wave_project_state(seed, grid, params), grid
    )
    return wave, seed_eta


def configuration(N: int):
    """Active geometry/window; the data builder uses N=768."""
    return replace(
        Configuration(),
        N=N,
        Lx=9.6,
        Ly=8.0,
        centre_x=4.8,
        centre_y=3.62,
        window_length=4.0 * np.pi,
        render_width=6.0,
        render_height=5.0,
        output_width=MASTER_SIZE[0],
        output_height=MASTER_SIZE[1],
    )


def add_states(a, b):
    return RSWState(a.u + b.u, a.v + b.v, a.eta + b.eta)


def sinc_filter_wave(wave, grid, params, window_length):
    """Apply the exact centered boxcar to a wave-branch state.

    The two Poincare frequencies are opposite in sign and sinc is even, so a
    single real Fourier multiplier applies to their sum in every component.
    """
    omega = np.sqrt(params.f**2 + params.g * params.H * grid.K2)
    multiplier = np.sinc(omega * window_length / (2.0 * np.pi))
    mask = grid.dealias_mask

    def filtered(component):
        return np.fft.ifft2(
            np.fft.fft2(component) * multiplier * mask
        ).real

    return RSWState(
        filtered(wave.u), filtered(wave.v), filtered(wave.eta)
    )


def construct_states(N: int):
    """Construct b, q and both centered linear means at reference time zero.

    The spiral and secondary arm are prescribed in the balanced height.
    wave_state() supplies an independent wave-branch field. Only that field
    is attenuated in the lab mean; the wave-frame mean retains b + q.
    """
    cfg = configuration(N)
    grid = SpectralGrid(N, Lx=cfg.Lx, Ly=cfg.Ly)
    params = RSWParameters(f=cfg.f, g=cfg.g, H=cfg.H, nu=0.0)
    sketch = SketchParameters()
    eta_balanced, arm_eta = refined_balanced_height(
        grid, cfg, sketch, 0.120
    )
    balanced = balanced_state_from_height(
        eta_balanced, grid, params
    )
    wave, seed_eta = wave_state(grid, cfg, params, sketch)
    averaged_wave = sinc_filter_wave(
        wave, grid, params, cfg.window_length
    )
    lab = add_states(balanced, averaged_wave)
    frame = add_states(balanced, wave)
    return {
        "cfg": cfg,
        "grid": grid,
        "params": params,
        "balanced": balanced,
        "wave": wave,
        "averaged_wave": averaged_wave,
        "lab": lab,
        "frame": frame,
        "seed_eta": seed_eta,
        "arm_eta": arm_eta,
    }


def regenerate_fields(N: int = 768) -> dict[str, np.ndarray]:
    """Calculate both means, then crop their heights; no saved fields are read."""
    if N < 12 or N % 2:
        raise ValueError("N must be even and at least 12")
    data = construct_states(N)
    return {
        "lab_crop": split_field(data["lab"].eta, data["lab"].eta,
                                data["grid"], data["cfg"]).astype(np.float32),
        "frame_crop": split_field(data["frame"].eta, data["frame"].eta,
                                  data["grid"], data["cfg"]).astype(np.float32),
        "scale": np.asarray(SCALE),
    }


def build_inputs() -> dict[str, np.ndarray]:
    """Build the N768 inputs without reading the cached field archive.

    The reference JSON supplies only export metadata and the colour profile.
    Both scalar fields are calculated from the initial condition above.
    """
    fields = regenerate_fields()
    reference = json.loads(DEFAULT_REFERENCE.read_text())
    profile = base64.b64decode(reference["srgb_profile_base64"], validate=True)
    fields["srgb_profile"] = np.frombuffer(profile, dtype=np.uint8).copy()
    fields["metadata_json"] = np.asarray(json.dumps(reference["metadata"], sort_keys=True))
    return fields


def load_fields(path: Path = DEFAULT_DATA) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}
    if fields["lab_crop"].shape != fields["frame_crop"].shape:
        raise ValueError("the two scalar crops must share one grid")
    for key in ("lab_crop", "frame_crop"):
        if fields[key].ndim != 2 or not np.all(np.isfinite(fields[key])):
            raise ValueError(f"invalid scalar input: {key}")
    if not np.isfinite(float(fields["scale"])) or float(fields["scale"]) <= 0:
        raise ValueError("the shared scalar scale must be positive and finite")
    return fields


def compose(fields: dict[str, np.ndarray], size=PREVIEW_SIZE) -> dict:
    """Resize both heights and join lab-left/frame-right where they agree."""
    width, height = size
    if width < 2 or height < 2 or 5 * width != 6 * height:
        raise ValueError("image dimensions must have aspect ratio 1.2:1")
    lab = np.asarray(Image.fromarray(fields["lab_crop"]).resize(
        size, Image.Resampling.BICUBIC))
    frame = np.asarray(Image.fromarray(fields["frame_crop"]).resize(
        size, Image.Resampling.BICUBIC))
    delta = frame - lab
    fig, ax = plt.subplots()
    try:
        contours = ax.contour(delta, levels=[0.0])
        candidates = []
        for path in contours.allsegs[0]:
            if len(path) < 2:
                continue
            top_bottom = abs(path[0,1]) < 1e-4 and abs(path[-1,1]-(height-1)) < 1e-4
            bottom_top = abs(path[-1,1]) < 1e-4 and abs(path[0,1]-(height-1)) < 1e-4
            if not (top_bottom or bottom_top):
                continue
            if bottom_top:
                path = path[::-1]
            length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
            score = float(np.mean(((path[:,0]-width/2)/width)**2)+0.015*length/height)
            candidates.append((score, path))
    finally:
        plt.close(fig)
    if not candidates:
        raise RuntimeError("no complete top-to-bottom equality contour; no image written")
    # Prefer a complete contour near the centre, with a small length penalty.
    candidates.sort(key=lambda item: item[0])
    score, seam = candidates[0]
    polygon = np.vstack((seam, [-1,height], [-1,-1], seam[0]))
    yy, xx = np.indices(delta.shape)
    mean_side = PolygonPath(polygon).contains_points(
        np.column_stack((xx.ravel(), yy.ravel()))).reshape(delta.shape)
    joined = np.where(mean_side, lab, frame)
    residual = map_coordinates(delta, [seam[:,1], seam[:,0]], order=1, mode="nearest")
    return {
        "lab": lab, "frame": frame, "joined": joined, "seam": seam,
        "mean_side": mean_side, "scale": np.asarray(fields["scale"]),
        "diagnostics": {
            "candidate_count": len(candidates), "seam_score": score,
            "seam_vertex_count": len(seam),
            "seam_difference_max_abs": float(np.max(np.abs(residual))),
            "seam_difference_rms": float(np.sqrt(np.mean(residual**2))),
            "mean_side_fraction": float(np.mean(mean_side)),
            "size": list(size),
            "scope": "two unchanged scalar averages; equality-contour join; no blending",
        },
    }


def colourize(field, scale):
    """Use the same fixed scale, tanh contrast and palette on both sides."""
    values = np.tanh(field / (0.78 * scale))
    rgb = np.stack([
        np.interp(values, STOPS, np.asarray(BLUE_COLOURS)[:, channel])
        for channel in range(3)
    ], axis=-1)
    return np.round(rgb).astype(np.uint8)


def render(fields, output_dir: Path, *, size=PREVIEW_SIZE, verify=False) -> dict:
    composition = compose(fields, size)
    rgb = colourize(composition["joined"], float(fields["scale"]))
    rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
    metadata = json.loads(str(fields["metadata_json"]))
    if verify:
        if tuple(size) != PREVIEW_SIZE:
            raise ValueError("reference-image verification applies to the 1080 x 900 image")
        if rgb_hash != metadata["reference_rgb_sha256"]:
            raise ValueError("graphical abstract does not match the reference RGB pixels")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = fields["srgb_profile"].astype(np.uint8).tobytes()
    image = Image.fromarray(rgb)
    image.save(output_dir / "graphical_abstract.png", dpi=(300,300), icc_profile=profile)
    image.save(output_dir / "graphical_abstract.jpg", dpi=(300,300), icc_profile=profile,
               quality=98, subsampling=0, optimize=True)
    image.resize((240,200), Image.Resampling.LANCZOS).save(
        output_dir / "graphical_abstract_thumb_240x200.png", icc_profile=profile)
    record = composition["diagnostics"]
    record["rgb_sha256"] = rgb_hash
    record["reference_rgb_match"] = rgb_hash == metadata["reference_rgb_sha256"]
    (output_dir / "graphical_abstract_diagnostics.json").write_text(json.dumps(record, indent=2)+"\n")
    np.savez_compressed(output_dir / "graphical_abstract_composition.npz",
                        **{key: value for key, value in composition.items() if key != "diagnostics"})
    return record
