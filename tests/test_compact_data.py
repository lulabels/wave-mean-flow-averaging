from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wave_averaging.rsw.projectors import projector_sanity
from wave_averaging.rsw.solver_pseudospectral import RSWParameters, RSWState, SpectralGrid


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class CompactDataCheckpoints(unittest.TestCase):
    def test_compact_input_hashes(self) -> None:
        manifest = json.loads((ROOT / "data" / "checksums.json").read_text())
        for relative, expected in manifest["figure_inputs"].items():
            self.assertEqual(sha256(ROOT / "data" / relative), expected, relative)

    def test_figure2_projection_split(self) -> None:
        with np.load(ROOT / "data" / "fig2_rsw_fields.npz", allow_pickle=False) as data:
            slow = np.asarray(data["eta_slow"])
            full = np.asarray(data["eta_full"])
            wave = np.asarray(data["eta_wave"])
        self.assertLess(float(np.max(np.abs(full - slow - wave))), 1.0e-15)
        self.assertAlmostEqual(float(np.max(np.abs(slow))), 0.08032694116889648, places=14)
        self.assertAlmostEqual(float(np.max(np.abs(full))), 0.11324508487663468, places=14)
        self.assertAlmostEqual(float(np.max(np.abs(wave))), 0.050567302554061236, places=14)

    def test_figure8_centered_retention(self) -> None:
        with np.load(ROOT / "data" / "fig7_8_rsw_products.npz", allow_pickle=False) as data:
            config = json.loads(str(data["config_json"]))
            retention = json.loads(str(data["retention_json"]))
        self.assertEqual(config["N"], 1024)
        self.assertEqual(config["params"]["nu"], 0.0)
        self.assertEqual(config["dt"], 0.00078125)
        self.assertEqual(config["analysis_cadence"], 0.00625)
        self.assertEqual(config["quadrature"], "trapezoid")
        self.assertAlmostEqual(retention["boxcar"]["2"]["lab_frame"], 0.1505336313671308, places=13)
        self.assertAlmostEqual(retention["boxcar"]["2"]["wave_frame"], 0.9697759463014903, places=13)
        self.assertAlmostEqual(retention["bump"]["2"]["lab_frame"], 0.2165644646763117, places=13)
        self.assertAlmostEqual(retention["bump"]["2"]["wave_frame"], 0.978346659185213, places=13)

    def test_figure8_forcing_is_near_resonant(self) -> None:
        expected = {
            "boxcar": (0.9303747467951753, 6.337770282744902e-09),
            "bump": (0.9269159312911793, 6.304319464217215e-09),
        }
        for name, (near_expected, exact_expected) in expected.items():
            with (ROOT / "data" / f"fig8_forcing_{name}.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            row = next(item for item in rows if float(item["window_length"]) == 1.0)
            near = float(row["rhs_finite_sideband_same_branch_slow_wave_to_wave_weighted_l2_fraction_of_finite_total"])
            exact = float(row["rhs_exact_strict_unweighted_l2_fraction_of_finite_total"])
            self.assertAlmostEqual(near, near_expected, places=13)
            self.assertAlmostEqual(exact, exact_expected, places=20)

    def test_figure9_projection(self) -> None:
        with np.load(ROOT / "data" / "fig9_projection_N256.npz", allow_pickle=False) as data:
            total = np.asarray(data["instant_zeta"])
            mean = np.asarray(data["numerical_mean_zeta"])
            wave = np.asarray(data["numerical_wave_zeta"])
            metadata = json.loads(str(data["metadata_json"]))
        self.assertLess(float(np.max(np.abs(total - mean - wave))), 7.0e-16)
        self.assertAlmostEqual(metadata["baker_z_inst_relative_l2"], 2.437357142040751e-14, places=25)
        self.assertLess(metadata["numeric_vs_analytic_mean_relative_l2"], 6.0e-16)
        self.assertLess(metadata["numeric_vs_analytic_wave_relative_l2"], 4.0e-16)

    def test_small_rsw_projector(self) -> None:
        grid = SpectralGrid(8)
        params = RSWParameters()
        rng = np.random.default_rng(1729)
        state = RSWState.from_array(rng.standard_normal((3, 8, 8)))
        checks = projector_sanity(state, grid, params)
        self.assertLess(checks["P0_idempotence_error"], 5.0e-16)
        self.assertLess(abs(checks["P0_energy_orthogonality"]), 5.0e-15)


if __name__ == "__main__":
    unittest.main()
