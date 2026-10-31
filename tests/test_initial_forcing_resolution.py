"""Check the initial-forcing resolution workflow and its failure detection."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generate_data"))
import check_rsw_initial_forcing_resolution as check


class InitialForcingResolutionChecks(unittest.TestCase):
    def test_initial_support_and_grid_independent_forcing(self):
        report = check.evaluate_resolution([32, 64, 128])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["quadratic_output_max_abs"], [8, 4])
        self.assertTrue(all(r["input_modal_count"] == 8 for r in report["results"].values()))
        self.assertLess(report["max_forcing_relative_change"], check.GRID_RELATIVE_TOL)
        self.assertLess(report["max_main_archive_absolute_difference"], check.ARCHIVE_ABSOLUTE_TOL)
        self.assertEqual(report["archive_reported_only_groups"], ["exact"])

    def test_grid_that_discards_quadratic_outputs_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "N=16 does not retain every quadratic output"):
            check.evaluate_resolution([16, 64])

    def test_requires_a_cross_grid_comparison(self):
        with self.assertRaisesRegex(ValueError, "at least two distinct grid sizes"):
            check.evaluate_resolution([64])

    def test_nonfinite_archived_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            for kernel in ("boxcar", "bump"):
                source = ROOT / f"data/fig8_forcing_{kernel}.csv"
                with source.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    names = reader.fieldnames
                    rows = list(reader)
                if kernel == "bump":
                    rows[-1][check.GROUP_COLUMNS["sideband"]] = "nan"
                with (destination / source.name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=names)
                    writer.writeheader()
                    writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Nonfinite archived forcing ratio"):
                check.evaluate_resolution([32, 64], forcing_dir=destination)

    def test_changed_archived_curve_causes_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            for kernel in ("boxcar", "bump"):
                source = ROOT / f"data/fig8_forcing_{kernel}.csv"
                with source.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    names = reader.fieldnames
                    rows = list(reader)
                if kernel == "boxcar":
                    key = check.GROUP_COLUMNS["sideband"]
                    rows[0][key] = str(float(rows[0][key]) + 0.02)
                with (destination / source.name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=names)
                    writer.writeheader()
                    writer.writerows(rows)
            report = check.evaluate_resolution([32, 64], forcing_dir=destination)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["main_archived_curves"])


if __name__ == "__main__":
    unittest.main()
