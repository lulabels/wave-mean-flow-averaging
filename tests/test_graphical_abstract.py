from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib"))
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import label, map_coordinates
from wave_averaging.graphical_abstract import (
    colourize, compose, construct_states, load_fields, regenerate_fields,
)
from wave_averaging.rsw.modal_reduced import ModalSystem


class GraphicalAbstractCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fields = load_fields()
        cls.composition = compose(cls.fields)

    def test_full_generation_recovers_archived_crops(self):
        regenerated = regenerate_fields()
        for key in ("lab_crop", "frame_crop"):
            np.testing.assert_allclose(regenerated[key], self.fields[key], rtol=0, atol=2e-7)
        self.assertEqual(float(regenerated["scale"]), float(self.fields["scale"]))

    def test_generation_commands_work_without_cached_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = Path(directory) / "code"
            shutil.copytree(ROOT / "wave_averaging", packet / "wave_averaging",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (packet / "data").mkdir()
            shutil.copyfile(ROOT / "data/graphical_abstract_reference.json",
                            packet / "data/graphical_abstract_reference.json")
            for relative in ("generate_data/build_graphical_abstract_data.py",
                             "figures/graphical_abstract.py"):
                target = packet / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            cached = packet / "data/graphical_abstract_fields.npz"
            self.assertFalse(cached.exists())

            def run(relative, *arguments):
                result = subprocess.run([sys.executable, "-I", str(packet / relative), *arguments],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run("generate_data/build_graphical_abstract_data.py")
            generated = packet / "work/generated_data/graphical_abstract_fields.npz"
            with np.load(generated, allow_pickle=False) as data:
                for key in self.fields:
                    np.testing.assert_array_equal(data[key], self.fields[key])
            run("figures/graphical_abstract.py", "--data", str(generated), "--verify")
            run("figures/graphical_abstract.py", "--regenerate-fields", "--verify")
            self.assertFalse(cached.exists())

    def test_boxcar_state_matches_independent_modal_average(self):
        data = construct_states(24)
        system = ModalSystem(data["grid"], data["params"])
        initial = data["frame"]
        amplitudes = system.state_to_modal(initial)
        multiplier = np.sinc(system.omega * data["cfg"].window_length / (2*np.pi))
        modal_mean = system.modal_to_state(amplitudes * multiplier)
        np.testing.assert_allclose(data["lab"].as_array(), modal_mean.as_array(), rtol=0, atol=2e-14)

    def test_contour_matches_both_unchanged_fields(self):
        c = self.composition
        residual = map_coordinates(c["frame"]-c["lab"], [c["seam"][:,1], c["seam"][:,0]],
                                   order=1, mode="nearest")
        self.assertLess(float(np.max(np.abs(residual))), 3e-12)
        mask = c["mean_side"]
        np.testing.assert_array_equal(c["joined"][mask], c["lab"][mask])
        np.testing.assert_array_equal(c["joined"][~mask], c["frame"][~mask])
        self.assertTrue(mask[:,0].all())
        self.assertFalse(mask[:,-1].any())
        self.assertEqual(label(mask)[1], 1)
        self.assertEqual(label(~mask)[1], 1)

    def test_preview_pixels_match_reference(self):
        rgb = colourize(self.composition["joined"], float(self.fields["scale"]))
        expected = json.loads(str(self.fields["metadata_json"]))["reference_rgb_sha256"]
        self.assertEqual(rgb.shape, (900,1080,3))
        self.assertEqual(hashlib.sha256(rgb.tobytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
