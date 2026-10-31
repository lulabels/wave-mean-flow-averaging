"""Check that a failed data download preserves a previously cached archive."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataDownload(unittest.TestCase):
    def test_failed_download_does_not_replace_previous_archive(self):
        download = load_module("packet_download", "generate_data/download_baker_data.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "work/downloads/baker_zenodo_data.zip"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"previous archive")
            from io import BytesIO
            with patch.object(download, "ROOT", root), patch.object(
                download.urllib.request, "urlopen", return_value=BytesIO(b"incomplete download")
            ):
                with self.assertRaisesRegex(ValueError, "md5 mismatch"):
                    download.prepare_input(archive=None, output_dir=root / "data", force_download=True)
            self.assertEqual(cache.read_bytes(), b"previous archive")
            self.assertFalse((root / "data" / Path(download.MEMBER).name).exists())


if __name__ == "__main__":
    unittest.main()
