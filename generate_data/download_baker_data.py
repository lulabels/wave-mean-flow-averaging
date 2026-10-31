#!/usr/bin/env python3
"""Download and verify the Baker et al. NetCDF input required by Figure 9."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/api/records/14237745/files/data.zip/content"
ARCHIVE_MD5 = "1bd4abf8c6f4e60d345216bcceb527b2"
ARCHIVE_SHA256 = "879c7d008607f6c3e9b96449a714c2a20d04dabfe68c4af3c2bc4935d48189b9"
MEMBER = "data/solver_lowpass_x_y_omega_2_Nint_1_strat_3_T_40.nc"
MEMBER_SHA256 = "3519bbfb89a4893f6605458b43e38f5bafed3537cc9900b6e8742a25ba2f4266"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(path: Path, algorithm: str, expected: str) -> None:
    actual = digest(path, algorithm)
    if actual != expected:
        raise ValueError(f"{path}: {algorithm} mismatch: {actual} != {expected}")


def prepare_input(*, archive: Path | None, output_dir: Path, force_download: bool = False) -> Path:
    output_dir = output_dir.resolve()
    output = output_dir / Path(MEMBER).name
    if archive is None and output.exists() and not force_download:
        verify(output, "sha256", MEMBER_SHA256)
        print(f"verified existing {output}")
        return output
    download_dir = ROOT / "work" / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    supplied_archive = archive is not None
    archive = archive.resolve() if supplied_archive else download_dir / "baker_zenodo_data.zip"
    if not supplied_archive and (force_download or not archive.exists()):
        temporary = archive.with_suffix(".zip.part")
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=30) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        verify(temporary, "md5", ARCHIVE_MD5)
        verify(temporary, "sha256", ARCHIVE_SHA256)
        temporary.replace(archive)

    verify(archive, "md5", ARCHIVE_MD5)
    verify(archive, "sha256", ARCHIVE_SHA256)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(".nc.part")
    with zipfile.ZipFile(archive) as bundle, bundle.open(MEMBER) as source, temporary_output.open("wb") as target:
        shutil.copyfileobj(source, target)
    verify(temporary_output, "sha256", MEMBER_SHA256)
    temporary_output.replace(output)
    print(f"wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="use an already downloaded Zenodo data.zip")
    source.add_argument("--force-download", action="store_true", help="replace cached download after verification")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "external" / "baker")
    args = parser.parse_args()
    prepare_input(archive=args.archive, output_dir=args.output_dir, force_download=args.force_download)


if __name__ == "__main__":
    main()
