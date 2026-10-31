#!/usr/bin/env python3
"""Export Figure 1 from its editable schematic, figure1.svg.

    python figures/figure1.py

Requires rsvg-convert on PATH, separately from the Python requirements.
Writes outputs/fig1.png at 5000 pixels wide and outputs/fig1.pdf. The SVG
contains the embedded sheet artwork, arrows and editable text labels; no
simulation or external image files are needed. Tested with rsvg-convert 2.62.2.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--svg", type=Path, default=ROOT / "figures/figure1.svg")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    converter = shutil.which("rsvg-convert")
    if converter is None:
        parser.exit(1, "Figure 1 export requires rsvg-convert on PATH. "
                       "The editable source is figures/figure1.svg.\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("png", ["-w", "5000"]), ("pdf", [])):
        output = args.output_dir / f"fig1.{suffix}"
        subprocess.run([converter, "-f", suffix, *options, "-o", str(output),
                        str(args.svg)], check=True)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
