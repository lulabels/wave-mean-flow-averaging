#!/usr/bin/env python3
"""Compare lab-frame (left) and wave-frame (right) surface-height averages.

    python figures/graphical_abstract.py --regenerate-fields --verify

This command builds the fields from source and writes the 1080 x 900 image
to outputs/graphical_abstract/graphical_abstract.png. Without --regenerate-fields,
the script uses the included data/graphical_abstract_fields.npz instead.

The construction is in wave_averaging/graphical_abstract.py:

    construct_states()   Prescribe a spiral-shaped balanced height field and
                         its geostrophic velocities. Add a separate height
                         packet projected onto the linear RSW wave branches.
    sinc_filter_wave()  Compute the lab-frame boxcar mean: balanced field plus
                         sinc(omega*T/2) times each wave mode. Removing the
                         linear wave evolution before averaging retains the
                         initial balanced + wave state in the wave frame.
    regenerate_fields() Crop the two height fields to the same 6 x 5 region.
    compose()           Interpolate to image pixels. Find a top-to-bottom
                         zero contour of frame_height - lab_height and use
                         the lab field on its left, the wave-frame field on
                         its right. Values agree on the curve; no blending.
    colourize()         Apply one fixed colour scale and palette to both sides.

The active configuration is N=768, Lx=9.6, Ly=8, f=g=H=1 and T=4*pi. The
averages use exact linear Fourier multipliers, without time stepping. The
spiral geometry is prescribed, and the image joins the lab-frame and wave-frame
height averages along a curve where their values agree.

For a separate data-building step, use generate_data/build_graphical_abstract_data.py
and pass its output with --data. The small reference JSON supplies the sRGB
profile and reference hashes. --verify checks RGB pixels against the reference
image. JPEG, thumbnail, composition arrays and diagnostics are also written
beside the PNG.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib"))
sys.path.insert(0, str(ROOT))
import numpy as np
from wave_averaging.graphical_abstract import DEFAULT_DATA, build_inputs, load_fields, render


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/graphical_abstract")
    parser.add_argument("--regenerate-fields", action="store_true")
    parser.add_argument("--verify", action="store_true", help="require reference RGB-pixel equality")
    args = parser.parse_args()
    if args.regenerate_fields:
        fields = build_inputs()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_dir / "graphical_abstract_fields.npz", **fields)
    else:
        fields = load_fields(args.data)
    print(json.dumps(render(fields, args.output_dir, verify=args.verify), indent=2))


if __name__ == "__main__":
    main()
