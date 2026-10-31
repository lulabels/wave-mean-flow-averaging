#!/usr/bin/env python3
"""Calculate the two surface-height fields used in the graphical abstract.

    python generate_data/build_graphical_abstract_data.py
    python figures/graphical_abstract.py --data work/generated_data/graphical_abstract_fields.npz --verify

The first command writes work/generated_data/graphical_abstract_fields.npz; the second
renders it to outputs/graphical_abstract/graphical_abstract.png.

wave_averaging/graphical_abstract.py supplies the prescribed spiral-shaped
balanced field and a separately projected wave packet on a 768 x 768 grid.
It calculates a centered T=4*pi lab-frame average using the exact sinc wave
multiplier. The wave-frame average retains the initial balanced + wave state.
Both height fields are cropped to the same 6 x 5 region and stored as lab_crop
and frame_crop, with a common colour scale. The renderer chooses their join.

No saved fields are read. data/graphical_abstract_reference.json contributes
only export/reference metadata and the colour profile. The numerical settings
are in configuration(), SketchParameters and the field constructors in the
shared module.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib"))
sys.path.insert(0, str(ROOT))

import numpy as np
from wave_averaging.graphical_abstract import build_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "work/generated_data/graphical_abstract_fields.npz")
    args = parser.parse_args()
    fields = build_inputs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fields)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
