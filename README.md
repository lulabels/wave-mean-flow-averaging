# Wave–mean flow averaging

This repository contains the code and data associated with the paper:

“A unifying view of averaging for wave–mean flow decomposition”  
by L. R. Seitz and Beth A. Wingate

The code can be used to run rotating shallow-water simulations and compare averages in the lab and wave frames. It also includes scripts to reproduce the figures and graphical abstract, together with numerical tests and symbolic checks of selected identities in the paper.

## Overview

This repository provides:

- [Rotating shallow-water solvers and averaging routines](wave_averaging/).
- [Simulation and data-generation scripts](generate_data/).
- [Plotting scripts for the paper figures and graphical abstract](figures/).
- [Saved figure data](data/).
- [Numerical tests](tests/) and optional [symbolic checks](verification/).

See [Installation and reproduction](REPRODUCING.md) for dependencies, commands and details of the included data. Figure 1 is supplied as an editable SVG; Figure 9 uses an additional download from Baker et al.

The large Figures 7–8 data file must be added separately to GitHub downloads; see [the data instructions](REPRODUCING.md#figure-data).

## Citation

Please cite this code deposit and the accompanying paper when using these materials. The code author is L. R. Seitz.

## License

The original software uses the [MIT License](LICENSE). The included original data and artwork use [CC BY 4.0](LICENSE-DATA.md). Third-party materials retain their own licenses.
