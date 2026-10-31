# Installation and reproduction

Instructions for the code accompanying “A unifying view of averaging for wave–mean flow decomposition”. See [README.md](README.md) for the paper and repository overview.

## Figure data

The full Zenodo ZIP includes `data/fig7_8_rsw_products.npz` (about 96 MiB). This file is omitted from GitHub because it exceeds GitHub's browser-upload limit.

**Archive status:** Zenodo DOI `10.5281/zenodo.23825819` is reserved; the archive has not yet been published. After publication, download the full code ZIP from https://doi.org/10.5281/zenodo.23825819 and either use that complete bundle or copy its `data/fig7_8_rsw_products.npz` into the `data/` folder of your GitHub download.

Add this file before running the full numerical test suite or reproducing Figures 7–8. Its SHA-256 checksum, also recorded in `data/checksums.json`, is:

```text
628f66d402335c3427b6c44134e815f7ed8504b92a8e11e29a627d7c7fed8a50
```

The smaller figure inputs are included on GitHub. Once the large file is in place, both distributions use the same reproduction commands below.

## Installation

With Conda:

```bash
conda env create -f environment.yml
conda activate wave-averaging
```

Alternatively, with Python 3.13.5:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.1
python -m pip install -r requirements.txt
python -m pip check
```

Both routes install the versions listed in `requirements.txt`.

## Reproducing the figures

From this directory, run:

```bash
python -m unittest discover -s tests -v
python reproduce_all.py
```

These commands check the numerical routines and input checksums, then draw Figures 2–8 and the graphical abstract from the included data. Images go to `outputs/`, with the abstract in `outputs/graphical_abstract/`. After installation, this works offline.

To produce one figure, for example Figure 5:

```bash
python reproduce_all.py --figures 5
```

Each figure script documents its inputs and settings:

| Figure | Script |
|---|---|
| 1 | [figure1.py](figures/figure1.py), [editable SVG](figures/figure1.svg) |
| 2 | [figure2.py](figures/figure2.py) |
| 3 | [figure3.py](figures/figure3.py) |
| 4 | [figure4.py](figures/figure4.py) |
| 5 | [figure5.py](figures/figure5.py) |
| 6 | [figure6.py](figures/figure6.py) |
| 7–8 | [figures7_8.py](figures/figures7_8.py) |
| 9 | [figure9.py](figures/figure9.py) |
| Graphical abstract | [graphical_abstract.py](figures/graphical_abstract.py) |

### Figure 1

The editable [figure1.svg](figures/figure1.svg) includes the embedded artwork. With `rsvg-convert` installed on `PATH` (tested with version 2.62.2), run:

```bash
python reproduce_all.py --figures 1
```

This writes a 5000-pixel-wide `outputs/fig1.png` and `outputs/fig1.pdf`. You can also open the schematic in an SVG editor.

### Graphical abstract

The 1080 × 900 image joins lab-frame and wave-frame height averages of a prescribed linear rotating shallow-water flow. The construction is documented in the [figure script](figures/graphical_abstract.py) and [shared module](wave_averaging/graphical_abstract.py).

To draw it from the included data:

```bash
python reproduce_all.py --graphical-abstract
```

To generate the fields and draw the image:

```bash
python generate_data/build_graphical_abstract_data.py
python figures/graphical_abstract.py --data work/generated_data/graphical_abstract_fields.npz --verify
```

`--verify` checks the image against the reference RGB pixels. The figure script also supports `--regenerate-fields --verify` to build and draw in one command. Field generation uses the source formulas and the included colour profile and reference metadata.

### Figure 9

Download the Baker et al. data and draw the figure:

```bash
python generate_data/download_baker_data.py
python reproduce_all.py --include-figure9
```

The downloader verifies the approximately 46 MB `data.zip` from [Zenodo record 14237745](https://doi.org/10.5281/zenodo.14237745), version 2.0, and extracts the required NetCDF into `data/external/`. Verified downloads are reused. The deposit lists CC BY 4.0; these third-party files are downloaded separately.

For an archive already downloaded on another machine:

```bash
python generate_data/download_baker_data.py --archive /path/to/data.zip
python figures/figure9.py
```

Figure 9 combines the downloaded Eulerian/Lagrangian fields with the included `data/fig9_projection_N256.npz`. The full projection driver is not included. [figure9.py](figures/figure9.py) documents the caption statistics and calculation timings.

## Rerunning the simulations

Commands and numerical methods are documented in each script:

| Script | Calculation |
|---|---|
| [run_rsw_truth.py](generate_data/run_rsw_truth.py) | Figure 2 and the N64 forcing-analysis input. |
| [run_refined_fig78.py](generate_data/run_refined_fig78.py) | N1024 simulation and averaging for Figures 7 and 8(a–c). |
| [build_compact_rsw_data.py](generate_data/build_compact_rsw_data.py) | Extract figure inputs from regenerated data. |
| [compare_refined_fig78.py](generate_data/compare_refined_fig78.py) | Compare fields and retained-wave fractions across refinement runs. |
| [analyze_rsw_forcing.py](generate_data/analyze_rsw_forcing.py) | Figure 8(d) modal forcing analysis. |

The N1024 run saves about 32 GB of history plus processed arrays; these outputs are not included. The Figure 8(d) interaction calculation needs several GB of cache space.

To check the spatial resolution of the Figure 8(d) initial-state forcing using the included data:

```bash
python generate_data/check_rsw_initial_forcing_resolution.py
```

This checks the prescribed band-limited initial state at N32, N64, N128, N256 and N1024 using the archived diagnostic's forward-window convention. It writes `work/fig8_initial_forcing_resolution/report.json` and exits nonzero if a check fails.

## Verification of calculus and algebra

Optional SymPy checks cover wave examples, averaging kernels, drift, reference-frame identities and pressure projection. The [verification README](verification/README.md) lists the equations and assumptions.

```bash
python -m pip install -r verification/requirements.txt
python -m unittest discover -s verification -p 'check_*.py' -v
```

## Folders

| Folder | Contents |
|---|---|
| [figures/](figures/) | Draw the paper figures and graphical abstract. |
| [generate_data/](generate_data/) | Run simulations, build figure inputs, and check the generated data. |
| [data/](data/) | Saved fields, diagnostics and refinement comparisons used by the figure scripts. |
| [wave_averaging/](wave_averaging/) | Shared Python routines, including the rotating shallow-water solver. |
| [tests/](tests/) | Numerical implementation checks for solvers, projectors, averaging and saved inputs. |
| [verification/](verification/) | Optional symbolic identities from the paper, with parameters and assumptions stated. |

Generated images go to `outputs/`, intermediate calculations and caches to `work/`, and Baker downloads to `data/external/`. These folders are Git-ignored. Data-generation examples write to `work/generated_data/`.

Scripts find their default inputs relative to their own location. Relative paths supplied on the command line are resolved from the calling directory.

## Rendering

Tested on macOS arm64. Numerical dependencies are NumPy and SciPy; plots use Matplotlib. PNG bytes depend on the renderer and installed fonts. `python reproduce_all.py --verify-hashes` compares the outputs with the paper-image hashes; numerical tests run independently.

The source Conda renderer uses FreeType 2.14.1; pinned PyPI wheels use 2.6.1 and can produce different bytes. Matplotlib tries Times New Roman, Times, Nimbus Roman and DejaVu Serif. Fonts are installed separately.
