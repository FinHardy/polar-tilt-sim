# polar-tilt-sim

Simulation and analysis code for the paper on how specimen tilt limits polarisation measurement
from convergent-beam electron diffraction (CBED) in ferroelectrics. The paper compares two
methods: Friedel-pair Bragg-disk intensity asymmetry and Kikuchi-band centre-of-mass (COM). It
covers PbTiO₃ (PTO), BaTiO₃ (BTO) and Cu–Cl boracite.

The multislice simulations use [abTEM](https://abtem.readthedocs.io). The production tilt
sweeps ran as SLURM array jobs on LEONARDO (CINECA). The thickness series ran on a single GPU.

## Layout

```
cifs/                    Input structures (polarised PTO, BTO, Cu-Cl boracite)
simulation_scripts/      abTEM simulations (one script per material and method)
figure_notebooks/        Notebooks that make every paper figure and the tilt-failure table
kikuchi_com.py           Corrected Kikuchi COM metric, shared by the notebooks and scripts
polarisation_direction.py  Sign convention: which way P points on the detector for each structure
compute_max_angles.py    Maximum scattering angle for the simulation grid
slim_zenodo.py           Builds the cropped float32 data files deposited on Zenodo
paper.mplstyle           Matplotlib style used for all figures
```

## Paper figures

| Paper figure | Notebook / script | Output (under `figure_notebooks/figures/`) |
|---|---|---|
| Fig. 2 Friedel-pair disks (selected ROIs) | `figure_notebooks/create_paper_figs_disks_intensity_MORE_DISKS.ipynb` | `corrected_final/corrected_final_less_disks_intensity.pdf` |
| Fig. 3 Kikuchi COM, short range | `figure_notebooks/create_paper_figs_kikuchi_short_range.ipynb` | `corrected_final/corrected_final_kikuchi_short_range.pdf` |
| Fig. 4 Method comparison | `figure_notebooks/combined_disk_intensity_and_kikuchi_plot.ipynb` | `corrected_final/corrected_final_combined_disk_kikuchi_normalised.pdf` |
| Fig. 5 Thickness series | `python simulation_scripts/thickness_series.py plot` | `data/thickness_series/thickness_series.pdf` |
| Table: tilt-failure angles | `figure_notebooks/tilt_failure_table.ipynb` | `corrected_final/tilt_failure_table.tex` |
| Fig. 1A Friedel-pair tilt grid | `figure_notebooks/create_paper_figs_disks_intensity_grid.ipynb` | `corrected_final/corrected_final_disks_intensity_grid.pdf` |
| Fig. 2A Kikuchi COM tilt grid | `figure_notebooks/create_paper_figs_kikuchi_grid.ipynb` | `corrected_final/corrected_final_kikuchi_grid.pdf` |
| Fig. 3A All disk ROIs | `figure_notebooks/create_paper_figs_disks_intensity_MORE_DISKS.ipynb` | `corrected_final/corrected_final_disks_intensity.pdf` |
| Fig. 4A Kikuchi COM, long range | `figure_notebooks/create_paper_figs_kikuchi_long_range.ipynb` | `corrected_final/corrected_final_kikuchi_long_range.pdf` |

The notebooks are committed with their outputs, so the figures can be seen on GitHub without
running anything. Each figure notebook can also write `unflipped_` (raw simulation sign) and
`aligned_` variants. The paper uses `corrected_final`, in which every material is shown with P
pointing to the top box. See `polarisation_direction.py`.

Supporting analysis:

- `figure_notebooks/kikuchi_tilt_sensitivity.ipynb`: Kikuchi COM sensitivity to tilt and to the box position.
- `simulation_scripts/tilt_thickness_series.py`: PTO tilt × thickness series.
- `simulation_scripts/boracite_disk_beam_tilt_sim.py` and `boracite_disk_sweep_2048.py`:
  beam tilt versus specimen tilt, and grid-convergence checks for boracite.

## Data

The simulated diffraction patterns (about 15 GB) are deposited on Zenodo: **DOI: TODO**.

The notebooks read from `data/` at the repository root, the same layout as the original
analysis. Download the Zenodo files and place them as follows:

| Zenodo file | Path the notebooks expect |
|---|---|
| `short_range_tilt/PTO_short_range_tilt.npz` | `data/short_range/PTO_short_range_21_sept_asymmetry_data.npz` |
| `short_range_tilt/BTO_short_range_tilt.npz` | `data/short_range/BTO_kikuchi_final_15_slice_asymmetry_data.npz` |
| `short_range_tilt/boracite_short_range_tilt.npz` | `data/short_range/1_slice_thickness/boracite_kikuchi_1_slice_asymmetry_data.npz` |
| `long_range_tilt/BTO_long_range_tilt.npz` | `data/long_range/BTO_long_range_kikuchi_asymmetry_data.npz` |
| `long_range_tilt/boracite_long_range_tilt.npz` | `data/long_range/Boracite_long_range_kikuchi_asymmetry_data.npz` |
| `long_range_tilt/PTO_long_range_tilt_export/` | `data/long_range_export/PTO_long_range_tilt_split_job58325574_export/` |
| `tilt_grids/{PTO,BTO,Boracite}_tilt_grid.npz` | `data/grids/kikuchi/{PTO,BTO,Boracite}_kikuchi_grid.npz` |
| `thickness_series/{PTO,BTO,Boracite}/` | `data/thickness_series/{PTO,BTO,Boracite}/` |

The Zenodo files have been slimmed by `slim_zenodo.py`, which makes two changes:

- Each 4096 × 4096 pattern is cropped to the central 2732 × 2732 px, the region inside the
  2/3 antialiasing bandlimit. The pattern centre stays at N/2. Stored box coordinates are
  shifted to match, and the shift is saved as `crop_offset_px` (682).
- The data are stored as float32, which is exactly what abTEM produced.

Asymmetry values computed from the slimmed files match the originals. Any pixel coordinate
hard-coded against the 4096 grid must be shifted by `crop_offset_px`.

## Environment

| | Simulations | Analysis and figures |
|---|---|---|
| Python | 3.11 | 3.10 |
| abTEM | 1.0.8 | 1.0.8 |
| ASE | 3.27 | 3.29 |
| NumPy | 2.3 | 1.26 |
| SciPy / Matplotlib | 1.16 / 3.10 | 1.15 / 3.10 |
| GPU | CuPy 13.6 (CUDA 12) | n/a |

```bash
pip install -r requirements.txt
mkdir -p ~/.config/matplotlib/stylelib && cp paper.mplstyle ~/.config/matplotlib/stylelib/
```

The notebooks call `plt.style.use("paper")` and add the repository root to `sys.path` so that
they can import `kikuchi_com` and `polarisation_direction`.

## Running the simulations

Settings common to all runs: 300 kV, 0.8 mrad convergence semi-angle, 4096² grid, full detector,
frozen phonons. Each script's docstring gives its usage and exact settings. The parameters
of each run are written to `parameters.txt` next to its output.

The tilt sweeps (`*_kikuchi_tilt_split_sim.py`, `*_disk_intensity_sim.py`) split the tilt angles
across a SLURM array, one task per tilt or group of tilts:

```bash
python simulation_scripts/PTO_kikuchi_tilt_split_sim.py --task_id $SLURM_ARRAY_TASK_ID --tilts_per_job 1
```

Output is written to `simulation_output/` under the directory you run from (set by `base_output_dir` in each script).

## Licence

See `LICENSE`.
