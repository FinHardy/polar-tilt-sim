"""
Disk Intensity Polarisation Mapping — Boracite

Runs a 1D tilt sweep over a narrow range around the zone axis, measuring
disk intensity asymmetry as a function of beam tilt. All 41 tilts run in a
single job.

Output:
    simulation_output/boracite_disk_intensity_sweeps/
        boracite_disk_intensity_job{run_id}/
            results.npz
            sim_images/
            sim_images_boxes/
            parameters.txt
"""

import argparse
import os
from datetime import datetime

import abtem
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

structure_file_dict = {
    "PTO": "cifs/pbtio3_polarised_2.cif",
    "boracite": "cifs/cu2-ortho-sym.cif",
    "BTO": "cifs/batio3_polarised.cif",
}

CROP_XY_SIZE = None

BEAM_ENERGY = 300e3  # eV
CONVERGENCE_ANGLE = 0.8  # mrad  — beam parameter, sets disk size
SLICE_THICKNESS = 1.0  # Angstroms
GRID_POINTS = 1024  # 4096

SCAN_PIXELS_X = 2
SCAN_PIXELS_Y = 2

GAMMA = 0.3
PERCENTILE_MIN = 0.5
PERCENTILE_MAX = 99.8

# Tilt sweep definition
TILT_Y_MIN = -0.25  # degrees
TILT_Y_MAX = 0.25  # degrees
TILT_Y_NPTS = 81
TILT_X_FIXED = 0.0  # degrees

# Material
structure_name = "boracite"
supercell_size = (7, 7, 40)
num_phonons = 10
# Detector limit set to the disk angular radius (= CONVERGENCE_ANGLE numerically,
# but this is a detector parameter — the collected scattering angle — not the beam angle).
DISK_DETECTOR_MAX_ANGLE = 35  # mrad
score_type = "com"
base_output_dir = "simulation_output/boracite_disk_intensity_sweeps"

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

try:
    from cuda.bindings import driver as _cuda_driver

    if not hasattr(_cuda_driver, "cuDeviceGetUuid_v2"):
        _cuda_driver.cuDeviceGetUuid_v2 = _cuda_driver.cuDeviceGetUuid
except Exception:
    pass

try:
    import cupy as cp

    abtem.config.set({"device": "gpu"})
    print("GPU acceleration enabled")
except ImportError:
    print("CuPy not installed - using CPU (will be slower)")
    abtem.config.set({"device": "cpu"})
except Exception as e:
    print(f"GPU not available - using CPU: {e}")
    abtem.config.set({"device": "cpu"})

# ============================================================================
# ASYMMETRY CALCULATION FUNCTIONS
# ============================================================================


def calculate_box_asymmetry_score(diffraction_pattern, box1_coords, box2_coords):
    from scipy.ndimage import center_of_mass

    if hasattr(diffraction_pattern, "array"):
        pattern = np.array(diffraction_pattern.array)
    else:
        pattern = np.array(diffraction_pattern)

    pattern = np.asarray(pattern, dtype=np.float64)

    if pattern.ndim > 2:
        pattern = np.squeeze(pattern)

    ny, nx = pattern.shape

    def parse_coords(coords, ny, nx):
        (y_min, y_max), (x_min, x_max) = coords
        if 0 <= y_min <= 1 and 0 <= y_max <= 1:
            y_min = int(y_min * ny)
            y_max = int(y_max * ny)
        if 0 <= x_min <= 1 and 0 <= x_max <= 1:
            x_min = int(x_min * nx)
            x_max = int(x_max * nx)
        y_min = max(0, min(int(y_min), ny - 1))
        y_max = max(0, min(int(y_max), ny))
        x_min = max(0, min(int(x_min), nx - 1))
        x_max = max(0, min(int(x_max), nx))
        return y_min, y_max, x_min, x_max

    y1_min, y1_max, x1_min, x1_max = parse_coords(box1_coords, ny, nx)
    y2_min, y2_max, x2_min, x2_max = parse_coords(box2_coords, ny, nx)

    mask1 = np.zeros((ny, nx), dtype=bool)
    mask2 = np.zeros((ny, nx), dtype=bool)
    mask1[y1_min:y1_max, x1_min:x1_max] = True
    mask2[y2_min:y2_max, x2_min:x2_max] = True

    mask_1_com = center_of_mass(pattern * mask1)
    mask_2_com = center_of_mass(pattern * mask2)

    midpoint_y = (mask_1_com[0] + mask_2_com[0]) / 2
    midpoint_x = (mask_1_com[1] + mask_2_com[1]) / 2

    center_y, center_x = ny / 2, nx / 2
    dy = midpoint_y - center_y
    dx = midpoint_x - center_x

    com_y_displacement = -(dy / ny) * 100

    box1_region = pattern[y1_min:y1_max, x1_min:x1_max]
    box2_region = pattern[y2_min:y2_max, x2_min:x2_max]

    box1_intensity = np.sum(box1_region)
    box2_intensity = np.sum(box2_region)
    box1_mean = np.mean(box1_region)
    box2_mean = np.mean(box2_region)
    total_intensity = box1_intensity + box2_intensity
    intensity_asymmetry = box1_intensity - box2_intensity

    return {
        "com_y_displacement": com_y_displacement,
        "intensity_asymmetry": intensity_asymmetry,
        "y_displacement": dy,
        "x_displacement": dx,
        "midpoint_y": midpoint_y,
        "midpoint_x": midpoint_x,
        "box1_com_y": mask_1_com[0],
        "box1_com_x": mask_1_com[1],
        "box2_com_y": mask_2_com[0],
        "box2_com_x": mask_2_com[1],
        "total_intensity": total_intensity,
        "box1_intensity": box1_intensity,
        "box2_intensity": box2_intensity,
        "box1_mean": box1_mean,
        "box2_mean": box2_mean,
        "box1_coords": (y1_min, y1_max, x1_min, x1_max),
        "box2_coords": (y2_min, y2_max, x2_min, x2_max),
    }


def calculate_tilt_induced_shift(tilt_x_deg, tilt_y_deg, pattern_shape, max_angle_mrad):
    tilt_x_mrad = tilt_x_deg * 1000 / 57.2958
    tilt_y_mrad = tilt_y_deg * 1000 / 57.2958

    ny, nx = pattern_shape

    if isinstance(max_angle_mrad, (tuple, list)):
        max_angle_y, max_angle_x = max_angle_mrad
    else:
        max_angle_y = max_angle_x = max_angle_mrad

    mrad_per_pixel_y = (2 * max_angle_y) / ny
    mrad_per_pixel_x = (2 * max_angle_x) / nx

    shift_y_px = -tilt_y_mrad / mrad_per_pixel_y
    shift_x_px = -tilt_x_mrad / mrad_per_pixel_x

    return shift_y_px, shift_x_px


def adjust_boxes_for_tilt(
    box1,
    box2,
    tilt_x_deg,
    tilt_y_deg,
    pattern_shape,
    max_angle_mrad,
    apply_y_shift=False,
    apply_x_shift=True,
):
    shift_y, shift_x = calculate_tilt_induced_shift(
        tilt_x_deg, tilt_y_deg, pattern_shape, max_angle_mrad
    )
    if not apply_y_shift:
        shift_y = 0.0
    if not apply_x_shift:
        shift_x = 0.0

    def shift_box(box, dy, dx):
        (y_min, y_max), (x_min, x_max) = box
        return ((int(y_min + dy), int(y_max + dy)), (int(x_min + dx), int(x_max + dx)))

    return shift_box(box1, shift_y, shift_x), shift_box(box2, shift_y, shift_x)


# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================


def run_disk_intensity_sweep(run_dir, box_pairs_dict):
    sim_images_dir = os.path.join(run_dir, "sim_images")
    sim_images_boxes_dir = os.path.join(run_dir, "sim_images_boxes")
    for d in (run_dir, sim_images_dir, sim_images_boxes_dir):
        os.makedirs(d, exist_ok=True)

    all_tilt_y = np.linspace(TILT_Y_MIN, TILT_Y_MAX, TILT_Y_NPTS)
    n_tilts = len(all_tilt_y)

    print("=" * 70)
    print(f"DISK INTENSITY SWEEP  |  {structure_name}")
    print(f"  Tilt-Y range : {TILT_Y_MIN} to {TILT_Y_MAX} deg  ({n_tilts} pts)")
    print(
        f"  Grid pts     : {GRID_POINTS}  |  Scan: {SCAN_PIXELS_X}x{SCAN_PIXELS_Y}  |  Phonons: {num_phonons}"
    )
    print(f"  Disk detector: {DISK_DETECTOR_MAX_ANGLE} mrad")
    print(f"  Output       : {run_dir}")
    print("=" * 70)

    # ---- Material setup ----------------------------------------------------
    ZONE_AXIS = [1, 1, 0]
    THERMAL_SIGMA = 0.1

    # ---- Build base supercell ----------------------------------------------
    atoms = read(structure_file_dict[structure_name])
    base_rotated_atoms = atoms.copy()

    zone_vector = np.array(ZONE_AXIS, dtype=float)
    zone_vector = zone_vector / np.linalg.norm(zone_vector)
    z_vector = np.array([0.0, 0.0, 1.0])
    rotation_axis = np.cross(zone_vector, z_vector)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(zone_vector, z_vector), -1.0, 1.0))
        base_rotated_atoms.rotate(
            np.degrees(angle), rotation_axis, center="COP", rotate_cell=True
        )

    base_rotated_atoms = abtem.orthogonalize_cell(base_rotated_atoms)
    base_supercell = base_rotated_atoms * supercell_size
    base_supercell.rotate(-45, "z", center="COP", rotate_cell=True)

    if CROP_XY_SIZE is not None:
        cell_lengths = base_supercell.get_cell().lengths()
        current_x, current_y, current_z = cell_lengths
        positions = base_supercell.get_positions()
        center_x, center_y = current_x / 2, current_y / 2
        half_crop = CROP_XY_SIZE / 2
        mask = (
            (positions[:, 0] >= center_x - half_crop)
            & (positions[:, 0] <= center_x + half_crop)
            & (positions[:, 1] >= center_y - half_crop)
            & (positions[:, 1] <= center_y + half_crop)
        )
        cropped_positions = positions[mask].copy()
        cropped_positions[:, 0] -= center_x - half_crop
        cropped_positions[:, 1] -= center_y - half_crop
        base_supercell = Atoms(
            symbols=[
                base_supercell[i].symbol for i in range(len(base_supercell)) if mask[i]
            ],
            positions=cropped_positions,
            cell=[CROP_XY_SIZE, CROP_XY_SIZE, current_z],
            pbc=base_supercell.pbc,
        )

    print(
        f"Supercell built: {supercell_size}, {len(base_supercell)} atoms, "
        f"cell={base_supercell.get_cell().lengths()}"
    )

    # ---- Storage arrays ----------------------------------------------------
    label_keys = list(box_pairs_dict.keys())
    diffraction_images = [None] * n_tilts
    asymmetry_grids = {k: np.full(n_tilts, np.nan) for k in label_keys}
    intensity_grids = {k: np.full(n_tilts, np.nan) for k in label_keys}
    com_grids = {k: np.full(n_tilts, np.nan) for k in label_keys}
    box_coords_box1 = {k: np.zeros((n_tilts, 4), dtype=int) for k in label_keys}
    box_coords_box2 = {k: np.zeros((n_tilts, 4), dtype=int) for k in label_keys}
    _img_shape = None

    results_npz = os.path.join(run_dir, "results.npz")

    # ---- Loop over tilt angles ---------------------------------------------
    for j, tilt_y in enumerate(all_tilt_y):
        tilt_x = TILT_X_FIXED
        print(f"\n  [{j + 1}/{n_tilts}] tilt_y={tilt_y:.4f}°")

        try:
            tilted_supercell = base_supercell.copy()
            if tilt_x != 0 or tilt_y != 0:
                tilted_supercell.rotate(tilt_x, "x", center="COP", rotate_cell=False)
                tilted_supercell.rotate(tilt_y, "y", center="COP", rotate_cell=False)

            frozen_phonon = abtem.FrozenPhonons(
                tilted_supercell,
                sigmas=THERMAL_SIGMA,
                num_configs=num_phonons,
            )
            potential = abtem.Potential(
                frozen_phonon,
                gpts=GRID_POINTS,
                slice_thickness=SLICE_THICKNESS,
                projection="finite",
            )

            extent_x, extent_y = potential.extent
            if abs(extent_x - extent_y) > 0.1:
                potential = abtem.Potential(
                    frozen_phonon,
                    gpts=GRID_POINTS,
                    sampling=potential.sampling[0],
                    slice_thickness=SLICE_THICKNESS,
                    projection="finite",
                )

            probe = abtem.Probe(
                energy=BEAM_ENERGY,
                semiangle_cutoff=CONVERGENCE_ANGLE,
                gpts=GRID_POINTS,
                sampling=potential.sampling,
                defocus=0,
            )

            scan = abtem.GridScan(
                start=[0, 0],
                end=[potential.extent[0], potential.extent[1]],
                gpts=(SCAN_PIXELS_X, SCAN_PIXELS_Y),
            )

            fourd_stem = probe.scan(
                scan=scan,
                potential=potential,
                detectors=abtem.PixelatedDetector(
                    max_angle=DISK_DETECTOR_MAX_ANGLE,
                    reciprocal_space=True,
                ),
            )
            result = fourd_stem.compute(progress_bar=False)
            diffraction_pattern = np.array(result.array).mean(axis=(0, 1))

            # Free GPU memory pool after each tilt to prevent VRAM exhaustion
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
            diffraction_images[j] = np.array(diffraction_pattern)
            if _img_shape is None:
                _img_shape = diffraction_pattern.shape

            img_filename = f"tilt_x_{tilt_x:.3f}_y_{tilt_y:.3f}.png"
            vmin = np.percentile(diffraction_pattern, PERCENTILE_MIN)
            vmax_img = np.percentile(diffraction_pattern, PERCENTILE_MAX)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(
                diffraction_pattern,
                cmap="inferno",
                origin="lower",
                norm=PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax_img),
            )
            ax.set_title(
                f"Tilt X={tilt_x:.3f}°, Y={tilt_y:.3f}°", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("X (pixels)", fontsize=10)
            ax.set_ylabel("Y (pixels)", fontsize=10)
            plt.colorbar(im, ax=ax, label="Intensity")
            plt.tight_layout()
            plt.savefig(
                os.path.join(sim_images_dir, img_filename), dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

            pattern_shape = diffraction_pattern.shape
            max_angle_mrad = result.max_angles

            fig_b, ax_b = plt.subplots(figsize=(8, 8))
            im_b = ax_b.imshow(
                diffraction_pattern,
                cmap="inferno",
                origin="lower",
                norm=PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax_img),
            )
            ax_b.set_title(
                f"Tilt X={tilt_x:.3f}°, Y={tilt_y:.3f}° (with boxes)",
                fontsize=12,
                fontweight="bold",
            )
            ax_b.set_xlabel("X (pixels)", fontsize=10)
            ax_b.set_ylabel("Y (pixels)", fontsize=10)
            plt.colorbar(im_b, ax=ax_b, label="Intensity")

            for label, (box1_base, box2_base) in box_pairs_dict.items():
                box1_coords, box2_coords = adjust_boxes_for_tilt(
                    box1_base,
                    box2_base,
                    tilt_x,
                    tilt_y,
                    pattern_shape,
                    max_angle_mrad,
                    apply_y_shift=False,
                    apply_x_shift=True,
                )

                box_result = calculate_box_asymmetry_score(
                    diffraction_pattern, box1_coords, box2_coords
                )

                intensity_grids[label][j] = box_result["intensity_asymmetry"]
                com_grids[label][j] = box_result["com_y_displacement"]

                recorded = (
                    box_result["intensity_asymmetry"]
                    if score_type == "intensity"
                    else box_result["com_y_displacement"]
                )
                asymmetry_grids[label][j] = recorded
                box_coords_box1[label][j] = box_result["box1_coords"]
                box_coords_box2[label][j] = box_result["box2_coords"]

                y1_min, y1_max, x1_min, x1_max = box_result["box1_coords"]
                y2_min, y2_max, x2_min, x2_max = box_result["box2_coords"]
                ax_b.add_patch(
                    Rectangle(
                        (x1_min, y1_min),
                        x1_max - x1_min,
                        y1_max - y1_min,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
                ax_b.add_patch(
                    Rectangle(
                        (x2_min, y2_min),
                        x2_max - x2_min,
                        y2_max - y2_min,
                        linewidth=2,
                        edgecolor="blue",
                        facecolor="none",
                    )
                )
                ax_b.plot(
                    box_result["box1_com_x"],
                    box_result["box1_com_y"],
                    "r+",
                    markersize=15,
                    markeredgewidth=2,
                )
                ax_b.plot(
                    box_result["box2_com_x"],
                    box_result["box2_com_y"],
                    "b+",
                    markersize=15,
                    markeredgewidth=2,
                )
                ax_b.plot(
                    box_result["midpoint_x"],
                    box_result["midpoint_y"],
                    "go",
                    markersize=10,
                    markeredgewidth=2,
                    fillstyle="none",
                )

                print(
                    f"    {label}: COM-Y={box_result['com_y_displacement']:+.6f}%  "
                    f"intens_asym={box_result['intensity_asymmetry']:+.7f}"
                )

            plt.tight_layout()
            plt.savefig(
                os.path.join(sim_images_boxes_dir, img_filename),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig_b)

        except Exception as e:
            print(f"    ERROR at tilt_y={tilt_y:.4f}: {e}")

        # Checkpoint after each tilt
        _save_npz(
            results_npz,
            all_tilt_y,
            diffraction_images,
            asymmetry_grids,
            intensity_grids,
            com_grids,
            box_coords_box1,
            box_coords_box2,
            _img_shape,
        )
        print(f"    Checkpoint saved -> {results_npz}")

    print(f"\nSweep complete. Results: {results_npz}")


def _save_npz(
    path,
    tilt_y_values,
    diffraction_images,
    asymmetry_grids,
    intensity_grids,
    com_grids,
    box_coords_box1,
    box_coords_box2,
    img_shape,
):
    n = len(tilt_y_values)

    if img_shape is not None:
        img_array = np.zeros((n,) + img_shape)
        for j, img in enumerate(diffraction_images):
            if img is not None:
                img_array[j] = img
            else:
                img_array[j] = np.nan
    else:
        img_array = np.array([])

    save_dict = dict(
        tilt_y_values=tilt_y_values,
        tilt_x_fixed=np.array(TILT_X_FIXED),
        diffraction_images=img_array,
    )
    for label in asymmetry_grids:
        save_dict[f"{label}"] = asymmetry_grids[label]
        save_dict[f"{label}_intensity_asymmetry"] = intensity_grids[label]
        save_dict[f"{label}_com_y_displacement"] = com_grids[label]
        save_dict[f"{label}_box1_coords"] = box_coords_box1[label]
        save_dict[f"{label}_box2_coords"] = box_coords_box2[label]

    np.savez(path, **save_dict)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Disk intensity polarisation mapping — boracite."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier. Defaults to SLURM_JOB_ID or a timestamp.",
    )
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = os.environ.get(
            "SLURM_JOB_ID",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    # ---- Box coordinates (update as needed for disk region) ----------------
    disk_box1 = ((200 * 4, 350 * 4), (497 * 4, 527 * 4))
    disk_box2 = ((674 * 4, 824 * 4), (497 * 4, 527 * 4))
    box_pairs = {"horizontal_boxes": (disk_box1, disk_box2)}

    run_dir = os.path.join(base_output_dir, f"boracite_disk_intensity_job{args.run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "parameters.txt"), "w") as f:
        f.write("Disk Intensity Polarisation Mapping Parameters\n")
        f.write("=" * 60 + "\n")
        f.write(f"Structure       : {structure_name}\n")
        f.write(f"Supercell       : {supercell_size}\n")
        f.write(f"Grid points     : {GRID_POINTS}\n")
        f.write(f"Scan pixels     : {SCAN_PIXELS_X}x{SCAN_PIXELS_Y}\n")
        f.write(f"Beam energy     : {BEAM_ENERGY} eV\n")
        f.write(f"Conv. angle     : {CONVERGENCE_ANGLE} mrad\n")
        f.write(f"Disk detector   : {DISK_DETECTOR_MAX_ANGLE} mrad\n")
        f.write(f"Slice thick.    : {SLICE_THICKNESS} A\n")
        f.write(f"Num phonons     : {num_phonons}\n")
        f.write(f"Score type      : {score_type}\n")
        f.write(f"Tilt Y range    : {TILT_Y_MIN} to {TILT_Y_MAX} deg\n")
        f.write(f"Tilt Y points   : {TILT_Y_NPTS}\n")
        f.write(f"Tilt X fixed    : {TILT_X_FIXED} deg\n")
        f.write(f"Run ID          : {args.run_id}\n")

    run_disk_intensity_sweep(run_dir, box_pairs)