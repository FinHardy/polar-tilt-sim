"""
Tilt-Split Simulation for BTO Asymmetry Analysis

Runs a subset of tilt angles from the full X-point sweep, designed to be
submitted as a SLURM array job where each task handles a small number of
tilt angles within the 24-hour cluster time limit.

All simulation settings (grid points, scan pixels, phonon configs, supercell)
are identical to non-batch_intensity_asymmetry_calc.py.

Usage:
    python tilt_split_sim.py --task_id INT [--tilts_per_job INT] [--run_id STR]

Arguments:
    --task_id INT       SLURM array task ID (0-indexed). Determines which tilt
                        angles this job processes.
    --tilts_per_job INT Number of tilt angles per job (default: 1).
    --run_id STR        Shared identifier for this run, used to name the output
                        directory so all tasks write to the same place. Defaults
                        to SLURM_ARRAY_JOB_ID if available, else a timestamp.

Output:
    /leonardo_work/EUHPC_E06_088/fhardy00/cbed_simulations/BTO_asymmetry_sweeps/
        BTO_tilt_split_job{run_id}/
            partial_results/
                tilt_task_000.npz      # one file per task
                sim_images/            # diffraction pattern PNGs
                sim_images_boxes/      # patterns with box overlays
            parameters.txt
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from datetime import datetime
from ase.io import read
from ase import Atoms
import abtem

# ============================================================================
# CONFIGURATION PARAMETERS  (match non-batch_intensity_asymmetry_calc.py)
# ============================================================================

structure_file_dict = {
    'PTO': 'cifs/pbtio3_polarised_2.cif',
    'boracite': 'cifs/cu2-ortho-sym.cif',
    'BTO': 'cifs/batio3_polarised.cif',
}

CROP_XY_SIZE = None

BEAM_ENERGY = 300e3          # eV
CONVERGENCE_ANGLE = 0.8      # mrad
SLICE_THICKNESS = 1.5        # Angstroms
GRID_POINTS = 1024 * 4       # 4096

SCAN_PIXELS_X = 4
SCAN_PIXELS_Y = 4

GAMMA = 0.3
PERCENTILE_MIN = 0.5
PERCENTILE_MAX = 99.8

# Full tilt sweep definition — must match non-batch script so indices align
TILT_Y_MIN = -0.6   # degrees
TILT_Y_MAX =  0.6   # degrees
TILT_Y_NPTS = 41
TILT_X_FIXED = 0.0  # degrees (1D sweep in Y only)

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

try:
    from cuda.bindings import driver as _cuda_driver
    if not hasattr(_cuda_driver, 'cuDeviceGetUuid_v2'):
        _cuda_driver.cuDeviceGetUuid_v2 = _cuda_driver.cuDeviceGetUuid
except Exception:
    pass

try:
    import cupy as cp
    abtem.config.set({'device': 'gpu'})
    print("GPU acceleration enabled")
except ImportError:
    print("CuPy not installed - using CPU (will be slower)")
    abtem.config.set({'device': 'cpu'})
except Exception as e:
    print(f"GPU not available - using CPU: {e}")
    abtem.config.set({'device': 'cpu'})

# ============================================================================
# ASYMMETRY CALCULATION FUNCTIONS  (unchanged from non-batch script)
# ============================================================================

def calculate_box_asymmetry_score(diffraction_pattern, box1_coords, box2_coords):
    """
    Calculate asymmetry score using masked center of mass approach.

    The y-displacement of the COM midpoint from the pattern centre is used as
    the asymmetry metric.
    """
    from scipy.ndimage import center_of_mass

    if hasattr(diffraction_pattern, 'array'):
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
        'com_y_displacement': com_y_displacement,
        'intensity_asymmetry': intensity_asymmetry,
        'y_displacement': dy,
        'x_displacement': dx,
        'midpoint_y': midpoint_y,
        'midpoint_x': midpoint_x,
        'box1_com_y': mask_1_com[0],
        'box1_com_x': mask_1_com[1],
        'box2_com_y': mask_2_com[0],
        'box2_com_x': mask_2_com[1],
        'total_intensity': total_intensity,
        'box1_intensity': box1_intensity,
        'box2_intensity': box2_intensity,
        'box1_mean': box1_mean,
        'box2_mean': box2_mean,
        'box1_coords': (y1_min, y1_max, x1_min, x1_max),
        'box2_coords': (y2_min, y2_max, x2_min, x2_max),
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


def adjust_boxes_for_tilt(box1, box2, tilt_x_deg, tilt_y_deg, pattern_shape,
                           max_angle_mrad, apply_y_shift=False, apply_x_shift=True):
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

def run_tilt_subset(
    task_id,
    tilts_per_job,
    run_dir,
    structure_name,
    supercell_size,
    detector_max_angle,
    box_pairs_dict,
    num_phonons,
    score_type='com',
):
    """
    Run simulations for the tilt-angle subset assigned to this task.

    Saves one partial NPZ to run_dir/partial_results/tilt_task_{task_id:03d}.npz
    and diffraction-pattern images under run_dir/partial_results/sim_images[_boxes]/.
    A checkpoint NPZ is written after every tilt so progress survives a job kill.
    """
    # ---- Set up directories ------------------------------------------------
    partial_dir = os.path.join(run_dir, 'partial_results')
    sim_images_dir = os.path.join(partial_dir, 'sim_images')
    sim_images_boxes_dir = os.path.join(partial_dir, 'sim_images_boxes')
    for d in (partial_dir, sim_images_dir, sim_images_boxes_dir):
        os.makedirs(d, exist_ok=True)

    # ---- Resolve which tilt-Y values this task owns ------------------------
    all_tilt_y = np.linspace(TILT_Y_MIN, TILT_Y_MAX, TILT_Y_NPTS)
    start_idx = task_id * tilts_per_job
    end_idx   = min(start_idx + tilts_per_job, TILT_Y_NPTS)

    if start_idx >= TILT_Y_NPTS:
        print(f"Task {task_id}: start_idx={start_idx} >= TILT_Y_NPTS={TILT_Y_NPTS}. Nothing to do.")
        return

    tilt_y_subset = all_tilt_y[start_idx:end_idx]
    global_indices = np.arange(start_idx, end_idx)
    n_tilts = len(tilt_y_subset)
    tilt_x_values = np.array([TILT_X_FIXED])

    print("=" * 70)
    print(f"TILT-SPLIT SIMULATION  |  task {task_id}  |  tilts {start_idx}–{end_idx - 1}")
    print(f"  Structure : {structure_name}")
    print(f"  Tilt-Y    : {tilt_y_subset}")
    print(f"  Grid pts  : {GRID_POINTS}  |  Scan: {SCAN_PIXELS_X}x{SCAN_PIXELS_Y}  |  Phonons: {num_phonons}")
    print(f"  Output    : {run_dir}")
    print("=" * 70)

    # ---- Material setup ----------------------------------------------------
    if structure_name == 'boracite':
        ZONE_AXIS = [1, 1, 0]
    else:
        ZONE_AXIS = [1, 0, 0]

    if structure_name == 'PTO':
        THERMAL_SIGMA = {"Pb": 0.1068, "Ti": 0.0872, "O": 0.0503}
    elif structure_name == 'BTO':
        THERMAL_SIGMA = {"Ba": 0.0785, "Ti": 0.0703, "O": 0.0656}
    else:
        THERMAL_SIGMA = 0.1

    # ---- Build base supercell (once per task) ------------------------------
    atoms = read(structure_file_dict[structure_name])
    base_rotated_atoms = atoms.copy()

    if ZONE_AXIS != [0, 0, 1]:
        zone_vector = np.array(ZONE_AXIS, dtype=float)
        zone_vector = zone_vector / np.linalg.norm(zone_vector)
        z_vector = np.array([0., 0., 1.])
        rotation_axis = np.cross(zone_vector, z_vector)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(zone_vector, z_vector), -1.0, 1.0))
            base_rotated_atoms.rotate(np.degrees(angle), rotation_axis,
                                      center='COP', rotate_cell=True)

    base_rotated_atoms = abtem.orthogonalize_cell(base_rotated_atoms)
    base_supercell = base_rotated_atoms * supercell_size

    if structure_name == 'boracite':
        base_supercell.rotate(-45, 'z', center='COP', rotate_cell=True)
    if structure_name == 'PTO':
        base_supercell.rotate(-90, 'z', center='COP', rotate_cell=True)

    if CROP_XY_SIZE is not None:
        cell_lengths = base_supercell.get_cell().lengths()
        current_x, current_y, current_z = cell_lengths
        positions = base_supercell.get_positions()
        center_x, center_y = current_x / 2, current_y / 2
        half_crop = CROP_XY_SIZE / 2
        mask = (
            (positions[:, 0] >= center_x - half_crop) &
            (positions[:, 0] <= center_x + half_crop) &
            (positions[:, 1] >= center_y - half_crop) &
            (positions[:, 1] <= center_y + half_crop)
        )
        cropped_positions = positions[mask].copy()
        cropped_positions[:, 0] -= (center_x - half_crop)
        cropped_positions[:, 1] -= (center_y - half_crop)
        base_supercell = Atoms(
            symbols=[base_supercell[i].symbol for i in range(len(base_supercell)) if mask[i]],
            positions=cropped_positions,
            cell=[CROP_XY_SIZE, CROP_XY_SIZE, current_z],
            pbc=base_supercell.pbc,
        )

    print(f"Supercell built: {supercell_size}, {len(base_supercell)} atoms, "
          f"cell={base_supercell.get_cell().lengths()}")

    # ---- Storage arrays for this task's results ----------------------------
    label_keys = list(box_pairs_dict.keys())
    diffraction_images   = [None] * n_tilts
    asymmetry_grids      = {k: np.full(n_tilts, np.nan) for k in label_keys}
    intensity_grids      = {k: np.full(n_tilts, np.nan) for k in label_keys}
    com_grids            = {k: np.full(n_tilts, np.nan) for k in label_keys}
    box_coords_box1      = {k: np.zeros((n_tilts, 4), dtype=int) for k in label_keys}
    box_coords_box2      = {k: np.zeros((n_tilts, 4), dtype=int) for k in label_keys}
    _img_shape = None

    partial_npz = os.path.join(partial_dir, f'tilt_task_{task_id:03d}.npz')

    # ---- Loop over assigned tilt angles ------------------------------------
    for local_j, (global_j, tilt_y) in enumerate(zip(global_indices, tilt_y_subset)):
        tilt_x = TILT_X_FIXED
        print(f"\n  [{local_j + 1}/{n_tilts}] tilt_y={tilt_y:.4f}° (global idx {global_j})")

        try:
            tilted_supercell = base_supercell.copy()
            if tilt_x != 0 or tilt_y != 0:
                tilted_supercell.rotate(tilt_x, 'x', center='COP', rotate_cell=False)
                tilted_supercell.rotate(tilt_y, 'y', center='COP', rotate_cell=False)

            frozen_phonon = abtem.FrozenPhonons(
                tilted_supercell,
                sigmas=THERMAL_SIGMA,
                num_configs=num_phonons,
            )
            potential = abtem.Potential(
                frozen_phonon,
                gpts=GRID_POINTS,
                slice_thickness=SLICE_THICKNESS,
                projection='finite',
            )

            # Ensure square extent
            extent_x, extent_y = potential.extent
            if abs(extent_x - extent_y) > 0.1:
                potential = abtem.Potential(
                    frozen_phonon,
                    gpts=GRID_POINTS,
                    sampling=potential.sampling[0],
                    slice_thickness=SLICE_THICKNESS,
                    projection='finite',
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
                    max_angle=detector_max_angle,
                    reciprocal_space=True,
                ),
            )
            result = fourd_stem.compute(progress_bar=False)
            diffraction_pattern = np.array(result.array).mean(axis=(0, 1))
            diffraction_images[local_j] = np.array(diffraction_pattern)
            if _img_shape is None:
                _img_shape = diffraction_pattern.shape

            # Save plain image
            img_filename = f"tilt_x_{tilt_x:.3f}_y_{tilt_y:.3f}.png"
            vmin = np.percentile(diffraction_pattern, PERCENTILE_MIN)
            vmax_img = np.percentile(diffraction_pattern, PERCENTILE_MAX)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(diffraction_pattern, cmap='inferno', origin='lower',
                           norm=PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax_img))
            ax.set_title(f"Tilt X={tilt_x:.3f}°, Y={tilt_y:.3f}°", fontsize=12, fontweight='bold')
            ax.set_xlabel('X (pixels)', fontsize=10)
            ax.set_ylabel('Y (pixels)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Intensity')
            plt.tight_layout()
            plt.savefig(os.path.join(sim_images_dir, img_filename), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Box analysis + annotated image
            pattern_shape = diffraction_pattern.shape
            max_angle_mrad = result.max_angles

            fig_b, ax_b = plt.subplots(figsize=(8, 8))
            im_b = ax_b.imshow(diffraction_pattern, cmap='inferno', origin='lower',
                                norm=PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax_img))
            ax_b.set_title(f"Tilt X={tilt_x:.3f}°, Y={tilt_y:.3f}° (with boxes)",
                           fontsize=12, fontweight='bold')
            ax_b.set_xlabel('X (pixels)', fontsize=10)
            ax_b.set_ylabel('Y (pixels)', fontsize=10)
            plt.colorbar(im_b, ax=ax_b, label='Intensity')

            for label, (box1_base, box2_base) in box_pairs_dict.items():
                box1_coords, box2_coords = adjust_boxes_for_tilt(
                    box1_base, box2_base,
                    tilt_x, tilt_y,
                    pattern_shape, max_angle_mrad,
                    apply_y_shift=False,
                    apply_x_shift=True,
                )

                box_result = calculate_box_asymmetry_score(
                    diffraction_pattern, box1_coords, box2_coords
                )

                intensity_grids[label][local_j] = box_result['intensity_asymmetry']
                com_grids[label][local_j]        = box_result['com_y_displacement']

                recorded = (box_result['intensity_asymmetry'] if score_type == 'intensity'
                            else box_result['com_y_displacement'])
                asymmetry_grids[label][local_j]     = recorded
                box_coords_box1[label][local_j]     = box_result['box1_coords']
                box_coords_box2[label][local_j]     = box_result['box2_coords']

                y1_min, y1_max, x1_min, x1_max = box_result['box1_coords']
                y2_min, y2_max, x2_min, x2_max = box_result['box2_coords']
                ax_b.add_patch(Rectangle((x1_min, y1_min), x1_max - x1_min, y1_max - y1_min,
                                         linewidth=2, edgecolor='red', facecolor='none'))
                ax_b.add_patch(Rectangle((x2_min, y2_min), x2_max - x2_min, y2_max - y2_min,
                                         linewidth=2, edgecolor='blue', facecolor='none'))
                ax_b.plot(box_result['box1_com_x'], box_result['box1_com_y'],
                          'r+', markersize=15, markeredgewidth=2)
                ax_b.plot(box_result['box2_com_x'], box_result['box2_com_y'],
                          'b+', markersize=15, markeredgewidth=2)
                ax_b.plot(box_result['midpoint_x'], box_result['midpoint_y'],
                          'go', markersize=10, markeredgewidth=2, fillstyle='none')

                print(f"    {label}: COM-Y={box_result['com_y_displacement']:+.6f}%  "
                      f"intens_asym={box_result['intensity_asymmetry']:+.7f}")

            plt.tight_layout()
            plt.savefig(os.path.join(sim_images_boxes_dir, img_filename),
                        dpi=150, bbox_inches='tight')
            plt.close(fig_b)

        except Exception as e:
            print(f"    ERROR at tilt_y={tilt_y:.4f}: {e}")

        # ---- Checkpoint save after every tilt --------------------------------
        _save_partial_npz(
            partial_npz, task_id, tilts_per_job,
            tilt_x_values, tilt_y_subset, global_indices,
            diffraction_images, asymmetry_grids, intensity_grids,
            com_grids, box_coords_box1, box_coords_box2, _img_shape,
        )
        print(f"    Checkpoint saved -> {partial_npz}")

    print(f"\nTask {task_id} complete. Final NPZ: {partial_npz}")


def _save_partial_npz(
    path, task_id, tilts_per_job,
    tilt_x_values, tilt_y_subset, global_indices,
    diffraction_images, asymmetry_grids, intensity_grids,
    com_grids, box_coords_box1, box_coords_box2, img_shape,
):
    """Assemble arrays and write the partial NPZ for this task."""
    n = len(tilt_y_subset)

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
        task_id=np.array(task_id),
        tilts_per_job=np.array(tilts_per_job),
        tilt_x_values=tilt_x_values,
        tilt_y_values=tilt_y_subset,
        global_tilt_indices=global_indices,
        diffraction_images=img_array,
    )
    for label in asymmetry_grids:
        save_dict[f'{label}']                      = asymmetry_grids[label]
        save_dict[f'{label}_intensity_asymmetry']  = intensity_grids[label]
        save_dict[f'{label}_com_y_displacement']   = com_grids[label]
        save_dict[f'{label}_box1_coords']          = box_coords_box1[label]
        save_dict[f'{label}_box2_coords']          = box_coords_box2[label]

    np.savez(path, **save_dict)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a tilt-angle subset for BTO asymmetry analysis.')
    parser.add_argument('--task_id',      type=int, required=True,
                        help='SLURM array task ID (0-indexed).')
    parser.add_argument('--tilts_per_job', type=int, default=1,
                        help='Number of tilt angles per job (default: 1).')
    parser.add_argument('--run_id',       type=str, default=None,
                        help='Shared run identifier (use $SLURM_ARRAY_JOB_ID). '
                             'Defaults to SLURM_ARRAY_JOB_ID env var or a timestamp.')
    args = parser.parse_args()

    # Resolve run_id
    if args.run_id is None:
        args.run_id = os.environ.get(
            'SLURM_ARRAY_JOB_ID',
            datetime.now().strftime('%Y%m%d_%H%M%S'),
        )

    # ---- Box coordinates (match non-batch_intensity_asymmetry_calc.py) -----
    kikuchi_box1 = ((200 * 4, 350 * 4), (497 * 4, 527 * 4))
    kikuchi_box2 = ((674 * 4, 824 * 4), (497 * 4, 527 * 4))
    kikuchi_box_pairs = {'horizontal_boxes': (kikuchi_box1, kikuchi_box2)}

    # ---- Active material: BTO ----------------------------------------------
    # To switch material, comment out the BTO block and uncomment the desired one.

    structure_name    = 'BTO'
    supercell_size    = (30, 30, 154)
    num_phonons       = 40
    detector_max_angle = 'full'
    score_type        = 'com'
    box_pairs         = kikuchi_box_pairs
    base_output_dir   = '/leonardo_work/EUHPC_E06_088/fhardy00/cbed_simulations/BTO_asymmetry_sweeps'

    # ---- PTO (commented out for future use) --------------------------------
    # structure_name    = 'PTO'
    # supercell_size    = (30, 30, 154)
    # num_phonons       = 40
    # detector_max_angle = 'full'
    # score_type        = 'com'
    # box_pairs         = kikuchi_box_pairs
    # base_output_dir   = '/leonardo_work/EUHPC_E06_088/fhardy00/cbed_simulations/PTO_asymmetry_sweeps'

    # ---- Boracite (commented out for future use) ---------------------------
    # structure_name    = 'boracite'
    # supercell_size    = (7, 7, 40)
    # num_phonons       = 40
    # detector_max_angle = 'full'
    # score_type        = 'com'
    # box_pairs         = kikuchi_box_pairs
    # base_output_dir   = '/leonardo_work/EUHPC_E06_088/fhardy00/cbed_simulations/boracite_asymmetry_sweeps'

    # ---- Build output directory (shared across all tasks in this array) ----
    run_dir = os.path.join(
        base_output_dir,
        f'{structure_name}_tilt_split_job{args.run_id}',
    )
    os.makedirs(run_dir, exist_ok=True)

    # Write parameters.txt — all tasks produce identical content so concurrent
    # writes are safe; no existence check needed (avoids TOCTOU race).
    params_file = os.path.join(run_dir, 'parameters.txt')
    with open(params_file, 'w') as f:
            f.write("Tilt-Split Simulation Parameters\n")
            f.write("=" * 60 + "\n")
            f.write(f"Structure     : {structure_name}\n")
            f.write(f"Supercell     : {supercell_size}\n")
            f.write(f"Grid points   : {GRID_POINTS}\n")
            f.write(f"Scan pixels   : {SCAN_PIXELS_X}x{SCAN_PIXELS_Y}\n")
            f.write(f"Beam energy   : {BEAM_ENERGY} eV\n")
            f.write(f"Conv. angle   : {CONVERGENCE_ANGLE} mrad\n")
            f.write(f"Detector angle: {detector_max_angle}\n")
            f.write(f"Slice thick.  : {SLICE_THICKNESS} A\n")
            f.write(f"Num phonons   : {num_phonons}\n")
            f.write(f"Score type    : {score_type}\n")
            f.write(f"Tilt Y range  : {TILT_Y_MIN} to {TILT_Y_MAX} deg\n")
            f.write(f"Tilt Y points : {TILT_Y_NPTS}\n")
            f.write(f"Tilt X fixed  : {TILT_X_FIXED} deg\n")
            f.write(f"Tilts per job : {args.tilts_per_job}\n")
            f.write(f"Run ID        : {args.run_id}\n")

    run_tilt_subset(
        task_id=args.task_id,
        tilts_per_job=args.tilts_per_job,
        run_dir=run_dir,
        structure_name=structure_name,
        supercell_size=supercell_size,
        detector_max_angle=detector_max_angle,
        box_pairs_dict=box_pairs,
        num_phonons=num_phonons,
        score_type=score_type,
    )
