"""
Compile Tilt-Split Partial Results into a Final NPZ

Reads all tilt_task_*.npz files from a tilt-split run directory, reassembles
them in tilt-angle order, and writes a final asymmetry_data.npz whose format
matches the output of non-batch_intensity_asymmetry_calc.py.  Also generates
a line plot of COM Y-displacement vs tilt angle.

Usage:
    python compile_tilt_results.py --run_dir PATH [--output_dir PATH]

Arguments:
    --run_dir PATH    Path to the split-job run folder, e.g.
                          kikuchi_simulations/BTO_asymmetry_sweeps/BTO_tilt_split_job12345/
                      The script searches this folder's partial_results/ subdirectory.
    --output_dir PATH Where to write the compiled outputs.  Defaults to a new
                      timestamped subfolder inside --run_dir.
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

TILT_Y_MIN = -2.5
TILT_Y_MAX =  2.5
TILT_Y_NPTS = 81
TILT_X_FIXED = 0.0

GAMMA = 0.3
PERCENTILE_MIN = 0.5
PERCENTILE_MAX = 99.8


def load_partial_npzs(partial_dir):
    """
    Load every tilt_task_*.npz file found in partial_dir.

    Returns a list of dicts, each containing the file's arrays plus the
    derived 'tilt_y_values' and 'global_tilt_indices'.
    """
    pattern = os.path.join(partial_dir, 'tilt_task_*.npz')
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No tilt_task_*.npz files found in {partial_dir}")

    print(f"Found {len(files)} partial NPZ file(s) in {partial_dir}")

    records = []
    for fpath in files:
        try:
            data = np.load(fpath, allow_pickle=False)
            record = {k: data[k] for k in data.files}
            record['_source_file'] = fpath
            records.append(record)
            print(f"  Loaded: {os.path.basename(fpath)}  "
                  f"(tilt_y={record['tilt_y_values']})")
        except (EOFError, Exception) as e:
            print(f"  SKIPPED (corrupt/incomplete): {os.path.basename(fpath)} — {e}")

    return records


def check_coverage(records):
    """
    Print a coverage report showing which of the 81 tilt indices are present,
    which are duplicated, and which are missing.

    Returns a sorted list of (global_idx, record) pairs for all completed tilts.
    """
    all_tilt_y = np.linspace(TILT_Y_MIN, TILT_Y_MAX, TILT_Y_NPTS)
    covered = {}  # global_idx -> record

    for rec in records:
        indices = rec['global_tilt_indices']
        tilt_ys = rec['tilt_y_values']
        for local_j, (gidx, ty) in enumerate(zip(indices, tilt_ys)):
            gidx = int(gidx)
            if gidx in covered:
                print(f"  WARNING: tilt index {gidx} (tilt_y={ty:.4f}) appears in multiple "
                      f"files — keeping the later one.")
            covered[gidx] = (rec, local_j)

    missing = [i for i in range(TILT_Y_NPTS) if i not in covered]
    if missing:
        print(f"\n  WARNING: {len(missing)}/{TILT_Y_NPTS} tilt indices are MISSING:")
        for idx in missing:
            print(f"    idx={idx:2d}  tilt_y={all_tilt_y[idx]:.4f}°")
    else:
        print(f"\n  All {TILT_Y_NPTS} tilt indices are present.")

    sorted_pairs = sorted(covered.items())  # [(global_idx, (rec, local_j)), ...]
    return sorted_pairs, missing


def assemble_arrays(sorted_pairs, label_keys):
    """
    Concatenate data from all partial NPZs in global-index order.

    Returns:
        tilt_y_values      : (N,)
        global_indices     : (N,)
        diffraction_images : (N, H, W)  — may include NaN slices for failed sims
        asymmetry_grids    : {label: (N,)}
        intensity_grids    : {label: (N,)}
        com_grids          : {label: (N,)}
        box_coords_box1    : {label: (N, 4)}
        box_coords_box2    : {label: (N, 4)}
    """
    tilt_y_list    = []
    global_idx_list = []
    img_list       = []
    asym_lists  = {k: [] for k in label_keys}
    intens_lists= {k: [] for k in label_keys}
    com_lists   = {k: [] for k in label_keys}
    bc1_lists   = {k: [] for k in label_keys}
    bc2_lists   = {k: [] for k in label_keys}

    for gidx, (rec, local_j) in sorted_pairs:
        tilt_y_list.append(float(rec['tilt_y_values'][local_j]))
        global_idx_list.append(gidx)

        imgs = rec.get('diffraction_images', np.array([]))
        if imgs.ndim >= 3 and local_j < imgs.shape[0]:
            img_list.append(imgs[local_j])
        else:
            img_list.append(None)

        for k in label_keys:
            asym_lists[k].append(float(rec[k][local_j]))
            intens_lists[k].append(float(rec[f'{k}_intensity_asymmetry'][local_j]))
            com_lists[k].append(float(rec[f'{k}_com_y_displacement'][local_j]))
            bc1 = rec[f'{k}_box1_coords']
            bc2 = rec[f'{k}_box2_coords']
            bc1_lists[k].append(bc1[local_j] if bc1.ndim > 1 else bc1)
            bc2_lists[k].append(bc2[local_j] if bc2.ndim > 1 else bc2)

    # Build image array
    valid_imgs = [x for x in img_list if x is not None]
    if valid_imgs:
        h, w = valid_imgs[0].shape
        img_array = np.zeros((len(img_list), h, w))
        for j, img in enumerate(img_list):
            img_array[j] = img if img is not None else np.nan
    else:
        img_array = np.array([])

    asym_arrays   = {k: np.array(v) for k, v in asym_lists.items()}
    intens_arrays = {k: np.array(v) for k, v in intens_lists.items()}
    com_arrays    = {k: np.array(v) for k, v in com_lists.items()}
    bc1_arrays    = {k: np.array(v) for k, v in bc1_lists.items()}
    bc2_arrays    = {k: np.array(v) for k, v in bc2_lists.items()}

    return (
        np.array(tilt_y_list),
        np.array(global_idx_list),
        img_array,
        asym_arrays, intens_arrays, com_arrays,
        bc1_arrays, bc2_arrays,
    )


def save_final_npz(output_dir, tilt_y_values, img_array,
                   asym_arrays, intens_arrays, com_arrays,
                   bc1_arrays, bc2_arrays):
    """
    Save asymmetry_data.npz in the same format as non-batch_intensity_asymmetry_calc.py.

    The non-batch script uses shape (points_y, points_x) = (81, 1) arrays.
    We reshape our (N,) vectors to (N, 1) so the format matches exactly.
    """
    tilt_x_values = np.array([TILT_X_FIXED])
    N = len(tilt_y_values)

    save_dict = dict(
        tilt_x_values=tilt_x_values,
        tilt_y_values=tilt_y_values,
    )

    # Reshape diffraction images: (N, H, W) -> (N, 1, H, W)
    if img_array.ndim == 3:
        save_dict['diffraction_images'] = img_array[:, np.newaxis, :, :]
    else:
        save_dict['diffraction_images'] = img_array

    for k in asym_arrays:
        save_dict[k]                          = asym_arrays[k].reshape(N, 1)
        save_dict[f'{k}_intensity_asymmetry'] = intens_arrays[k].reshape(N, 1)
        save_dict[f'{k}_com_y_displacement']  = com_arrays[k].reshape(N, 1)
        save_dict[f'{k}_box1_coords']         = bc1_arrays[k].reshape(N, 1, -1)
        save_dict[f'{k}_box2_coords']         = bc2_arrays[k].reshape(N, 1, -1)

    out_path = os.path.join(output_dir, 'asymmetry_data.npz')
    np.savez(out_path, **save_dict)
    print(f"\nFinal NPZ saved: {out_path}")
    return out_path


def make_line_plots(output_dir, tilt_y_values, asym_arrays, intens_arrays, com_arrays,
                    structure_name='BTO'):
    """Generate line plots (COM Y-displacement and intensity asymmetry vs tilt Y)."""
    for metric_name, metric_arrays in [('com_y_displacement', com_arrays),
                                        ('intensity_asymmetry', intens_arrays)]:
        label_keys = list(metric_arrays.keys())
        n_plots = len(label_keys)
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        for ax, key in zip(axes, label_keys):
            values = metric_arrays[key]

            # Replace NaN with masked array for clean plotting
            valid = ~np.isnan(values)
            x = tilt_y_values[valid]
            y = values[valid]

            ax.plot(x, y, 'o-', linewidth=2, markersize=6, color='#2E86AB')
            ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
            ax.fill_between(x, 0, y, where=(y >= 0), alpha=0.18, color='red')
            ax.fill_between(x, 0, y, where=(y < 0),  alpha=0.18, color='blue')

            ylabel = ('COM Y Displacement (%)' if metric_name == 'com_y_displacement'
                      else 'Intensity Asymmetry (box1 - box2)')
            ax.set_xlabel('Tilt Y (degrees)', fontsize=13, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
            ax.set_title(f'{structure_name} — {key}\n(Tilt X = {TILT_X_FIXED}°)',
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            # Mark any NaN/missing tilts as vertical grey bars
            missing_y = tilt_y_values[~valid]
            for my in missing_y:
                ax.axvline(my, color='gray', linestyle='--', alpha=0.4, linewidth=1)

            if len(missing_y):
                ax.text(0.01, 0.99, f'{len(missing_y)} missing tilt(s)',
                        transform=ax.transAxes, fontsize=9, color='gray',
                        va='top', ha='left')

        plt.tight_layout()
        fname = f"lineplot_{structure_name}_{metric_name}_compiled.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Line plot saved: {fpath}")


def discover_label_keys(records):
    """Infer the box-pair label keys from the first partial NPZ."""
    rec = records[0]
    keys = []
    for k in rec.keys():
        if k.endswith('_com_y_displacement'):
            base = k[: -len('_com_y_displacement')]
            keys.append(base)
    return keys if keys else ['horizontal_boxes']


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile tilt-split partial results.')
    parser.add_argument('--run_dir',    type=str, required=True,
                        help='Path to the split-job run folder '
                             '(e.g. kikuchi_simulations/BTO_asymmetry_sweeps/BTO_tilt_split_job12345).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write compiled outputs. '
                             'Defaults to a compiled_TIMESTAMP subfolder inside --run_dir.')
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip('/')
    partial_dir = os.path.join(run_dir, 'partial_results')

    if not os.path.isdir(partial_dir):
        print(f"ERROR: partial_results/ not found under {run_dir}")
        sys.exit(1)

    # ---- Load partial results -----------------------------------------------
    records = load_partial_npzs(partial_dir)
    label_keys = discover_label_keys(records)
    print(f"\nDetected label keys: {label_keys}")

    # ---- Coverage report ----------------------------------------------------
    print("\n--- Coverage Report ---")
    sorted_pairs, missing_indices = check_coverage(records)

    if not sorted_pairs:
        print("ERROR: No valid tilt data found. Exiting.")
        sys.exit(1)

    # ---- Assemble arrays ----------------------------------------------------
    print("\n--- Assembling Arrays ---")
    (tilt_y_values, global_indices, img_array,
     asym_arrays, intens_arrays, com_arrays,
     bc1_arrays, bc2_arrays) = assemble_arrays(sorted_pairs, label_keys)

    print(f"Assembled {len(tilt_y_values)} tilt points.")

    # ---- Output directory ---------------------------------------------------
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(run_dir, f'compiled_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # ---- Infer structure name from run_dir for plot titles ------------------
    dirname = os.path.basename(run_dir)
    structure_name = 'BTO'
    for candidate in ('PTO', 'BTO', 'boracite'):
        if dirname.startswith(candidate):
            structure_name = candidate
            break

    # ---- Save final NPZ -----------------------------------------------------
    save_final_npz(
        output_dir, tilt_y_values, img_array,
        asym_arrays, intens_arrays, com_arrays,
        bc1_arrays, bc2_arrays,
    )

    # ---- Copy parameters.txt if present -------------------------------------
    src_params = os.path.join(run_dir, 'parameters.txt')
    if os.path.isfile(src_params):
        import shutil
        dst_params = os.path.join(output_dir, 'parameters.txt')
        shutil.copy2(src_params, dst_params)
        print(f"Copied parameters.txt -> {dst_params}")

    # Append compilation metadata
    with open(os.path.join(output_dir, 'parameters.txt'), 'a') as f:
        f.write("\n--- Compilation ---\n")
        f.write(f"Compiled at     : {datetime.now().isoformat()}\n")
        f.write(f"Tilt points used: {len(tilt_y_values)}/{TILT_Y_NPTS}\n")
        f.write(f"Missing indices : {missing_indices if missing_indices else 'none'}\n")
        f.write(f"Source run_dir  : {os.path.abspath(run_dir)}\n")

    # ---- Generate plots -----------------------------------------------------
    print("\n--- Generating Plots ---")
    make_line_plots(output_dir, tilt_y_values, asym_arrays, intens_arrays,
                    com_arrays, structure_name=structure_name)

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPILATION COMPLETE")
    print("=" * 60)
    print(f"  Tilt points compiled : {len(tilt_y_values)}/{TILT_Y_NPTS}")
    if missing_indices:
        print(f"  Missing indices      : {missing_indices}")
        print("  (NaN rows inserted for missing tilts in the NPZ.)")
    print(f"  Output directory     : {os.path.abspath(output_dir)}")
    print("=" * 60)
