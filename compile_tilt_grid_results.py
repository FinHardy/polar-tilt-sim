"""
Compile Tilt-Grid Partial Results into a Final NPZ

Reads all tilt_task_*.npz files from a tilt_grid_sim run directory, assembles
them into a 2D (tilt_y, tilt_x) grid, and writes a final asymmetry_data_grid.npz.
Also generates heatmap plots of COM Y-displacement and intensity asymmetry.

Usage:
    python compile_tilt_grid_results.py --run_dir PATH [--output_dir PATH]
                                        [--n_tilt_y INT] [--n_tilt_x INT]

Arguments:
    --run_dir PATH    Path to the grid-job run folder, e.g.
                          BTO_asymmetry_sweeps/BTO_tilt_grid_job12345/
    --output_dir PATH Where to write compiled outputs.  Defaults to a
                      compiled_TIMESTAMP subfolder inside --run_dir.
    --n_tilt_y INT    Number of tilt_y grid points (default: 7).
    --n_tilt_x INT    Number of tilt_x grid points (default: 7).
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

GAMMA = 0.3
PERCENTILE_MIN = 0.5
PERCENTILE_MAX = 99.8


def load_partial_npzs(partial_dir):
    """Load every tilt_task_*.npz from partial_dir. Returns list of dicts."""
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
                  f"task={int(record['task_id'])}  "
                  f"tilt_y={float(record['tilt_y']):.4f}°  "
                  f"tilt_x={float(record['tilt_x']):.4f}°")
        except Exception as e:
            print(f"  SKIPPED (corrupt/incomplete): {os.path.basename(fpath)} — {e}")

    return records


def check_coverage(records, n_tilt_y, n_tilt_x):
    """
    Report 2D coverage. Returns dict keyed by (iy, ix) -> record,
    and lists of missing (iy, ix) pairs.
    """
    total = n_tilt_y * n_tilt_x
    covered = {}

    for rec in records:
        task_id = int(rec['task_id'])
        iy = int(rec.get('tilt_y_idx', task_id // n_tilt_x))
        ix = int(rec.get('tilt_x_idx', task_id % n_tilt_x))
        key = (iy, ix)
        if key in covered:
            print(f"  WARNING: grid point ({iy},{ix}) appears in multiple files — keeping later one.")
        covered[key] = rec

    missing = [(iy, ix) for iy in range(n_tilt_y) for ix in range(n_tilt_x)
               if (iy, ix) not in covered]

    if missing:
        print(f"\n  WARNING: {len(missing)}/{total} grid points are MISSING:")
        for iy, ix in missing:
            print(f"    iy={iy}  ix={ix}  task_id={iy * n_tilt_x + ix}")
    else:
        print(f"\n  All {total} grid points are present.")

    return covered, missing


def assemble_grid(covered, n_tilt_y, n_tilt_x, label_keys):
    """
    Build 2D arrays (n_tilt_y, n_tilt_x) for each metric.

    Returns:
        tilt_y_values   : (n_tilt_y,)
        tilt_x_values   : (n_tilt_x,)
        img_array       : (n_tilt_y, n_tilt_x, H, W)  or None
        asym_grid, intens_grid, com_grid : {label: (n_tilt_y, n_tilt_x)}
        bc1_grid, bc2_grid               : {label: (n_tilt_y, n_tilt_x, 4)}
    """
    # Collect tilt axis values from data
    tilt_y_vals = np.full(n_tilt_y, np.nan)
    tilt_x_vals = np.full(n_tilt_x, np.nan)

    asym_grid   = {k: np.full((n_tilt_y, n_tilt_x), np.nan) for k in label_keys}
    intens_grid = {k: np.full((n_tilt_y, n_tilt_x), np.nan) for k in label_keys}
    com_grid    = {k: np.full((n_tilt_y, n_tilt_x), np.nan) for k in label_keys}
    bc1_grid    = {k: np.full((n_tilt_y, n_tilt_x, 4), np.nan) for k in label_keys}
    bc2_grid    = {k: np.full((n_tilt_y, n_tilt_x, 4), np.nan) for k in label_keys}

    # First pass: figure out image shape
    img_shape = None
    for rec in covered.values():
        img = rec.get('diffraction_image', np.array([]))
        if img.ndim == 2 and img.size > 0:
            img_shape = img.shape
            break

    if img_shape is not None:
        img_array = np.full((n_tilt_y, n_tilt_x) + img_shape, np.nan)
    else:
        img_array = None

    for (iy, ix), rec in covered.items():
        tilt_y_vals[iy] = float(rec['tilt_y'])
        tilt_x_vals[ix] = float(rec['tilt_x'])

        if img_array is not None:
            img = rec.get('diffraction_image', np.array([]))
            if img.ndim == 2 and img.shape == img_shape:
                img_array[iy, ix] = img

        for k in label_keys:
            asym_grid[k][iy, ix]    = float(rec[k])
            intens_grid[k][iy, ix]  = float(rec[f'{k}_intensity_asymmetry'])
            com_grid[k][iy, ix]     = float(rec[f'{k}_com_y_displacement'])
            bc1 = rec[f'{k}_box1_coords']
            bc2 = rec[f'{k}_box2_coords']
            if bc1.shape == (4,):
                bc1_grid[k][iy, ix] = bc1
                bc2_grid[k][iy, ix] = bc2

    return tilt_y_vals, tilt_x_vals, img_array, asym_grid, intens_grid, com_grid, bc1_grid, bc2_grid


def save_final_npz(output_dir, tilt_y_values, tilt_x_values, img_array,
                   asym_grid, intens_grid, com_grid, bc1_grid, bc2_grid):
    save_dict = dict(
        tilt_y_values=tilt_y_values,
        tilt_x_values=tilt_x_values,
    )

    if img_array is not None:
        save_dict['diffraction_images'] = img_array

    for k in asym_grid:
        save_dict[k]                          = asym_grid[k]
        save_dict[f'{k}_intensity_asymmetry'] = intens_grid[k]
        save_dict[f'{k}_com_y_displacement']  = com_grid[k]
        save_dict[f'{k}_box1_coords']         = bc1_grid[k]
        save_dict[f'{k}_box2_coords']         = bc2_grid[k]

    out_path = os.path.join(output_dir, 'asymmetry_data_grid.npz')
    np.savez(out_path, **save_dict)
    print(f"\nFinal NPZ saved: {out_path}")
    return out_path


def make_heatmap_plots(output_dir, tilt_y_values, tilt_x_values,
                       asym_grid, intens_grid, com_grid, structure_name='BTO'):
    """Generate 2D heatmap plots for each metric and box-pair label."""
    for metric_name, metric_grids in [('com_y_displacement', com_grid),
                                       ('intensity_asymmetry', intens_grid)]:
        for label, grid in metric_grids.items():
            fig, ax = plt.subplots(figsize=(7, 6))

            # Use actual tilt values as axis ticks where available
            ty = tilt_y_values.copy()
            tx = tilt_x_values.copy()
            # Fill any missing axis values by interpolation for display only
            ty_nan = np.isnan(ty)
            tx_nan = np.isnan(tx)
            if ty_nan.any():
                idx = np.arange(len(ty))
                ty[ty_nan] = np.interp(idx[ty_nan], idx[~ty_nan], ty[~ty_nan])
            if tx_nan.any():
                idx = np.arange(len(tx))
                tx[tx_nan] = np.interp(idx[tx_nan], idx[~tx_nan], tx[~tx_nan])

            vmax = np.nanmax(np.abs(grid))
            vmin_plot = -vmax if vmax > 0 else -1

            im = ax.imshow(
                grid,
                origin='lower',
                aspect='auto',
                extent=[tx[0], tx[-1], ty[0], ty[-1]],
                cmap='RdBu_r',
                vmin=vmin_plot,
                vmax=vmax,
            )

            plt.colorbar(im, ax=ax, label=metric_name.replace('_', ' ').title())

            ylabel = 'COM Y Displacement (%)' if metric_name == 'com_y_displacement' else 'Intensity Asymmetry'
            ax.set_xlabel('Tilt X (degrees)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Tilt Y (degrees)', fontsize=12, fontweight='bold')
            ax.set_title(f'{structure_name} — {label}\n{metric_name.replace("_", " ").title()}',
                         fontsize=13, fontweight='bold')

            plt.tight_layout()
            fname = f"heatmap_{structure_name}_{label}_{metric_name}.png"
            fpath = os.path.join(output_dir, fname)
            plt.savefig(fpath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Heatmap saved: {fpath}")


def discover_label_keys(records):
    """Infer box-pair label keys from the first partial NPZ."""
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
    parser = argparse.ArgumentParser(description='Compile 2D tilt-grid partial results.')
    parser.add_argument('--run_dir',   type=str, required=True,
                        help='Path to the grid-job run folder.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to write compiled outputs.')
    parser.add_argument('--n_tilt_y',  type=int, default=7,
                        help='Number of tilt_y grid points (default: 7).')
    parser.add_argument('--n_tilt_x',  type=int, default=7,
                        help='Number of tilt_x grid points (default: 7).')
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
    covered, missing = check_coverage(records, args.n_tilt_y, args.n_tilt_x)

    if not covered:
        print("ERROR: No valid grid data found. Exiting.")
        sys.exit(1)

    # ---- Assemble 2D arrays -------------------------------------------------
    print("\n--- Assembling Grid ---")
    (tilt_y_values, tilt_x_values, img_array,
     asym_grid, intens_grid, com_grid,
     bc1_grid, bc2_grid) = assemble_grid(covered, args.n_tilt_y, args.n_tilt_x, label_keys)

    print(f"Assembled {len(covered)}/{args.n_tilt_y * args.n_tilt_x} grid points.")

    # ---- Output directory ---------------------------------------------------
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(run_dir, f'compiled_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # ---- Infer structure name from run_dir ----------------------------------
    dirname = os.path.basename(run_dir)
    structure_name = 'BTO'
    for candidate in ('PTO', 'BTO', 'boracite'):
        if dirname.startswith(candidate):
            structure_name = candidate
            break

    # ---- Save final NPZ -----------------------------------------------------
    save_final_npz(
        output_dir, tilt_y_values, tilt_x_values, img_array,
        asym_grid, intens_grid, com_grid, bc1_grid, bc2_grid,
    )

    # ---- Copy parameters.txt ------------------------------------------------
    src_params = os.path.join(run_dir, 'parameters.txt')
    if os.path.isfile(src_params):
        import shutil
        dst_params = os.path.join(output_dir, 'parameters.txt')
        shutil.copy2(src_params, dst_params)
        print(f"Copied parameters.txt -> {dst_params}")

    with open(os.path.join(output_dir, 'parameters.txt'), 'a') as f:
        f.write("\n--- Compilation ---\n")
        f.write(f"Compiled at    : {datetime.now().isoformat()}\n")
        f.write(f"Grid points    : {len(covered)}/{args.n_tilt_y * args.n_tilt_x}\n")
        f.write(f"Missing points : {missing if missing else 'none'}\n")
        f.write(f"Source run_dir : {os.path.abspath(run_dir)}\n")

    # ---- Generate heatmap plots ---------------------------------------------
    print("\n--- Generating Heatmaps ---")
    make_heatmap_plots(output_dir, tilt_y_values, tilt_x_values,
                       asym_grid, intens_grid, com_grid, structure_name=structure_name)

    # ---- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPILATION COMPLETE")
    print("=" * 60)
    print(f"  Grid points compiled : {len(covered)}/{args.n_tilt_y * args.n_tilt_x}")
    if missing:
        print(f"  Missing points       : {missing}")
    print(f"  Output directory     : {os.path.abspath(output_dir)}")
    print("=" * 60)
