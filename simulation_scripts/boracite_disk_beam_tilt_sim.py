"""
Disk Intensity Polarisation Mapping — Boracite, BEAM TILT

Same simulation as boracite_disk_intensity_sim.py (same supercell, zone axis,
probe, detector, phonons, tilt range), except the tilt is applied to the beam
(abTEM Probe(tilt=...), i.e. a tilted Fresnel propagator) instead of rotating
the atoms. The specimen therefore stays fixed and only the illumination moves.

Tilt convention: rotating the crystal by +theta about y is the same geometry as
rotating the beam by -theta about y in the crystal frame, so a crystal tilt of
tilt_y (deg) is reproduced by beam tilt_x = -tilt_y in mrad (abTEM x = array
axis 0 = the measured/polar axis). The two curves can then be compared tilt for
tilt. The analysis also reports the correlation against the mirrored CIF curve
as a check on this sign.

abTEM's propagator tilt leaves the direct beam at k = 0, so the pattern stays
centred and the disk boxes need no shift with tilt, exactly as for the CIF tilt.

Stages (one file):
  1. simulate   -> {out_dir}/results.npz, checkpointed after every tilt and
                   resumed automatically if re-run with the same --out_dir
  2. analyse    -> disk-pair asymmetry (notebook metric, same boxes) for
                   CIF tilt vs beam tilt, figure + table

Usage:
  python simulation_scripts/boracite_disk_beam_tilt_sim.py
  python simulation_scripts/boracite_disk_beam_tilt_sim.py --analyse_only
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ============================================================================
# CONFIGURATION PARAMETERS (copied from boracite_disk_intensity_sim.py)
# ============================================================================

STRUCTURE_FILE = os.path.join(ROOT, "cifs/cu2-ortho-sym.cif")

BEAM_ENERGY = 300e3  # eV
CONVERGENCE_ANGLE = 0.8  # mrad
SLICE_THICKNESS = 1.0  # Angstroms
GRID_POINTS = 1024

SCAN_PIXELS_X = 2
SCAN_PIXELS_Y = 2

TILT_Y_MIN = -0.25  # degrees
TILT_Y_MAX = 0.25  # degrees
TILT_Y_NPTS = 81

ZONE_AXIS = [1, 1, 0]
THERMAL_SIGMA = 0.1
supercell_size = (7, 7, 40)
num_phonons = 10
DISK_DETECTOR_MAX_ANGLE = 35  # mrad

# CIF-tilt reference run and the boracite disk boxes from
# create_paper_figs_disks_intensity_MORE_DISKS.ipynb (box1 only; box2 is the mirror)
CIF_TILT_NPZ = os.path.join(ROOT, "data/boracite_disks.npz")
BOXES = [
    ("100", "1x1", ((267, 283), (293, 313))),
    ("100", "1x3", ((267, 283), (266, 340))),
    ("100", "1x5", ((267, 284), (236, 367))),
    ("200", "1x1", ((240, 255), (293, 313))),
    ("200", "1x3", ((240, 255), (266, 340))),
    ("200", "1x5", ((241, 255), (237, 371))),
]
DEFAULT_OUT_DIR = os.path.join(ROOT, "data/beam_tilt/boracite_disks_beam_tilt")
FIG_DIR = os.path.join(ROOT, "figure_notebooks/figures")


# ============================================================================
# SIMULATION
# ============================================================================


def build_supercell():
    """Identical to the CIF-tilt script, without the per-tilt rotation."""
    import abtem
    from ase.io import read

    atoms = read(STRUCTURE_FILE)
    zone = np.array(ZONE_AXIS, dtype=float)
    zone /= np.linalg.norm(zone)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(zone, z)
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        angle = np.degrees(np.arccos(np.clip(zone @ z, -1.0, 1.0)))
        atoms.rotate(angle, axis, center="COP", rotate_cell=True)

    atoms = abtem.orthogonalize_cell(atoms)
    supercell = atoms * supercell_size
    supercell.rotate(-45, "z", center="COP", rotate_cell=True)
    return supercell


def make_potential(supercell):
    import abtem

    # one phonon ensemble reused for every tilt (the specimen never moves),
    # so tilt-to-tilt changes are pure beam tilt; the CIF run drew fresh configs per tilt.
    frozen = abtem.FrozenPhonons(supercell, sigmas=THERMAL_SIGMA, num_configs=num_phonons, seed=0)
    potential = abtem.Potential(
        frozen, gpts=GRID_POINTS, slice_thickness=SLICE_THICKNESS, projection="finite"
    )
    ex, ey = potential.extent
    if abs(ex - ey) > 0.1:
        potential = abtem.Potential(
            frozen,
            gpts=GRID_POINTS,
            sampling=potential.sampling[0],
            slice_thickness=SLICE_THICKNESS,
            projection="finite",
        )
    return potential


def simulate_pattern(potential, beam_tilt_mrad):
    import abtem

    probe = abtem.Probe(
        energy=BEAM_ENERGY,
        semiangle_cutoff=CONVERGENCE_ANGLE,
        gpts=GRID_POINTS,
        sampling=potential.sampling,
        defocus=0,
        tilt=beam_tilt_mrad,
    )
    scan = abtem.GridScan(
        start=[0, 0], end=list(potential.extent), gpts=(SCAN_PIXELS_X, SCAN_PIXELS_Y)
    )
    detector = abtem.PixelatedDetector(max_angle=DISK_DETECTOR_MAX_ANGLE, reciprocal_space=True)
    result = probe.scan(scan=scan, potential=potential, detectors=detector).compute(
        progress_bar=False
    )
    return np.asarray(result.array).mean(axis=(0, 1))


def setup_gpu():
    import abtem

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
        return cp
    except Exception as e:
        print(f"GPU not available - using CPU: {e}")
        abtem.config.set({"device": "cpu"})
        return None


def run_sweep(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.npz")
    tilt_y = np.linspace(TILT_Y_MIN, TILT_Y_MAX, TILT_Y_NPTS)
    beam_tilt_x_mrad = -np.radians(tilt_y) * 1e3

    images, done = None, np.zeros(len(tilt_y), bool)
    if os.path.exists(path):
        d = np.load(path)
        if np.allclose(d["tilt_y_values"], tilt_y):
            images, done = d["diffraction_images"], d["done"]
            print(f"Resuming: {done.sum()}/{len(tilt_y)} tilts already in {path}")
        else:
            sys.exit(f"{path} has a different tilt grid; use another --out_dir")
    if done.all():
        return path

    cp = setup_gpu()
    supercell = build_supercell()
    print(f"Supercell {supercell_size}: {len(supercell)} atoms, cell={supercell.cell.lengths()}")
    potential = make_potential(supercell)

    for j in np.flatnonzero(~done):
        t0 = time.time()
        pattern = simulate_pattern(potential, (beam_tilt_x_mrad[j], 0.0))
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        if images is None:
            images = np.full((len(tilt_y),) + pattern.shape, np.nan)
        images[j], done[j] = pattern, True

        # write-then-rename so a kill mid-save never corrupts the checkpoint
        tmp = path[:-4] + ".tmp.npz"
        np.savez(
            tmp,
            tilt_y_values=tilt_y,
            beam_tilt_x_mrad=beam_tilt_x_mrad,
            diffraction_images=images,
            done=done,
            grid_points=GRID_POINTS,
            num_phonons=num_phonons,
        )
        os.replace(tmp, path)
        print(
            f"[{done.sum()}/{len(tilt_y)}] tilt_y={tilt_y[j]:+.5f} deg "
            f"(beam {beam_tilt_x_mrad[j]:+.4f} mrad)  {time.time() - t0:.0f} s  -> checkpoint",
            flush=True,
        )
    return path


# ============================================================================
# ANALYSIS: disk-pair asymmetry, CIF tilt vs beam tilt
# ============================================================================


def disk_curves(images):
    """Notebook metric: (b1 - b2)/(b1 + b2) * 100, box2 = exact mirror of box1."""
    from kikuchi_com import mirror_box

    H = images.shape[1]
    out = {}
    for order, width, ((y0, y1), (x0, x1)) in BOXES:
        Y0, Y1, _, _ = mirror_box((y0, y1, x0, x1), H)
        b1 = images[:, y0:y1, x0:x1].sum(axis=(1, 2))
        b2 = images[:, Y0:Y1, x0:x1].sum(axis=(1, 2))
        out[(order, width)] = (b1 - b2) / (b1 + b2) * 100
    return out


def analyse(beam_npz):
    import matplotlib.pyplot as plt

    plt.style.use("paper")

    cif = np.load(CIF_TILT_NPZ)
    beam = np.load(beam_npz)
    t = beam["tilt_y_values"]
    assert np.allclose(cif["tilt_y_values"], t), "tilt grids differ"
    assert cif["diffraction_images"].shape[1:] == beam["diffraction_images"].shape[1:], (
        "pattern shapes differ: the two runs used different extent/detector settings"
    )
    done = beam["done"]
    c_cif = disk_curves(cif["diffraction_images"].astype(float))
    c_beam = disk_curves(beam["diffraction_images"])

    i0 = int(np.argmin(abs(t)))
    print(f"\nbeam tilt: {done.sum()}/{len(t)} tilts done")
    print(f"{'box':12s} {'r(beam,cif)':>12s} {'r(beam,cif mirrored)':>21s} "
          f"{'rms diff %':>11s} {'cif @0 %':>9s} {'beam @0 %':>10s}")
    for key in c_cif:
        a, b = c_cif[key], c_beam[key]
        m = done & done[::-1]  # points where the mirrored comparison is also defined
        r = np.corrcoef(a[done], b[done])[0, 1] if done.sum() > 2 else np.nan
        r_m = np.corrcoef(a[::-1][m], b[m])[0, 1] if m.sum() > 2 else np.nan
        rms = np.sqrt(np.mean((a[done] - b[done]) ** 2))
        print(f"{'{' + key[0] + '} ' + key[1]:12s} {r:>12.3f} {r_m:>21.3f} {rms:>11.3f} "
              f"{a[i0]:>+9.3f} {b[i0] if done[i0] else np.nan:>+10.3f}")

    # raw convention: tilt as simulated, no polar flip (see simulated-polarisation notes)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2), sharex=True, sharey="row",
                             layout="constrained")
    for ax, (key, a) in zip(axes.flat, c_cif.items()):
        b = np.where(done, c_beam[key], np.nan)
        ax.axhline(0, color="black", lw=0.5, zorder=0)
        ax.axvline(0, color="black", lw=0.5, zorder=0)
        ax.plot(t, a, "-", lw=0.9, label="CIF tilt")
        ax.plot(t, b, "-o", lw=0.9, ms=1.8, markeredgewidth=0, label="Beam tilt")
        ax.set_title("{" + key[0] + "}  " + key[1].replace("x", r"$\times$"), fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"Tilt about $y$ (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Disk-pair asymmetry (%)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Boracite: CIF tilt vs beam tilt", fontsize=9)

    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"unflipped_boracite_disks_cif_vs_beam_tilt.{ext}"))
    print(f"saved {FIG_DIR}/unflipped_boracite_disks_cif_vs_beam_tilt.{{pdf,png}}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--analyse_only", action="store_true")
    args = p.parse_args()

    npz = os.path.join(args.out_dir, "results.npz") if args.analyse_only else run_sweep(args.out_dir)
    analyse(npz)
