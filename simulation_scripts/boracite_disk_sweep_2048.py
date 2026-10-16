#!/usr/bin/env python
"""Boracite Friedel-disk tilt sweep, rerun at a converged grid.

Same physics as boracite_disk_intensity_sim.py (7x7x40 along [110], -45 deg about z, 300 kV,
0.8 mrad, 1.0 A slices, sigma 0.1 A, 2x2 scan, 35 mrad detector, 81 tilts over +-0.25 deg about y),
except:
  - 2048 grid points instead of 1024 (1024 under-samples: {100} 1x1 at zero tilt -88.7 % at 1024,
    -71.9 % at 2048, -68.8 % at 4096);
  - 3 frozen-phonon configurations with the SAME seeds at every tilt (3 configs are within 1 point of
    10, at zero tilt and at -0.2 deg), so the curve carries no per-tilt phonon jitter.

Each tilt is saved to data/boracite_disks_2048_parts/ as it finishes; a restart skips finished tilts.
When all are done, data/boracite_disks_2048.npz is written with the same keys the disk notebooks read
(tilt_y_values, diffraction_images).

Usage (env sim-cbed-311, GPU), about 6 h:
    python boracite_disk_sweep_2048.py          # all 81 tilts
    python boracite_disk_sweep_2048.py 1        # first tilt only (smoke test)
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "data", "boracite_disks_2048_parts")
OUT = os.path.join(ROOT, "data", "boracite_disks_2048.npz")
TILTS = np.linspace(-0.25, 0.25, 81)
GPTS, N_PHONONS, SEED = 2048, 3, 0


def build():
    import abtem
    from ase.io import read

    atoms = read(os.path.join(ROOT, "cifs/cu2-ortho-sym.cif"))
    z, v = np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    ax = np.cross(v, z)
    atoms.rotate(np.degrees(np.arccos(v @ z)), ax / np.linalg.norm(ax), center="COP", rotate_cell=True)
    atoms = abtem.orthogonalize_cell(atoms) * (7, 7, 40)
    atoms.rotate(-45, "z", center="COP", rotate_cell=True)
    return atoms


def pattern(atoms, tilt_y):
    import abtem

    a = atoms.copy()
    if tilt_y != 0:
        a.rotate(tilt_y, "y", center="COP", rotate_cell=False)   # as in the production script
    fp = abtem.FrozenPhonons(a, sigmas=0.1, num_configs=N_PHONONS, seed=SEED)
    pot = abtem.Potential(fp, gpts=GPTS, slice_thickness=1.0, projection="finite")
    probe = abtem.Probe(energy=300e3, semiangle_cutoff=0.8, gpts=GPTS, sampling=pot.sampling, defocus=0)
    scan = abtem.GridScan(start=[0, 0], end=list(pot.extent), gpts=(2, 2))
    det = abtem.PixelatedDetector(max_angle=35, reciprocal_space=True)
    arr = np.asarray(probe.scan(scan=scan, potential=pot, detectors=det).compute(progress_bar=False).array)
    return arr.reshape(-1, *arr.shape[-2:]).mean(0).astype(np.float32)   # mean over phonons and scan


def main(n_run):
    import abtem

    abtem.config.set({"device": "gpu"})
    os.makedirs(PARTS, exist_ok=True)
    atoms = build()
    t0 = time.time()
    for i, t in enumerate(TILTS[:n_run]):
        part = os.path.join(PARTS, f"tilt_{i:03d}.npy")
        if os.path.exists(part):
            continue
        p = pattern(atoms, float(t))
        np.save(part + ".tmp.npy", p)
        os.replace(part + ".tmp.npy", part)
        print(f"[{i + 1}/{len(TILTS)}] tilt_y {t:+.5f} deg  {time.time() - t0:.0f} s  shape {p.shape}", flush=True)
    parts = [os.path.join(PARTS, f"tilt_{i:03d}.npy") for i in range(len(TILTS))]
    if all(os.path.exists(f) for f in parts):
        np.savez(OUT, tilt_y_values=TILTS, diffraction_images=np.stack([np.load(f) for f in parts]),
                 grid_points=GPTS, num_phonons=N_PHONONS, seed=SEED)
        print(f"saved {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else len(TILTS))
