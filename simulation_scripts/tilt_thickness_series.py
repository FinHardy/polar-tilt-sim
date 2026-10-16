#!/usr/bin/env python
"""PTO tilt x thickness series: does the Kikuchi vs Friedel-pair tilt tolerance depend on
thickness and convergence angle? (Shao et al. 2023, SI Fig. S6, PTO [100], 101 nm, 2 mrad:
Bragg pairs reverse at 0.04 deg, Kikuchi tolerates 0.23 deg; the paper finds the reverse
ordering at 60 nm, 0.8 mrad.)

One multislice run per (convergence angle, tilt, phonon configuration) records exit planes
every 10 nm to 100 nm, so each tilt sweep gives every thickness at once (see
thickness_series.py). Tilt is applied as in the production scripts: atoms rotated about y
(along the polar axis, lab x) about their centre, cell fixed.

Everything else as the production sims: 300 kV, 2x2 scan over the full extent, 4096^2 grid,
1.0 A slices, same thermal sigmas, full detector. Disk boxes are widened at 2 mrad so they
hold the whole disk (radius 12 px vs 4.8 px at 0.8 mrad; lattice spacing 30 px).

Output per (semiangle, tilt): OUT/<semiangle>mrad/tilt_<+0.000>/sum.npz (summed
scan-averaged patterns for all planes, configs done) and config_###.npz (metrics).
Restarts skip finished configurations.

Perpendicular tilt (--axis x): atoms rotated about x instead, i.e. tilt about the polar axis, so
the pattern moves perpendicular to P. Written to OUT + "_perp". The response is even in tilt by
symmetry, so PERP_TILTS samples one side plus two negative checks.

Usage (env: sim-cbed-311, GPU):
    python tilt_thickness_series.py time                      # one config, one tilt: timing
    python tilt_thickness_series.py run --semiangle 0.8 --phonons 10
    python tilt_thickness_series.py run --semiangle 2.0 --phonons 10
    python tilt_thickness_series.py run --semiangle 2.0 --phonons 5 --axis x
    python tilt_thickness_series.py analyse [--axis x]        # failure angle vs thickness
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from thickness_series import (MATERIALS, ROOT, build_atoms, exit_plane_indices,  # noqa: E402
                              save_atomic)
from kikuchi_com import DISK_LATTICE, FULL_BOX, disk_row_box, masked_com, mirror_box  # noqa: E402

OUT = os.path.join(ROOT, "data", "tilt_thickness_series")
NAME = "PTO"
CFG = dict(MATERIALS[NAME], reps=(30, 30, 259))       # 259 x 3.8862 A = 100.7 nm
N = 4096
TARGETS = np.arange(100.0, 1001.0, 100.0)             # A: 10, 20, ..., 100 nm
TILTS = np.round(np.arange(-0.30, 0.3001, 0.03), 3) + 0.0   # deg, 21 tilts (+0.0: no "-0")
PERP_TILTS = np.round(np.r_[-0.24, -0.12, np.arange(0.0, 0.3001, 0.03)], 3) + 0.0
AXIS = "y"                                            # rotation axis: y = along P, x = perpendicular
DISK_HALF = {0.8: DISK_LATTICE[NAME][1], 2.0: 14}     # px about each disk centre
DISKS = [("001 1x1", 1, 1), ("001 1x3", 1, 3), ("002 1x3", 2, 3), ("003 1x3", 3, 3)]


def simulate(atoms, semiangle, seed):
    """Scan-averaged pattern at each exit plane for one phonon configuration."""
    import abtem

    fp = abtem.FrozenPhonons(atoms, sigmas=CFG["sigma"], num_configs=1, seed=seed)
    pot = abtem.Potential(fp, gpts=N, slice_thickness=CFG["slice"], projection="finite")
    idx, depths, n_keep = exit_plane_indices(np.asarray(pot.slice_thickness), TARGETS)
    pot = abtem.Potential(fp, gpts=N, slice_thickness=CFG["slice"], projection="finite",
                          exit_planes=idx)
    probe = abtem.Probe(energy=300e3, semiangle_cutoff=semiangle, sampling=pot.sampling,
                        defocus=0)
    scan = abtem.GridScan(start=[0, 0], end=list(pot.extent), gpts=(2, 2))
    det = abtem.PixelatedDetector(max_angle="full", reciprocal_space=True)
    arr = np.asarray(probe.scan(scan=scan, potential=pot, detectors=det)
                     .compute(progress_bar=False).array)
    plane_ax = [i for i, s in enumerate(arr.shape[:-2]) if s == len(idx)][-1]
    other = tuple(i for i in range(arr.ndim - 2) if i != plane_ax)
    return arr.mean(axis=other)[:n_keep].astype(np.float32), depths


def metrics(p, semiangle):
    """Disk asymmetries (%), Kikuchi COM (A^-1), whole-pattern halves (%) for one pattern.
    Raw simulation frame: + = brighter / displaced towards higher row index (the side P
    points to in the PTO CIF, as thickness_series.signals with P_REVERSED False)."""
    n0, n1 = p.shape
    disks = []
    for _, order, n_disks in DISKS:
        (lo, hi), (x0, x1) = disk_row_box((n0, n1), DISK_LATTICE[NAME][0],
                                          DISK_HALF[semiangle], order, n_disks)
        a, b, _, _ = mirror_box((lo, hi, x0, x1), n0)
        top, bot = p[lo:hi, x0:x1].sum(), p[a:b, x0:x1].sum()
        disks.append((bot - top) / (bot + top) * 100)
    kik = masked_com(p, FULL_BOX, mirror_box(FULL_BOX, n0), 0) / CFG["extent0"]
    top, bot = p[: n0 // 2].sum(), p[n0 // 2 + 1:].sum()
    return np.array(disks), kik, (bot - top) / (bot + top) * 100


def run_tilt(base, semiangle, tilt, n_configs, seed0=2000):
    d = os.path.join(OUT, f"{semiangle:.1f}mrad", f"tilt_{tilt:+.3f}")
    os.makedirs(d, exist_ok=True)
    sum_path = os.path.join(d, "sum.npz")
    done, total = [], None
    if os.path.exists(sum_path):
        s = np.load(sum_path)
        done, total = list(s["configs"]), s["sum"]
    for i in (i for i in range(n_configs) if i not in done):
        t0 = time.time()
        atoms = base.copy()
        if tilt:
            atoms.rotate(tilt, AXIS, center="COP", rotate_cell=False)
        pats, depths = simulate(atoms, semiangle, seed0 + i)
        total = pats.astype(np.float64) if total is None else total + pats
        done.append(i)
        save_atomic(sum_path, sum=total.astype(np.float32), configs=np.array(done), depths=depths)
        m = [metrics(p, semiangle) for p in pats]
        save_atomic(os.path.join(d, f"config_{i:03d}.npz"), depths=depths, seed=seed0 + i,
                    disks=np.array([x[0] for x in m]), kikuchi=np.array([x[1] for x in m]),
                    halves=np.array([x[2] for x in m]))
        print(f"  {semiangle} mrad tilt {tilt:+.3f} config {i} {time.time() - t0:.0f} s  "
              f"Kikuchi at {depths[-1]/10:.0f} nm {m[-1][1]:+.4f}", flush=True)


def run(semiangle, n_configs, tilts=TILTS):
    import abtem

    abtem.config.set({"device": "gpu"})
    base = build_atoms(CFG, CFG["reps"])
    print(f"PTO {CFG['reps']}  cell {np.round(np.diag(base.cell.array), 1)} A  "
          f"{semiangle} mrad  {len(tilts)} tilts x {n_configs} configs", flush=True)
    for t in tilts:
        run_tilt(base, semiangle, float(t), n_configs)


def first_flip(t, y):
    """|tilt| nearest zero on each side where y leaves its zero-tilt sign (linear interp.)."""
    i0 = int(np.argmin(abs(t)))
    s0, out = np.sign(y[i0]), []
    for step in (-1, 1):
        i, hit = i0, np.nan
        while 0 <= i + step < len(t):
            j = i + step
            if np.sign(y[j]) != s0:
                hit = abs(t[i] + (t[j] - t[i]) * y[i] / (y[i] - y[j]))
                break
            i = j
        out.append(hit)
    return out          # [- side, + side]; nan = sign kept over the whole range


_t = np.array([-0.06, -0.03, 0.0, 0.03, 0.06])
assert np.allclose(first_flip(_t, np.array([1., 1., 1., -1., -1.])), [np.nan, 0.015], equal_nan=True)


def analyse(tilts=TILTS):
    labels = [d[0] for d in DISKS] + ["Kikuchi", "halves"]
    for sa in sorted(os.listdir(OUT)):
        rows = []
        for t in tilts:
            p = os.path.join(OUT, sa, f"tilt_{t:+.3f}")
            cfgs = sorted(f for f in os.listdir(p) if f.startswith("config_")) if os.path.isdir(p) else []
            if not cfgs:
                continue
            c = [np.load(os.path.join(p, f)) for f in cfgs]
            val = np.concatenate([np.mean([x["disks"] for x in c], 0),
                                  np.mean([x["kikuchi"] for x in c], 0)[:, None],
                                  np.mean([x["halves"] for x in c], 0)[:, None]], axis=1)
            rows.append((t, c[0]["depths"] / 10, val, len(c)))
        if len(rows) < 3:
            continue
        t = np.array([r[0] for r in rows])
        depths, v = rows[0][1], np.stack([r[2] for r in rows])     # (tilt, plane, metric)
        print(f"\n{sa}: {len(t)} tilts, configs per tilt {min(r[3] for r in rows)}-"
              f"{max(r[3] for r in rows)}.  Tolerance (deg) = min of first sign change each side")
        print("t(nm) " + "".join(f"{lab:>12s}" for lab in labels))
        for k, dep in enumerate(depths):
            tol = [np.nanmin(first_flip(t, v[:, k, j])) for j in range(v.shape[2])]
            print(f"{dep:5.0f} " + "".join(f"{x:>12.3f}" if np.isfinite(x) else f"{'>0.3':>12s}"
                                          for x in tol))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("time", "run", "analyse"))
    ap.add_argument("--semiangle", type=float, default=0.8, choices=(0.8, 2.0))
    ap.add_argument("--phonons", type=int, default=10)
    ap.add_argument("--axis", default="y", choices=("y", "x"))
    a = ap.parse_args()
    AXIS = a.axis
    tilts = TILTS
    if AXIS == "x":
        OUT, tilts = OUT + "_perp", PERP_TILTS
    if a.action == "time":
        OUT = os.path.join(OUT, "_timing")
        run(a.semiangle, 1, tilts=[0.0])
    elif a.action == "run":
        run(a.semiangle, a.phonons, tilts=tilts)
    else:
        analyse(tilts)
