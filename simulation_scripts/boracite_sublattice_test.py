#!/usr/bin/env python
"""Does boracite's Kikuchi COM follow the heavy Cu/Cl sublattice or the net polarisation?

Hybrid structures are built from the non-polar reference (the midpoint of the two P
domains, polarisation_direction.domain_disp) with only some sublattices given their polar
displacement:
    full   all displaced (the production structure)      P = -0.71 uC/cm2
    ref    none displaced (sanity: Kikuchi ~ 0)           P =  0
    CuCl   Cu and Cl only                                 P = +9.5   (opposite to full)
    BO     B and O only                                   P = -10.2  (same as full)
If the Kikuchi COM follows the Cu/Cl displacement, CuCl has the same sign as full and BO
the opposite sign. If it follows P, the reverse. (Point-charge P is quoted with Cl as
the displacement origin; full = CuCl + BO up to a rigid translation.)

Everything else is thickness_series.py: zero tilt, 4096 grid, 2x2 scan, full detector, and
exit planes at 20/30/40 nm in one 40 nm run. The same phonon seeds are used for every
variant, so the comparisons are paired. Signals are in the paper's aligned frame
(thickness_series.signals), where the production boracite Kikuchi COM is negative.

Usage (env: sim-cbed-311, GPU):
    python boracite_sublattice_test.py run --phonons 8
    python boracite_sublattice_test.py report
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thickness_series as ts  # noqa: E402
from polarisation_direction import Q, boracite_op, domain_disp  # noqa: E402

OUT = os.path.join(ts.ROOT, "data", "boracite_sublattice")
TARGETS = np.array([200.0, 300.0, 400.0])     # A
REPS = (7, 7, 34)                             # 34 x 12.1 A = 41 nm
VARIANTS = {"full": ("Cu", "Cl", "B", "O"), "ref": (), "CuCl": ("Cu", "Cl"), "BO": ("B", "O")}


def hybrid(keep):
    from ase.io import read

    atoms = read(os.path.join(ts.ROOT, ts.MATERIALS["Boracite"]["cif"]))
    u = domain_disp(atoms, boracite_op(atoms))
    sym = np.array(atoms.get_chemical_symbols())
    atoms.positions = atoms.positions - u + np.isin(sym, keep)[:, None] * u
    return atoms


def point_P(atoms):
    """Nominal P along the polar axis (lab x after orientation), uC/cm2, by domain pairing."""
    from polarisation_direction import lab_rotation
    q = np.array([Q[s] for s in atoms.get_chemical_symbols()])
    d = domain_disp(atoms, boracite_op(atoms))
    return (lab_rotation([1, 1, 0], -45) @ (q @ d))[0] / atoms.get_volume() * 1602.177


def build_for(keep):
    def build_atoms(cfg, reps):
        import abtem
        atoms = hybrid(keep)
        v = np.array(cfg["zone"], float) / np.linalg.norm(cfg["zone"])
        axis = np.cross(v, [0.0, 0.0, 1.0])
        atoms.rotate(np.degrees(np.arccos(v[2])), axis / np.linalg.norm(axis),
                     center="COP", rotate_cell=True)
        atoms = abtem.orthogonalize_cell(atoms) * reps
        atoms.rotate(cfg["zrot"], "z", center="COP", rotate_cell=True)
        return atoms
    return build_atoms


def check():
    full, ref = hybrid(VARIANTS["full"]), hybrid(())
    from ase.io import read
    orig = read(os.path.join(ts.ROOT, ts.MATERIALS["Boracite"]["cif"]))
    assert np.allclose(full.positions, orig.positions)
    assert abs(point_P(ref)) < 1e-6 and abs(point_P(full) + 0.71) < 0.01
    for k, keep in VARIANTS.items():
        print(f"  {k:5s} P = {point_P(hybrid(keep)):+7.2f} uC/cm2")


def report():
    rows = {}
    for k in VARIANTS:
        d = os.path.join(OUT, k, "Boracite")
        if not os.path.exists(os.path.join(d, "sum.npz")):
            continue
        r = ts.collect("Boracite", out=os.path.join(OUT, k))
        per = np.array([np.load(os.path.join(d, f)).get("kikuchi")
                        for f in sorted(os.listdir(d)) if f.startswith("config_")])
        rows[k] = (r, per)
    print("Kikuchi COM (A^-1, aligned frame; production boracite is negative)")
    print("  variant    P      " + "  ".join(f"{t:>12.0f} nm" for t in TARGETS / 10))
    for k, (r, _) in rows.items():
        print(f"  {k:5s} {point_P(hybrid(VARIANTS[k])):+7.2f}  " + "  ".join(
            f"{v:+.4f}±{s:.4f}" for v, s in zip(r["kik"], r["kik_sem"])))
    if "full" in rows:
        print("paired difference from full (same seeds), mean ± sem:")
        base = rows["full"][1]
        for k, (_, per) in rows.items():
            n = min(len(per), len(base))
            dlt = per[:n] - base[:n]
            print(f"  {k:5s} - full  " + "  ".join(
                f"{m:+.4f}±{s:.4f}" for m, s in zip(dlt.mean(0), dlt.std(0, ddof=1) / np.sqrt(n))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("check", "run", "report"))
    ap.add_argument("--phonons", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    check()
    if a.action == "run":
        for k in a.variants:
            print(f"=== {k}", flush=True)
            ts.build_atoms = build_for(VARIANTS[k])
            ts.run("Boracite", a.phonons, reps=REPS, out=os.path.join(OUT, k), targets=TARGETS)
        report()
    elif a.action == "report":
        report()
