#!/usr/bin/env python
"""Zero-tilt thickness series: Friedel disk-pair vs Kikuchi masked-COM signal.

PTO, BTO and boracite at zero tilt, sampled every 5 nm from 5 to 60 nm. One multislice
run per phonon configuration records exit planes at every thickness: multislice only
propagates forwards, so the wave at depth t inside the 60 nm crystal IS the exit wave
of a t-thick crystal. That gives all 12 thicknesses for the cost of one simulation.

Same physics as the production sims (simulation_scripts/*_kikuchi_tilt_split_sim.py):
300 kV, 0.8 mrad probe, 2x2 scan over the full extent, 4096^2 grid, full detector,
zone axis -> z, orthogonalize, boracite rotated -45 deg about z, the same thermal
sigmas. Slices are 1.0 A for every material (the PTO production runs used 1.5 A). (The production scripts also rebuild the potential to
"force a square extent"; in abTEM 1.0.8 that is a no-op -- the extent stays the
periodic cell -- so it is left out.)

Signals, both in "+P terms" with every material aligned to PTO's polarisation
(BTO and boracite are simulated with P reversed; see polarisation_direction.py):
    + = brighter on the side P points to.
In the paper figures P points to the bottom box (abTEM: +x = higher row index).
    disks    (I_P - I_-P) / (I_P + I_-P) x 100, per box from the disk notebook
    Kikuchi  masked COM along the polar axis (A^-1), boxes as in the Kikuchi figures

Checkpointing: after every phonon configuration the running sum of the scan-averaged
patterns (all thicknesses) and that configuration's metrics are written to
OUT/<material>/. A restart skips configurations already in the sum. Re-running with a
larger --phonons adds configurations to an existing run.

Usage (env: sim-cbed-311, GPU):
    python thickness_series.py run --material all --phonons 20
    python thickness_series.py run --material PTO --phonons 20     # one material
    python thickness_series.py plot                                 # figure only
    python thickness_series.py smoke                                # 1-minute test
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from kikuchi_com import (BORACITE_EXTENT, DISK_LATTICE, FULL_BOX, angular_box,  # noqa: E402
                         disk_row_box, masked_com, mirror_box)
from polarisation_direction import P_REVERSED  # noqa: E402

OUT = os.path.join(ROOT, "data", "thickness_series")
FIG = os.path.join(ROOT, "figure_notebooks", "figures", "thickness_series")
N = 4096
LAMBDA = 0.0196870                 # A, 300 kV
THICKNESSES = np.arange(50.0, 601.0, 50.0)   # A: 5, 10, ..., 60 nm

# Disk boxes: (label, (order, n_disks)), built by kikuchi_com.disk_row_box on the disk
# lattice -- the same boxes as the disk notebooks (the reciprocal pixel size 1/extent
# does not depend on the grid size, so the lattice is the same on the 4096 grid).
MATERIALS = {
    "PTO": dict(
        formula=r"PbTiO$_3$", cif="cifs/pbtio3_polarised_2.cif", zone=[1, 0, 0], zrot=0,
        reps=(30, 30, 155), slice=1.0, sigma={"Pb": 0.1068, "Ti": 0.0872, "O": 0.0503},
        extent0=30 * 3.9423, kikuchi_box=FULL_BOX, sim_thickness=154 * 3.8862,
        disks=[("001 1×3", (1, 3)),
               ("002 1×3", (2, 3)),
               ("003 1×3", (3, 3))]),
    "BTO": dict(
        formula=r"BaTiO$_3$", cif="cifs/batio3_polarised.cif", zone=[1, 0, 0], zrot=0,
        reps=(30, 30, 151), slice=1.0, sigma={"Ba": 0.0785, "Ti": 0.0703, "O": 0.0656},
        extent0=30 * 4.0230, kikuchi_box=FULL_BOX, sim_thickness=154 * 3.9905,
        disks=[("001 1×3", (1, 3)),
               ("002 1×3", (2, 3)),
               ("003 1×3", (3, 3))]),
    "Boracite": dict(
        formula=r"Cu$_3$B$_7$ClO$_{13}$", cif="cifs/cu2-ortho-sym.cif", zone=[1, 1, 0], zrot=-45,
        reps=(7, 7, 50), slice=1.0, sigma=0.1,
        extent0=BORACITE_EXTENT, sim_thickness=40 * 12.098,
        kikuchi_box=angular_box(LAMBDA / BORACITE_EXTENT * 1e3, LAMBDA / BORACITE_EXTENT * 1e3,
                                (40.0, 100.0)),
        disks=[("002 1×1", (1, 1)),
               ("002 1×3", (1, 3)),
               ("004 1×3", (2, 3))]),   # Pca2_1 indices: row n is 00(2n), see kikuchi_com.reflection
}


# ── structure and simulation ─────────────────────────────────────────────────
def build_atoms(cfg, reps):
    """The production scripts' orientation pipeline."""
    import abtem
    from ase.io import read

    atoms = read(os.path.join(ROOT, cfg["cif"]))
    z = np.array([0.0, 0.0, 1.0])
    v = np.array(cfg["zone"], float)
    v /= np.linalg.norm(v)
    axis = np.cross(v, z)
    if np.linalg.norm(axis) > 1e-6:
        atoms.rotate(np.degrees(np.arccos(v @ z)), axis / np.linalg.norm(axis),
                     center="COP", rotate_cell=True)
    atoms = abtem.orthogonalize_cell(atoms) * reps
    if cfg["zrot"]:
        atoms.rotate(cfg["zrot"], "z", center="COP", rotate_cell=True)
    return atoms


def exit_plane_indices(slice_thicknesses, targets):
    """Slice index after which each target depth is reached (nearest boundary).

    abTEM 1.0.8 indexes past the end of `exit_planes` unless the last entry is the
    final slice, so that slice is appended; `n_keep` planes are the targets.
    """
    depth = np.cumsum(slice_thicknesses)
    idx = [int(np.argmin(abs(depth - t))) for t in targets]
    assert len(set(idx)) == len(idx), "targets closer than one slice"
    n_keep = len(idx)
    if idx[-1] != len(depth) - 1:
        idx.append(len(depth) - 1)
    return tuple(idx), depth[idx[:n_keep]], n_keep


def simulate_config(atoms, cfg, seed, targets, gpts):
    """Scan-averaged pattern at each exit plane, one phonon configuration."""
    import abtem

    fp = abtem.FrozenPhonons(atoms, sigmas=cfg["sigma"], num_configs=1, seed=seed)
    pot = abtem.Potential(fp, gpts=gpts, slice_thickness=cfg["slice"], projection="finite")
    idx, depths, n_keep = exit_plane_indices(np.asarray(pot.slice_thickness), targets)
    pot = abtem.Potential(fp, gpts=gpts, slice_thickness=cfg["slice"], projection="finite",
                          exit_planes=idx)
    probe = abtem.Probe(energy=300e3, semiangle_cutoff=0.8, sampling=pot.sampling, defocus=0)
    scan = abtem.GridScan(start=[0, 0], end=list(pot.extent), gpts=(2, 2))
    det = abtem.PixelatedDetector(max_angle="full", reciprocal_space=True)
    arr = np.asarray(probe.scan(scan=scan, potential=pot, detectors=det)
                     .compute(progress_bar=False).array)
    # (..., planes, scan x, scan y, kx, ky): average everything but planes and detector
    plane_ax = [i for i, s in enumerate(arr.shape[:-2]) if s == len(idx)][-1]
    other = tuple(i for i in range(arr.ndim - 2) if i != plane_ax)
    return arr.mean(axis=other)[:n_keep].astype(np.float32), depths


# ── signals ──────────────────────────────────────────────────────────────────
def signals(p, cfg, name):
    """(disk values %, Kikuchi A^-1) for one pattern, in +P terms (aligned)."""
    n0, n1 = p.shape
    sign = -1.0 if P_REVERSED[name] else 1.0       # aligned at zero tilt: y(0) -> -y(0)
    disks = []
    for _, (order, n_disks) in cfg["disks"]:
        (lo, hi), (x0, x1) = disk_row_box((n0, n1), *DISK_LATTICE[name], order, n_disks)
        box = (lo, hi, x0, x1)
        a, b, _, _ = mirror_box(box, n0)
        top = p[box[0]:box[1], box[2]:box[3]].sum()       # low index = the -P side
        bot = p[a:b, box[2]:box[3]].sum()                 # high index = the +P side
        disks.append(sign * (bot - top) / (bot + top) * 100)
    kb = cfg["kikuchi_box"]
    kik = sign * masked_com(p, kb, mirror_box(kb, n0), 0) / cfg["extent0"]
    return np.array(disks), kik


# ── run with checkpoints ─────────────────────────────────────────────────────
def save_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def run(name, n_configs, gpts=N, reps=None, out=OUT, targets=THICKNESSES, seed0=1000):
    import abtem

    abtem.config.set({"device": "gpu"})
    cfg = MATERIALS[name]
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    sum_path = os.path.join(d, "sum.npz")
    done, total, depths = [], None, None
    if os.path.exists(sum_path):
        s = np.load(sum_path)
        done, total, depths = list(s["configs"]), s["sum"], s["depths"]
    todo = [i for i in range(n_configs) if i not in done]
    print(f"{name}: {len(done)} configurations done, {len(todo)} to run", flush=True)
    if not todo:
        return
    atoms = build_atoms(cfg, reps or cfg["reps"])
    with open(os.path.join(d, "parameters.txt"), "w") as f:   # rewritten on every (re)start
        f.write(f"material        : {name}\ncif             : {cfg['cif']}\n"
                f"zone axis       : {cfg['zone']}  z rotation {cfg['zrot']} deg\n"
                f"supercell       : {reps or cfg['reps']}  cell {np.round(np.diag(atoms.cell.array), 3).tolist()} A\n"
                f"grid points     : {gpts}\nslice thickness : {cfg['slice']} A\n"
                f"thermal sigma   : {cfg['sigma']}\nbeam            : 300 kV, 0.8 mrad, defocus 0\n"
                f"scan            : 2x2 over the full extent\ndetector        : full\n"
                f"exit planes at  : {np.round(np.asarray(targets) / 10, 2).tolist()} nm\n"
                f"phonon configs  : {n_configs} requested, seeds {seed0}+i\n"
                f"kikuchi box     : {tuple(int(v) for v in cfg['kikuchi_box'])}\n"
                f"disk boxes      : {cfg['disks']} (order, n_disks; kikuchi_com.disk_row_box)\n"
                f"abtem           : {abtem.__version__}\n")
    for i in todo:
        t0 = time.time()
        pats, depths = simulate_config(atoms, cfg, seed0 + i, targets, gpts)
        total = pats.astype(np.float64) if total is None else total + pats
        done.append(i)
        save_atomic(sum_path, sum=total.astype(np.float32), configs=np.array(done),
                    depths=depths)
        sig = [signals(p, cfg, name) for p in pats]
        save_atomic(os.path.join(d, f"config_{i:03d}.npz"), depths=depths, seed=seed0 + i,
                    disks=np.array([s[0] for s in sig]), kikuchi=np.array([s[1] for s in sig]))
        print(f"  {name} config {i} ({len(done)}/{n_configs}) {time.time() - t0:.0f} s   "
              f"Kikuchi at {depths[-1]/10:.0f} nm {sig[-1][1]:+.4f} A^-1", flush=True)
    np.savez_compressed(os.path.join(d, "mean_patterns.npz"),
                        patterns=(total / len(done)).astype(np.float32),
                        depths=depths, n_configs=len(done))
    print(f"{name}: saved mean_patterns.npz ({len(done)} configurations)", flush=True)


# ── figure ───────────────────────────────────────────────────────────────────
def collect(name, out=OUT):
    d = os.path.join(out, name)
    s = np.load(os.path.join(d, "sum.npz"))
    cfg = MATERIALS[name]
    mean = s["sum"] / len(s["configs"])
    sig = [signals(p, cfg, name) for p in mean]
    return dict(t=s["depths"] / 10, disks=np.array([x[0] for x in sig]),
                kik=np.array([x[1] for x in sig]), n=len(s["configs"]))


def plot(out=OUT, fig_path=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use("paper")
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_kik, c_disk = cycle[0], cycle[1:4]
    ls_disk = [(0, (4, 1.8)), (0, (1.5, 1.5)), (0, (6, 1.5, 1.5, 1.5))]
    names = [k for k in MATERIALS if os.path.exists(os.path.join(out, k, "sum.npz"))]
    if not names:
        sys.exit(f"no results in {out}")
    res = {k: collect(k, out) for k in names}
    d_lim = max(np.abs(r["disks"]).max() for r in res.values()) * 1.12
    k_lim = max(np.abs(r["kik"]).max() for r in res.values()) * 1.12

    mm = 1 / 25.4
    fig, axes = plt.subplots(1, len(names), figsize=(180 * mm, 62 * mm), layout="constrained",
                             squeeze=False)
    axes = axes[0]
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.04)
    for col, (name, ax) in enumerate(zip(names, axes)):
        r, cfg = res[name], MATERIALS[name]
        ax.axhline(0, color="black", lw=0.5, zorder=0)
        ax.axvline(cfg["sim_thickness"] / 10, color="#9a9a96", lw=0.7, ls=(0, (1, 1.4)), zorder=0)
        handles = []
        for j, (label, _) in enumerate(cfg["disks"]):
            h, = ax.plot(r["t"], r["disks"][:, j], color=c_disk[j], ls=ls_disk[j], lw=0.9,
                         marker="o", ms=1.8, mew=0, label=label)
            handles.append(h)
        ax.set_ylim(-d_lim, d_lim)
        ax.set_xlim(0, 62)
        ax.tick_params(axis="y", right=False, which="both")
        ax.set_xlabel("Thickness (nm)")
        if col == 0:
            ax.set_ylabel("Disk-pair asymmetry towards P (%)")
        else:
            ax.set_yticklabels([])

        ax2 = ax.twinx()
        h, = ax2.plot(r["t"], r["kik"], color=c_kik, lw=1.3, marker="o", ms=2.2, mew=0,
                      label="Kikuchi COM")
        handles.append(h)
        ax2.set_ylim(-k_lim, k_lim)
        ax2.tick_params(axis="y", left=False, which="both", colors=c_kik)
        if col == len(names) - 1:
            ax2.set_ylabel(r"Kikuchi COM towards P ($\mathrm{\AA}^{-1}$)", color=c_kik)
        else:
            ax2.set_yticklabels([])
        ax.legend(handles=handles, loc="lower left", fontsize=5.5, handlelength=2.0)
        ax.set_title(rf"{cfg['formula']}  ({r['n']} phonon configs)", pad=3, fontsize=8)
        ax2.text(0.03, 0.965, f"({'abc'[col]})", transform=ax2.transAxes, va="top", ha="left",
                 fontsize=8, fontweight="bold", zorder=10,
                 bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))

    fig.legend([Line2D([], [], color="#9a9a96", lw=0.7, ls=(0, (1, 1.4)))],
               ["thickness of the production simulations"], loc="outside lower center",
               fontsize=6.5)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path + ".pdf")
    fig.savefig(fig_path + ".png")
    print(f"saved {fig_path}.pdf / .png")
    for name, r in res.items():
        print(f"\n{name}  (+ = brighter towards P)\n  t(nm)  " + "  ".join(
            f"{lab:>10s}" for lab, _ in MATERIALS[name]["disks"]) + "   Kikuchi (A^-1)")
        for i, t in enumerate(r["t"]):
            print(f"  {t:5.1f}  " + "  ".join(f"{v:+10.1f}" for v in r["disks"][i])
                  + f"   {r['kik'][i]:+.4f}")


def smoke():
    """Tiny end-to-end run (small grid, thin crystal, 2 configs) into a scratch folder."""
    import tempfile

    out = tempfile.mkdtemp(prefix="thickness_smoke_")
    for name, reps in (("PTO", (30, 30, 8)), ("Boracite", (7, 7, 3))):
        run(name, 2, gpts=4096, reps=reps, out=out, targets=np.array([10.0, 20.0, 30.0]))
        run(name, 2, gpts=4096, reps=reps, out=out, targets=np.array([10.0, 20.0, 30.0]))  # resume
    plot(out=out, fig_path=os.path.join(out, "thickness_series"))
    print(f"smoke test OK -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("run", "plot", "smoke"))
    ap.add_argument("--material", default="all", choices=("all", *MATERIALS))
    ap.add_argument("--phonons", type=int, default=20)
    a = ap.parse_args()
    if a.action == "smoke":
        smoke()
    elif a.action == "plot":
        plot()
    else:
        for name in (MATERIALS if a.material == "all" else [a.material]):
            run(name, a.phonons)
        plot()
