import numpy as np
import abtem
from ase.io import read

CIF = {
    "PTO": "cifs/pbtio3_polarised_2.cif",
    "BTO": "cifs/batio3_polarised.cif",
    "boracite": "cifs/cu2-ortho-sym.cif",
}
BEAM_ENERGY = 300e3
SLICE = 1.5

# (label, structure, supercell_size, gpts)
CONFIGS = [
    ("PTO  disk",     "PTO",      (30, 30, 154), 4096),
    ("BTO  disk",     "BTO",      (30, 30, 154), 4096),
    ("bora disk",     "boracite", (7, 7, 40),    4096),
    ("PTO  kikuchi",  "PTO",      (30, 30, 154), 4096),
    ("BTO  kikuchi",  "BTO",      (30, 30, 154), 4096),
    ("bora kikuchi",  "boracite", (7, 7, 40),    4096),
]


def build_supercell(structure, supercell_size):
    ZONE_AXIS = [1, 1, 0] if structure == "boracite" else [1, 0, 0]
    atoms = read(CIF[structure])
    a = atoms.copy()
    if ZONE_AXIS != [0, 0, 1]:
        zv = np.array(ZONE_AXIS, float); zv /= np.linalg.norm(zv)
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(zv, z)
        if np.linalg.norm(axis) > 1e-6:
            axis /= np.linalg.norm(axis)
            ang = np.arccos(np.clip(np.dot(zv, z), -1, 1))
            a.rotate(np.degrees(ang), axis, center="COP", rotate_cell=True)
    a = abtem.orthogonalize_cell(a)
    sc = a * supercell_size
    if structure == "boracite":
        sc.rotate(-45, "z", center="COP", rotate_cell=True)
    return sc


def wavelength_A(eV):
    h, m, e, c = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0
    return h / np.sqrt(2 * m * e * eV) / np.sqrt(1 + e * eV / (2 * m * c**2)) * 1e10


lam = wavelength_A(BEAM_ENERGY)
print(f"lambda(300kV) = {lam:.6f} A\n")
print(f"{'config':14s} {'extent(A)':>18s} {'sampling(A)':>16s} {'gpts':>6s} "
      f"{'Nyquist(mrad)':>13s} {'2/3 cutoff(mrad)':>17s}")

for label, struct, scsize, gpts in CONFIGS:
    sc = build_supercell(struct, scsize)
    pot = abtem.Potential(sc, gpts=gpts, slice_thickness=SLICE, projection="finite")
    ex, ey = pot.extent[0], pot.extent[1]
    if abs(ex - ey) > 0.1:
        pot = abtem.Potential(sc, gpts=gpts, sampling=pot.sampling[0],
                              slice_thickness=SLICE, projection="finite")
    sx, sy = pot.sampling
    # max SAMPLED (Nyquist) angle per axis, take min (limiting)
    nyq_x = lam / (2 * sx) * 1e3
    nyq_y = lam / (2 * sy) * 1e3
    nyq = min(nyq_x, nyq_y)
    cutoff = (2 / 3) * nyq
    print(f"{label:14s} {ex:8.2f},{ey:8.2f} {sx:7.4f},{sy:7.4f} {gpts:6d} "
          f"{nyq:13.1f} {cutoff:17.1f}")
