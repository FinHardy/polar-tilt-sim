"""Corrected Kikuchi centre-of-mass asymmetry metric.

Three fixes relative to the version used in create_paper_figs_kikuchi.ipynb and
simulation_scripts/*_kikuchi_tilt_split_sim.py:

1. `mirror_box` — the mirrored box was one pixel off. For a half-open slice
   [lo, hi) the exact reflection about the centre N/2 (row r -> N - r) is
   [N + 1 - hi, N + 1 - lo), not [N - hi, N - lo).

2. `masked_com` — the metric is the centroid of the union of the two boxes,
   not the midpoint of the two boxes' individual centroids. The midpoint is
   blind to the intensity imbalance between the boxes, which is the signal.

3. `POLAR_AXIS` — abTEM returns the pattern indexed [kx, ky], so array axis 0
   is lab x. Which array axis carries the polarisation depends on the zone
   axis the simulation was run with; use `zone_axis_ratio` to check a dataset.

Centre is N/2 exactly (verified: direct beam and mirror symmetry both land on
index 2048.000 of a 4096 grid), NOT (N-1)/2.
"""

import numpy as np

N = 4096
CENTRE = N / 2.0
FULL_BOX = (800, 1400, 1988, 2108)  # (a0_min, a0_max, a1_min, a1_max)


# The simulation box of the boracite sims. The scripts rotate the 7x7 supercell by -45 deg
# about z and abTEM takes the bounding box: 7 * 17.1032 * sqrt(2). Commensurate with the
# rotated lattice (7 x 24.19 A), so it is a valid periodic box. Confirmed from the data:
# 606-px disk patterns at +-35 mrad, and the Bragg spacing (168 A measured).
BORACITE_EXTENT = 7 * 17.1032 * np.sqrt(2)     # 169.31 A


def angular_box(mpp0, mpp1, r_mrad, half_width_mrad=10.0, n=N):
    """Polar-strip box (a0_lo, a0_hi, a1_lo, a1_hi) on the low-index side, from angles.

    Same layout as FULL_BOX: rows n/2 - r_out .. n/2 - r_in (inclusive), columns
    n/2 - w .. n/2 + w (half-open), so boxes from different materials cover the same
    scattering angles even when their mrad-per-pixel differ.
    """
    r_in, r_out = round(r_mrad[0] / mpp0), round(r_mrad[1] / mpp0)
    w = round(half_width_mrad / mpp1)
    return (n // 2 - r_out, n // 2 - r_in + 1, n // 2 - w, n // 2 + w)


def third_box(box=FULL_BOX):
    """First third of a box along axis 0, as the published notebook defines it."""
    lo, hi, x0, x1 = box
    return (lo, lo + (hi - lo) // 3, x0, x1)


THIRD_BOX = third_box()


def mirror_box(box, n=N):
    """Exact reflection of a half-open box about the pattern centre n/2."""
    lo, hi, x0, x1 = box
    return (n + 1 - hi, n + 1 - lo, x0, x1)


def masked_com(pattern, box1, box2=None, axis=0):
    """Centroid of the union of the two boxes, along `axis`, relative to centre.

    Returns displacement in pixels. Positive = towards higher index.
    """
    if box2 is None:
        box2 = mirror_box(box1, pattern.shape[axis])
    p = pattern.T if axis == 1 else pattern
    tot = 0.0
    wsum = 0.0
    for lo, hi, x0, x1 in (box1, box2):
        rows = p[lo:hi, x0:x1].sum(axis=1)
        w = rows.sum()
        tot += (rows * np.arange(lo, hi)).sum()
        wsum += w
    return tot / wsum - pattern.shape[axis] / 2.0


# Disk lattice of the CBED sims: disks sit exactly on (H/2 + i*s, W/2 + j*s), with the direct
# beam at (H/2, W/2). s = nrep for PTO/BTO (30 px) and 28 px for boracite (4 x the 7 px
# fundamental of its 7 x 24.19 A box). The half-width gives the disk (direct-beam radius
# 4.5 px PTO/BTO, 6.5 px boracite) a ~2.5 px margin.
DISK_LATTICE = {"PTO": (30, 7), "BTO": (30, 7), "Boracite": (28, 9)}   # (spacing, half) px

# Disk row `order` (DISK_LATTICE steps from the direct beam, towards +P) is the 00l row, P || c
# in every cell: l = order for P4mm PTO/BTO (30 px = c*), l = 2*order for Pca2_1 boracite
# (28 px on the 169.3 A box = 2c*; 001 is forbidden by the 2_1 screw axis along c).
L_STEP = {"PTO": 1, "BTO": 1, "Boracite": 2}
ROW_NAME = {1: "1st row", 2: "2nd row", 3: "3rd row"}


def reflection(name, order, mirror=False):
    """Index of the central disk of row `order`, e.g. '002'; mirror=True gives its Friedel
    partner as mathtext/LaTeX, e.g. '$00\\bar{2}$'."""
    l = L_STEP[name] * int(order)
    return rf"$00\bar{{{l}}}$" if mirror else f"00{l}"


assert reflection("PTO", 3) == "003" and reflection("Boracite", "2") == "004"
assert reflection("BTO", 1, mirror=True) == r"$00\bar{1}$"


def disk_row_box(shape, spacing, half, order, n_disks):
    """Box1 ((y_lo, y_hi), (x_lo, x_hi)) around the `n_disks` central disks of the row
    `order` lattice steps above the direct beam (low row index). Half-open, +-half px about
    each disk centre, so mirror_box maps it exactly onto the same disks of row -order."""
    H, W = shape
    cy, k = H // 2 - order * spacing, (n_disks - 1) // 2 * spacing
    return ((cy - half, cy + half + 1), (W // 2 - k - half, W // 2 + k + half + 1))


def old_mirror_box(box, n=N):
    """The buggy mirror used in the published analysis, kept for comparison."""
    lo, hi, x0, x1 = box
    return (n - hi, n - lo, x0, x1)


def midpoint_com(pattern, box1, box2=None, axis=0):
    """The published metric: midpoint of the two boxes' *separate* centroids.

    Kept so the two defects can be separated. It is blind to the intensity
    imbalance between the boxes, and so discards the signal; use `masked_com`.
    """
    if box2 is None:
        box2 = mirror_box(box1, pattern.shape[axis])
    p = pattern.T if axis == 1 else pattern
    cs = []
    for lo, hi, x0, x1 in (box1, box2):
        rows = p[lo:hi, x0:x1].sum(axis=1)
        cs.append((rows * np.arange(lo, hi)).sum() / rows.sum())
    return sum(cs) / 2.0 - pattern.shape[axis] / 2.0


def stream_patterns(path, key="diffraction_images"):
    """Yield each 4096x4096 pattern from an NPZ without materialising the 10 GB
    array. np.load has no mmap for NPZ members, so read the member's raw stream.
    """
    import zipfile

    zf = zipfile.ZipFile(path)
    name = next(n for n in zf.namelist() if n.startswith(key))
    with zf.open(name) as f:
        version = np.lib.format.read_magic(f)
        # public readers; the private _read_array_header was dropped in numpy 2
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"unsupported .npy version {version} in {path}")
        shape, _, dtype = readers[version](f)
        nper = int(np.prod(shape[-2:]))
        for _ in range(int(np.prod(shape[:-2]))):
            buf = f.read(nper * dtype.itemsize)
            yield np.frombuffer(buf, dtype=dtype).reshape(shape[-2:]).astype(np.float64)


def zone_axis_ratio(pattern):
    """Ratio of the antialias-aperture radius on axis 0 to axis 1.

    abTEM's antialias aperture is a circle in reciprocal space, so in pixels it
    is an ellipse with semi-axis ratio = extent_x / extent_y. That identifies
    which lab axis the long (polar) cell axis was placed on:
      PTO  zone [100] -> 1.01444    PTO  zone [010] -> 0.98577
      BTO  zone [100] -> 1.00814    BTO  zone [010] -> 0.99192
      boracite        -> 1.00000 (square; use the static asymmetry instead)
    """
    r = []
    for axis in (0, 1):
        prof = pattern.sum(axis=1 - axis)
        c = len(prof) // 2
        thr = prof[c:c + 200].mean() * 1e-6
        hi = np.flatnonzero(prof[c:] < thr)[0]
        lo = np.flatnonzero(prof[:c][::-1] < thr)[0]
        r.append((hi + lo) / 2.0)
    return r[0] / r[1]


def _self_check():
    # mirror is exact: reflecting twice is the identity, and a box straddling
    # the centre maps to itself.
    assert mirror_box(mirror_box(FULL_BOX)) == FULL_BOX
    assert mirror_box((2000, 2096, 0, 10)) == (2001, 2097, 0, 10)
    assert mirror_box((2048, 2049, 0, 10)) == (2048, 2049, 0, 10)
    assert THIRD_BOX == (800, 1000, 1988, 2108)
    assert mirror_box(mirror_box(THIRD_BOX)) == THIRD_BOX

    # A pattern obeying p[N - r] == p[r] (abTEM's fftshifted convention: row 0
    # is the unpaired Nyquist row, row N/2 is the centre) must give zero.
    rng = np.random.default_rng(0)
    p = rng.random((N, N))
    p[N // 2 + 1:] = p[1:N // 2][::-1]
    b1 = (800, 1400, 1988, 2108)
    assert abs(masked_com(p, b1)) < 1e-9, masked_com(p, b1)
    assert abs(masked_com(p, THIRD_BOX)) < 1e-9

    assert abs(midpoint_com(p, b1)) < 1e-9, midpoint_com(p, b1)

    # the old mirror convention is off by ~0.5 px on the same symmetric pattern
    old = old_mirror_box(b1)
    assert old == (N - b1[1], N - b1[0], b1[2], b1[3])
    assert abs(masked_com(p, b1, old)) > 0.4
    assert abs(midpoint_com(p, b1, old)) > 0.4

    # masked_com tracks intensity imbalance, midpoint_com does not
    q = p.copy()
    q[b1[0]:b1[1], b1[2]:b1[3]] *= 1.10
    assert abs(midpoint_com(q, b1) - midpoint_com(p, b1)) < 1e-9
    assert abs(masked_com(q, b1)) > 40
    # angular_box reproduces FULL_BOX from its own angles (PTO calibration)
    lam = 0.0196870
    m0, m1 = lam / (30 * 3.9423) * 1e3, lam / (30 * 3.8862) * 1e3
    assert angular_box(m0, m1, (649 * m0, 1248 * m0), 60 * m1) == FULL_BOX
    # disk boxes: centred on the lattice, and the mirror lands on the opposite row's disks
    (y0, y1), (x0, x1) = disk_row_box((424, 418), 30, 7, 1, 3)
    assert ((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2) == (182, 209) and x1 - x0 == 2 * 30 + 15
    a, b, _, _ = mirror_box((y0, y1, x0, x1), 424)
    assert (a + b - 1) / 2 == 212 + 30
    print("kikuchi_com self-check OK")


if __name__ == "__main__":
    _self_check()
