# Kikuchi asymmetry analysis — audit findings

Audit date: 2026-09-18. Derived entirely from the saved `.npz` simulation outputs, not from
the code's own reported numbers. Three defects were reported by a colleague; all three are
real, though one is real for a different reason than stated and one affects only PbTiO₃.

This document is written for whoever picks up the cluster side. It states what is wrong, how
each claim was verified (so you can re-verify independently), what to change in the
simulation scripts, and what to run. Every number quoted here was measured, not estimated.

---

## 0. Conventions you must know before reading anything else

These are not documented anywhere in the existing code and getting them wrong is the root of
two of the three defects.

**abTEM returns diffraction patterns indexed `[kx, ky]`.** Array axis 0 is lab *x*. The
analysis code calls axis 0 "y" throughout (`ny, nx = pattern.shape`, `com_y_displacement`,
`apply_y_shift`). That naming is wrong but *internally consistent*, so it does not by itself
cause an error — it caused an error only where someone reasoned about the displayed image
("up-down") instead of the array.

**The pattern centre is `N/2` exactly**, i.e. index 2048.0 of a 4096 grid — not `(N-1)/2`.
abTEM returns an fftshifted grid, so for even `N` the zero-frequency pixel sits at index
`N/2`. Verified two ways on the PTO zero-tilt pattern:

- direct-beam centroid: `(2048.004, 2048.640)`
- mirror-symmetry centre of each marginal profile (the `S` maximising `Σ f(i)·f(S-i)`,
  centre `= S/2`): `2048.0` on both axes

`center_y = ny / 2` in the existing code is therefore **correct**. Do not "fix" it.

**Reflection about that centre maps row `r → N - r`.** For a half-open slice `[lo, hi)`
covering rows `lo … hi-1`, the reflected rows are `N-hi+1 … N-lo`, i.e. the half-open slice
`[N+1-hi, N+1-lo)`. See §2.

**Which array axis carries the polarisation depends on the zone axis.** For PbTiO₃ and
BaTiO₃ (both P4mm, polar along **c** = [001]) simulated down zone axis [100], the rotation +
`abtem.orthogonalize_cell` puts **c on lab x = array axis 0**. Confirmed identical in abTEM
1.0.5 and 1.0.8:

```
ZONE_AXIS [1,0,0] -> ortho cell lengths [3.9423 3.8862 3.8862]  c on lab axis 0
ZONE_AXIS [0,1,0] -> ortho cell lengths [3.8862 3.9423 3.8862]  c on lab axis 1
```

For boracite (Pca2₁, polar along **c**), zone axis [1,1,0] followed by the `-45°` z-rotation
sends **c → lab (−1, 0, 0) = array axis 0** as well. So all materials are *supposed* to have
the polar axis on array axis 0.

**`rotate(tilt_y, "y")` moves the pattern along array axis 0.** Verified on
`grids/kikuchi/PTO_kikuchi_grid.npz`, a 9×9 `tilt_x` × `tilt_y` sweep. Computing the corrected
metric (§1) over the grid:

- along **axis 0**: varies monotonically down index `i` (= `tilt_y`), symmetric across `j`
- along **axis 1**: varies across index `j` (= `tilt_x`), antisymmetric in `j`, and **exactly
  zero at `j = 4` (`tilt_x = 0`)** for every `i` — values −0.24, −0.55, −0.07, 0.28, 0.27,
  0.67, −0.22, 0.28, 0.18 px against a ±150 px range

Grid index `i` was confirmed to be `tilt_y` by matching grid frame `(8,4)` against the 1D
sweep's `+0.625°` frame (residual 0.457 vs 0.86–0.93 for the alternatives).

So the boxes (on axis 0) and the tilt (which moves axis 0) agree with each other. Whether
they agree with the *crystal* depends on the zone axis actually used — see §3.

---

## 1. Defect A — the COM metric discards the signal

**Status: confirmed. Affects every Kikuchi result.**

`calculate_box_asymmetry_score` computes each box's centroid separately and takes their
midpoint:

```python
midpoint_y = (mask_1_com[0] + mask_2_com[0]) / 2
```

The intended metric is the centroid of the *union* of the two boxes — mask everything except
the two boxes, take the centroid of what is left:

```python
combined = (I1*c1 + I2*c2) / (I1 + I2)
```

These differ by exactly

```
combined - midpoint = (c1 - c2)/2 · (I1 - I2)/(I1 + I2)
```

With the published boxes, `(c1 - c2)/2 ≈ -948 px`. The midpoint cancels the
`(I1-I2)/(I1+I2)` term entirely, and that term *is* the Kikuchi band asymmetry. The published
metric is blind to the intensity imbalance it was meant to measure.

Scale of the difference: published curves span ±3 px; the corrected metric spans ±200 px on
the same data.

Note the corrected metric is essentially the normalised intensity asymmetry expressed in
pixels, not a centroid shift in any physical sense. That is worth stating plainly in the
paper rather than calling it a "centre-of-mass displacement".

---

## 2. Defect B — one-pixel error in the box mirror

**Status: confirmed, but NOT where it was reported.** The reported claim was "the centre
point is off by a pixel". `center_y = ny/2` is correct (§0). The off-by-one is in
`infer_box2_from_box1`:

```python
box2_y_min = pattern_height - y_max      # WRONG
box2_y_max = pattern_height - y_min
```

Correct:

```python
box2_y_min = pattern_height + 1 - y_max
box2_y_max = pattern_height + 1 - y_min
```

Concretely, for `N = 4096`:

| box | box1 | published box2 | exact mirror |
|---|---|---|---|
| full | `(800, 1400)` | `(2696, 3296)` | `(2697, 3297)` |
| top-third | `(800, 1000)` | `(3096, 3296)` | `(3097, 3297)` |

Both box sizes carry it. Effect: a **constant −0.511 px bias** in every pattern, every
material, every tilt.

### Independent proof

Array axis 1 is non-polar in the correctly-oriented datasets, so the true answer there is
zero. Measuring the midpoint metric on axis 1 at zero tilt:

| material | published mirror | exact mirror |
|---|---|---|
| PTO | −0.5789 | −0.0700 |
| BTO | −0.4618 | +0.0444 |
| Boracite | −0.6271 | −0.1130 |

The exact mirror recovers zero; the published one does not.

### This bias *was* the published result

Taking the even part of the tilt curve, `e(t) = ½[y(t) + y(−t)]` — a polar offset is even in
tilt, a geometric response is odd and averages out:

| material | box | published | mirror fixed | fully fixed, axis 0 | fully fixed, axis 1 |
|---|---|---|---|---|---|
| PTO | full | −0.513 (78.6σ) | −0.007 (1.1σ) | −0.183 (2.8σ) | **−26.479 (6.6σ)** |
| PTO | third | −0.490 (101.3σ) | +0.001 (0.3σ) | −0.276 (2.6σ) | **−38.150 (7.7σ)** |
| BTO | full | −0.612 (10.7σ) | −0.103 (1.8σ) | +2.265 (1.6σ) | −0.018 (0.3σ) |
| BTO | third | −0.472 (19.8σ) | +0.024 (1.2σ) | +1.729 (0.9σ) | +0.019 (0.2σ) |
| Boracite | full | −0.477 (12.3σ) | +0.065 (1.7σ) | −3.634 (3.4σ) | +1.178 (12.6σ) |
| Boracite | third | −0.385 (12.3σ) | +0.114 (3.7σ) | −2.644 (1.7σ) | −0.397 (0.9σ) |

All units px. The published offset is the same −0.4 to −0.6 px in all three materials — that
is the bug, not physics. Fixing the mirror alone drops every material below 2σ.

> **The σ values in this table are wrong and understate the evidence. Superseded — see §10.**
> They were computed treating the scatter of the even part across tilt points as random noise.
> A repeat run (2026-09-21) proves that scatter is *deterministic structure*, reproducible
> between independent runs at r = 1.000. Use the even-part RMS diagnostic in §10 instead. The
> `<even>` means themselves are unaffected; only the error bars were mis-specified.

**The top-third box made the artefact look better, not worse.** PTO's published significance
goes 78.6σ → 101.3σ, BTO 10.7σ → 19.8σ. The bias is unchanged but the scatter is smaller in
the narrower box, so the spurious offset appears *more* significant. If the top-third box was
selected partly because it gave a cleaner result, that selection was driven by the defect.

The top-third box does genuinely help the real signal, though: on PTO's actual polar axis it
gives −38.2 px at 7.7σ versus −26.5 px at 6.6σ for the full box.

---

## 3. Defect C — wrong zone axis, PbTiO₃ only

**Status: confirmed for PTO. BTO and boracite are correctly oriented. All disk datasets are
correctly oriented.**

The reported claim was "the wrong polarisation direction was simulated". More precisely: two
PTO Kikuchi runs used `ZONE_AXIS = [0,1,0]` instead of `[1,0,0]`. That places the polar
**c**-axis on array axis 1, while the analysis boxes and the `rotate(tilt_y, "y")` tilt are
both on array axis 0. So for PTO the sweep measured the non-polar direction *and* tilted
perpendicular to **P**.

This is not a crystallographic error — in P4mm, *a* and *b* are equivalent, so [010] is a
valid zone axis and the physics simulated is sound. It is a convention mismatch with an
analysis that hard-codes axis 0. That is exactly why it produced plausible-looking curves
instead of obvious garbage.

### How to detect it on any dataset (no intensity assumptions)

abTEM's antialias aperture is a **circle in reciprocal space**. If `extent_x ≠ extent_y` the
reciprocal pixel size differs per axis, so in pixels the aperture is an **ellipse** whose
semi-axis ratio equals `extent_x / extent_y = c/a`. Measure it by finding where the summed
profile along each axis collapses:

```python
def zone_axis_ratio(pattern):
    r = []
    for axis in (0, 1):
        prof = pattern.sum(axis=1 - axis)
        c = len(prof) // 2
        thr = prof[c:c + 200].mean() * 1e-6
        hi = np.flatnonzero(prof[c:] < thr)[0]
        lo = np.flatnonzero(prof[:c][::-1] < thr)[0]
        r.append((hi + lo) / 2.0)
    return r[0] / r[1]
```

Expected values:

| | ratio |
|---|---|
| PTO zone [100] (correct) | 1.01444 |
| PTO zone [010] (wrong) | 0.98577 |
| BTO zone [100] (correct) | 1.00814 |
| BTO zone [010] (wrong) | 0.99192 |
| boracite | 1.00000 (square — this test carries no information; use §0's analytic result) |

**Do not** try to identify the orientation from Bragg spot spacing. The supercell is 30×30
unit cells and the grid is commensurate with it, so the spot period is exactly 30 px on both
axes by construction, regardless of orientation. That test is void.

### Dataset audit

| dataset | ratio | polar axis | verdict |
|---|---|---|---|
| `long_range/PTO_long_range_kikuchi_asymmetry_data.npz` | 0.98577 | axis 1 | **WRONG — zone [010]** |
| `short_range/1_slice_thickness/PTO_kikuchi_1_clice_final_asymmetry_data.npz` | 0.98613 | axis 1 | **WRONG — zone [010]** |
| `short_range/PTO_kikuchi_15_slice_final_asymmetry_data.npz` | 1.00831 | axis 0 | **NOT PTO — see below** |
| `INCOMPLETE_PTO_kikuchi_final_asymmetry_data.npz` | 1.01521 | axis 0 | correct |
| `grids/kikuchi/PTO_kikuchi_grid.npz` | 1.01521 | axis 0 | correct |
| `long_range/BTO_long_range_kikuchi_asymmetry_data.npz` | 1.00831 | axis 0 | correct |
| `short_range/BTO_kikuchi_final_15_slice_asymmetry_data.npz` | 1.00831 | axis 0 | correct |
| `long_range/Boracite_long_range_kikuchi_asymmetry_data.npz` | 1.00000 | axis 0 | correct |

### Corroborating evidence from the data itself

`long_range/PTO_long_range...` at zero tilt: corrected metric reads **+0.03 px on axis 0** and
**−24.15 px on axis 1**. `short_range/BTO...`: **−8.17 px on axis 0**, **−0.25 px on axis 1**.
The two are swapped, exactly as the ratio test predicts. Across the full sweep, PTO's axis-1
signal is even in `tilt_y` to better than 1 px (−43.8/−44.2, −35.2/−36.1, +24.6/+25.3 …),
confirming it is a real static asymmetry rather than noise.

### Consequence

The existing PTO long-range sweep **cannot be rescued in analysis**. The tilt is baked into
the simulation and it sweeps the non-polar axis. The only thing recoverable from it is the
zero-tilt static asymmetry on axis 1 (−26.5 px, 6.6σ), which is a valid measurement.

---

## 4. Other dataset problems found

**`BTO_kikuchi_20_phonons_asymmetry_data.npz` and
`long_range/BTO_long_range_kikuchi_asymmetry_data.npz` are byte-identical.** Same size
(10240 MB), same md5 over the first 200 MB (`6256f7b2096a0e6b43fabe1f2cc4f5df`), computed
metrics correlate at exactly 1.000 with difference sd 0.0. They are one run stored twice.

Consequences: the BTO long-range panel is a **20-phonon** run, and **no repeat run exists for
any material**, so there is no convergence information anywhere. The corrected metric has
±62 px point-to-point scatter while the BTO polar signal is +2.3 px — without a repeat, that
error bar is an assumption, not a measurement.

**`short_range/PTO_kikuchi_15_slice_final_asymmetry_data.npz` is not PbTiO₃.** Its aperture
ratio is 1.00831, which is BaTiO₃'s `c/a` (1.00814), not PbTiO₃'s in either orientation
(1.01444 / 0.98577). Either it is mislabelled BTO or it was run from a different CIF with
`c/a = 1.0083`. Check the originating run directory's `parameters.txt` and CIF path.

**`short_range/boracite_kikuchi_15_slice_final_asymmetry_data.npz` is incomplete.** It holds
only 20 of 41 tilts, all negative (−0.6 to −0.03). It cannot be symmetrised at all.

**`INCOMPLETE_PTO_kikuchi_final_asymmetry_data.npz` is misnamed — it is complete.** 41 tilts
spanning −0.6 to +0.6, no blank frames, correctly oriented. This is probably already the PTO
short-range dataset. Verify its slice thickness matches the BTO/boracite short-range runs
before spending cluster time reproducing it.

**`parameters.txt` provenance is unreliable.** The PTO one for run 39539133 records
`Num phonons : 40` while the script sets `num_phonons = 20` for PTO. Trust neither without
checking the run directory.

---

## 5. Defects in `simulation_scripts/*_kikuchi_tilt_split_sim.py`

Line numbers refer to `PTO_kikuchi_tilt_split_sim.py`; the boracite variant is structurally
identical.

### Blockers — the script as committed will not perform a sweep

**`TILT_Y_NPTS = 1` (line 72).** `np.linspace(-0.6, 0.6, 1)` returns `[-0.6]`, a single point.
Submitting an 81-task array gives:

```
--task_id 0 : runs tilts [-0.6]
--task_id 1+: start_idx >= TILT_Y_NPTS -> prints "Nothing to do", returns, exits 0
```

One tilt computed, 80 tasks silently no-op. They exit successfully so nothing alerts you.

**Tilt range is ±0.6 (lines 70–71), i.e. the short-range configuration.** To match the
existing BTO/boracite long-range files, set `TILT_Y_MIN/MAX = ∓2.5` and `TILT_Y_NPTS = 81`,
which reproduces their 0.0625° step.

### Silent-corruption risks

**`--tilt_y` without `--task_id` makes all tasks collide.** Line 748 defaults `task_id` to 0
when `args.task_id is None`, and the output filename is `tilt_task_{task_id:03d}.npz`
(line 375). Every task writes `tilt_task_000.npz` concurrently and records
`global_tilt_indices=[0]`. If you use the override path, pass **both** flags.

**Failed tilts are swallowed and written as data.** Line 570 catches every exception, prints,
and continues; `_save_partial_npz` then stores `img_array[j] = np.nan`. The task exits 0, so
SLURM reports success. This has already happened — the analysis notebook carries
`corrupted_indices = {75}` for BTO, which is precisely this failure mode surviving into the
dataset. Re-raise, or at minimum `sys.exit(1)`, so a failed tilt fails its task.

**The "Ensure square extent" block is a no-op (lines 402–410).** It rebuilds the Potential
passing *both* `gpts` and `sampling`; `gpts` wins and the extent stays `30c × 30a`. Proof:
the saved patterns are 4096×4096 *and* their antialias aperture is elliptical with ratio
exactly `c/a`, which is only possible if the extents remained unequal. Consequence:
**mrad/px differs between the two axes by ~1.4%**. Harmless for symmetry work; wrong if
anyone converts COM pixels to mrad.

**Box tilt-tracking is inert, and inverted if revived (line 492).**
`apply_y_shift=False, apply_x_shift=True` — with `TILT_X_FIXED = 0` the x-shift is 0 and the
y-shift is discarded, so the boxes never move. This matches the saved data (box coordinates
are identical at every tilt) and matches what the corrected analysis assumed, so it has not
corrupted anything. But the pattern moves along array axis 0 under `tilt_y` — the axis the
code calls "y" and has disabled. Decide deliberately: keep boxes fixed (current behaviour,
recommended, and what all existing analysis assumes) or track, in which case it is
`apply_y_shift=True`.

### Not a defect

**`ZONE_AXIS` is correct in the committed script** (line 305: `[1,0,0]` for non-boracite).
The bad PTO runs came from an edited copy, which is why the defect is invisible in version
control. Nothing guards against a recurrence — see §6.

**CIF paths are relative** (`../cifs/...`, line 49), so jobs must launch from a subdirectory
of the project. This one fails loudly, since `read()` sits outside the try block.

**The recorded `com_y_displacement` column carries Defects A and B**, but the raw patterns are
saved, so all of it can be recomputed. Ignore that column.

---

## 6. Recommended guard

Add this immediately after the supercell is built, before any compute. It costs nothing and
would have caught the zone-axis defect before 24 h of cluster time:

```python
# The analysis boxes and the tilt both act on array axis 0, so the polar axis
# must land on lab x. Fail fast rather than after a day of compute.
_L = base_rotated_atoms.get_cell().lengths()
if structure_name in ("PTO", "BTO"):
    assert _L[0] > _L[1], (
        f"{structure_name}: polar c is not on lab x (ortho cell {_L}). "
        f"Check ZONE_AXIS — expected [1,0,0], got {ZONE_AXIS}.")
```

For PTO the correct cell is `[3.9423, 3.8862, 3.8862]`; the `[010]` mistake gives
`[3.8862, 3.9423, 3.8862]`. For BTO, `[4.023, 3.9905, 3.9905]`. Boracite is square after the
−45° rotation, so the assertion does not apply; its orientation was verified analytically
(§0).

---

## 7. What to run

Ranked. Roughly 110–130 tilt points total, comparable to one existing long-range array job.

**1. PTO long-range re-run with `ZONE_AXIS = [1,0,0]` — 81 tilt points.** The blocker for the
paper. `tilt_y` −2.5…2.5, `TILT_Y_NPTS = 81`, supercell `(30,30,154)`, matching BTO and
boracite so the three-panel figure is consistent. Nothing else fixes the PTO panel.

**2. A convergence pair for BTO — DOWNGRADED, see §10.** A PTO repeat now exists and shows
frozen-phonon noise is only ~0.3 px per point, with the tilt structure reproducible at
r = 1.000. If BTO behaves the same — and there is no reason to expect otherwise at the same
settings — its signal is already far outside the noise and no repeat is needed. A ~9-point
BTO repeat is still worth doing as a cheap confirmation, but it is no longer a blocker.

Do **not** spend cluster time raising `num_phonons` to reduce the ±40–60 px tilt-to-tilt
scatter. That scatter is physics, not sampling error, and more configurations will not
reduce it.

**3. Finish boracite short-range — 21 tilt points.** Tilts 0 to +0.6 are missing (§4).

**4. Probably nothing — check first.** There is no PTO short-range 15-slice file (the one so
named is BTO), but `INCOMPLETE_PTO_kikuchi_final_asymmetry_data.npz` may already serve. Check
its slice thickness against the BTO/boracite short-range runs before queuing anything.

### Do not re-run

BTO and boracite long-range (orientation correct), and all three disk datasets. The disk
patterns were checked via their detector-crop aspect ratio, which is itself the `c/a`
signature: PTO 424×418 = 1.01435 against `c/a` = 1.01444; BTO 432×428; boracite 606×606
square. Polar axis on array axis 0 in all of them.

### Optional follow-up

There is a crude three-point thickness series already (1-slice, 15-slice, full), and the
corrected metric shows the polar offset absent in the thin runs and present in the thick ones.
A proper thickness series would turn that into a result. New science rather than paper repair,
so it is last.

---

## 8. Corrected analysis code

Self-contained; matches `kikuchi_com.py` in the analysis repo.

```python
import numpy as np

N = 4096
FULL_BOX  = (800, 1400, 1988, 2108)   # (axis0_min, axis0_max, axis1_min, axis1_max)
THIRD_BOX = (800, 1000, 1988, 2108)   # first third along axis 0


def mirror_box(box, n=N):
    """Exact reflection of a half-open box about the pattern centre n/2."""
    lo, hi, x0, x1 = box
    return (n + 1 - hi, n + 1 - lo, x0, x1)


def masked_com(pattern, box1, box2=None, axis=0):
    """Centroid of the UNION of the two boxes along `axis`, relative to the centre.

    This is the correct metric. Returns pixels; positive = towards higher index.
    """
    if box2 is None:
        box2 = mirror_box(box1, pattern.shape[axis])
    p = pattern.T if axis == 1 else pattern
    tot = wsum = 0.0
    for lo, hi, x0, x1 in (box1, box2):
        rows = p[lo:hi, x0:x1].sum(axis=1)
        tot  += (rows * np.arange(lo, hi)).sum()
        wsum += rows.sum()
    return tot / wsum - pattern.shape[axis] / 2.0


def stream_patterns(path, key="diffraction_images"):
    """Yield each 4096x4096 pattern without materialising the 10 GB array.

    np.load has no mmap for NPZ members, so read the member's raw stream.
    Uses the public header readers: numpy 2 removed _read_array_header.
    """
    import zipfile
    zf = zipfile.ZipFile(path)
    name = next(n for n in zf.namelist() if n.startswith(key))
    with zf.open(name) as f:
        version = np.lib.format.read_magic(f)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        shape, _, dtype = readers[version](f)
        nper = int(np.prod(shape[-2:]))
        for _ in range(int(np.prod(shape[:-2]))):
            buf = f.read(nper * dtype.itemsize)
            yield np.frombuffer(buf, dtype=dtype).reshape(shape[-2:]).astype(np.float64)
```

The polar signature is the **even part** of the tilt curve:

```python
def even_part(t, y):
    i0 = int(np.argmin(abs(t)))
    n  = min(i0, len(t) - 1 - i0)
    return (y[i0-n:i0] + y[i0+1:i0+1+n][::-1]) / 2
```

Report `even_part(...).mean()` with SEM `std(ddof=1)/sqrt(n)`. A purely geometric tilt
response is odd and averages to zero; only a polar offset survives.

---

## 9. Checklist before launching anything

- [ ] `TILT_Y_NPTS` set to the real number of points, not 1
- [ ] `TILT_Y_MIN/MAX` match the sweep you intend (±2.5 long-range, ±0.6 short-range)
- [ ] `ZONE_AXIS` guard from §6 added and passing
- [ ] exception handler at line 570 re-raises or exits non-zero
- [ ] if using `--tilt_y`, also pass `--task_id` so output filenames are unique
- [ ] `num_phonons` chosen deliberately, and recorded somewhere that is actually true
- [ ] `structure_name` block uncommented is the one you meant
- [ ] launched from a directory where `../cifs/` resolves


---

## 10. Correction — the error model (added 2026-09-21)

A second correctly-oriented PTO short-range run
(`short_range/PTO_short_range_21_sept_asymmetry_data.npz`, 41 tilts −0.6…0.6, zone [100]
confirmed by aperture ratio 1.01481) turned out to be an **independent repeat** of
`INCOMPLETE_PTO_kikuchi_final_asymmetry_data.npz` — same size and geometry, different md5,
2.2 % L1 difference between the zero-tilt patterns. That is the convergence pair §7 asked for.

### Frozen-phonon noise is tiny; the tilt structure is real

| quantity | full box | third box |
|---|---|---|
| run-to-run σ per point (polar axis) | **0.30 px** | 0.57 px |
| scatter of the curve itself | 41 px | 48 px |
| correlation between runs | **r = 1.000** | r = 0.996 |

The ±41 px tilt-to-tilt swing is **deterministic dynamical structure**, not sampling noise.
More frozen-phonon configurations will not reduce it.

### Consequence: the earlier significance numbers were mis-specified

Every σ in §2 divided the across-tilt scatter of the even part by √n, treating deterministic
structure as random error. That inflates the error bar by roughly two orders of magnitude and
made real signals look insignificant (PTO 0.7σ, BTO 1.6σ, boracite 3.4σ). Disregard them.

The mean of the even part is also a poor summary statistic: e(t) oscillates over ±40 px and
crosses zero repeatedly (10 of 20 points negative), so its mean is window-dependent — it is
reproducible to 0.013 px between runs but it is not a physical constant.

### Use this diagnostic instead

**Does the curve have a deterministic even component at all?** A polar offset is even in tilt;
a geometric response is odd. The non-polar array axis is a built-in control within the same
dataset — it must show none. Compare the RMS of the even part on the two axes:

| dataset | box | e_rms axis 0 | e_rms axis 1 | ratio | polar axis |
|---|---|---|---|---|---|
| PTO 21-Sept short | full | 22.21 | 0.21 | 108 | axis 0 |
| PTO 21-Sept short | third | 29.82 | 0.40 | 75 | axis 0 |
| PTO earlier short | full | 22.29 | 0.28 | 79 | axis 0 |
| PTO earlier short | third | 29.82 | 0.58 | 52 | axis 0 |
| PTO long (zone [010]) | full | 0.42 | 25.40 | 61 | **axis 1 — wrong zone** |
| PTO long (zone [010]) | third | 0.66 | 31.47 | 48 | **axis 1 — wrong zone** |
| BTO long | full | 8.96 | 0.33 | 27 | axis 0 |
| BTO long | third | 11.95 | 0.53 | 23 | axis 0 |
| Boracite long | full | 6.78 | 0.59 | 11 | axis 0 |
| Boracite long | third | 9.73 | 2.76 | 3.5 | axis 0 |

All units px. On the non-polar axis the even part is not merely small — it is *uncorrelated*
between the two PTO runs (r = −0.047), i.e. pure noise with no structure. On the polar axis
it is reproducible at r = 1.000. That contrast is the result.

This diagnostic is self-controlled, needs no repeat run, and independently reproduces the
zone-axis conclusion of §3 — note the PTO long-range row has the ratio inverted.

### Prefer the full box over the top-third box

The full box is markedly more reproducible. Mean even part across the two runs:

- full box: −3.687 vs −3.700 px (differ by **0.013 px**)
- third box: −4.585 vs −1.160 px (differ by **3.4 px**, a systematic offset, not scatter)

Combined with §2 — where the top-third box inflated the *artefact's* apparent significance —
there is no longer a case for the top-third box. Use the full box.

### Minor discrepancy worth checking cluster-side

The two PTO runs' aperture radii differ by 2 px (1337 vs 1335, ratios 1.01481 vs 1.01521),
a ~0.15 % difference in reciprocal-space calibration. Total intensity matches to 4e-5. Likely
threshold sensitivity, but if the two runs used different supercell sizes or sampling that
would explain the third-box systematic offset above. Check both run directories'
`parameters.txt` and supercell settings.
