"""Make zenodo_upload_slim/ from zenodo_upload/: crop patterns to the bandlimited region, float32.

Patterns are 4096 x 4096 with the 2/3 antialiasing bandlimit at r = 1365 px; outside it the
intensity is numerical noise (~1e-14 of the total). Crop keeps the central 2732 x 2732
(pixels 682:3414), so the pattern centre stays at N/2 and box mirrors (N + 1 - hi) map exactly.
abTEM output is float32, so the float64 upcast carried no information.

Usage: python slim_zenodo.py [file ...]   (no args = everything)
"""

import shutil
import sys
from pathlib import Path

import numpy as np

SRC, DST = Path("zenodo_upload"), Path("zenodo_upload_slim")
N0, N1 = 4096, 2732
OFF = (N0 - N1) // 2  # 682
CROP = slice(OFF, OFF + N1)


def box_sums(img, coords):
    y0, y1, x0, x1 = (int(c) for c in coords)  # tilt grids store integer coords as float64
    return img[y0:y1, x0:x1].sum(dtype=np.float64)


def convert_tilt(src, dst):
    d = dict(np.load(src))
    full = d["diffraction_images"]
    lead = full.shape[:-2]
    out = full[..., CROP, CROP].astype(np.float32)
    assert np.array_equal(out, full[..., CROP, CROP]), "data was not float32-exact"

    # verify on every frame; box sums before vs after crop must match
    b1, b2 = d["horizontal_boxes_box1_coords"], d["horizontal_boxes_box2_coords"]
    worst = 0.0
    for idx in np.ndindex(*lead):
        c1, c2 = b1[idx], b2[idx]
        assert c1.min() >= OFF and c1.max() <= OFF + N1, (src, idx, c1)
        assert c2.min() >= OFF and c2.max() <= OFF + N1, (src, idx, c2)
        a_old = box_sums(full[idx], c1) - box_sums(full[idx], c2)
        a_new = box_sums(out[idx], c1 - OFF) - box_sums(out[idx], c2 - OFF)
        a_ref = d["horizontal_boxes_intensity_asymmetry"][idx]
        worst = max(worst, abs(a_new - a_old), abs(a_new - a_ref))
        lost = 1 - out[idx].sum(dtype=np.float64) / full[idx].sum(dtype=np.float64)
        assert abs(lost) < 1e-10, (src, idx, lost)
    del full

    d["diffraction_images"] = out
    d["horizontal_boxes_box1_coords"] = b1 - OFF
    d["horizontal_boxes_box2_coords"] = b2 - OFF
    d["crop_offset_px"] = np.int64(OFF)
    d["original_shape_px"] = np.int64(N0)
    np.savez_compressed(dst, **d)
    return f"max |box asym diff| {worst:.2e}"


def convert_thickness(src, dst):
    d = dict(np.load(src))
    full = d["patterns"]
    out = full[..., CROP, CROP]
    lost = 1 - out.sum(axis=(1, 2), dtype=np.float64) / full.sum(axis=(1, 2), dtype=np.float64)
    assert np.abs(lost).max() < 1e-10, lost
    d["patterns"] = np.ascontiguousarray(out, dtype=np.float32)
    d["crop_offset_px"] = np.int64(OFF)
    d["original_shape_px"] = np.int64(N0)
    np.savez_compressed(dst, **d)
    return f"max lost fraction {np.abs(lost).max():.1e}"


def handle(src):
    dst = DST / src.relative_to(SRC)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.name == "sum.npz":
        return "skipped (duplicate of mean_patterns.npz)"
    if src.name.endswith("_tilt.npz") or src.name.endswith("_tilt_grid.npz"):
        return convert_tilt(src, dst)
    if src.name == "mean_patterns.npz":
        return convert_thickness(src, dst)
    shutil.copy2(src, dst)  # small files and the PTO strip export (indexed in 4096 px coords)
    return "copied"


if __name__ == "__main__":
    files = [Path(a) for a in sys.argv[1:]] or sorted(p for p in SRC.rglob("*") if p.is_file())
    for f in files:
        print(f, "->", handle(f), flush=True)
