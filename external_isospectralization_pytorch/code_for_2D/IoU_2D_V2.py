#!/usr/bin/env python3
"""
IoU_2D_V2.py
=================
A GUI‑driven tool for **maximum Intersection‑over‑Union (IoU)** between two
planar meshes (OBJ / PLY).

New in this version
-------------------
* **Centroid jittering** – after normal centroid alignment, the *movable* shape
  is translated over a user‑specified grid of offsets, and a full 360 ° rotation
  sweep is performed at **each** offset. The single best IoU over *all* angle ×
  offset combinations is reported.
* Two new CLI flags:
  * `--jitter-range R` (default 0): maximum translation in ±x, ±y (same units as
    your mesh, e.g.
    millimetres). `R=0` reproduces the old behaviour.
  * `--jitter-step S` (default 0.05): step size for the translation grid.

Quick start
-----------
```bash
pip install trimesh shapely numpy tqdm           # tqdm → nice progress bar
python iou_calculator.py --jitter-range 0.2 --jitter-step 0.05
```
Choose your two meshes in the file dialogs; the script then searches the full
(jitter × rotation) space for the best overlap.

Notes
-----
* Meshes must be essentially 2‑D (flat on the XY‑plane).
* The search cost grows with `(2R/S + 1)^2 × (360/angle_step)` – keep `R` and
  `S` modest if your shapes are complex.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

# tqdm is optional – fall back gracefully if unavailable
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable  # dummy wrapper

# ---------------------------------------------------------------------------
# GUI file dialog (tkinter)
# ---------------------------------------------------------------------------
try:
    from tkinter import Tk, filedialog, messagebox
except ImportError as exc:  # pragma: no cover
    raise ImportError("tkinter is required for the GUI file picker but is missing.") from exc

###############################################################################
# Geometry helpers
###############################################################################

def mesh_to_polygon(path: str | Path) -> Polygon:
    """Load an OBJ/PLY mesh and convert it to a 2‑D `shapely.Polygon`."""
    mesh = trimesh.load(str(path), force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Mesh at '{path}' is empty or unsupported.")

    vertices_2d = mesh.vertices[:, :2]

    triangles = []
    for face in mesh.faces:
        pts = vertices_2d[face]
        if np.linalg.matrix_rank(pts - pts[0]) < 2:  # degenerate in 2‑D
            continue
        poly = Polygon(pts)
        if poly.is_valid and poly.area > 0:
            triangles.append(poly)

    if not triangles:
        raise ValueError(f"No valid planar faces found in '{path}'.")

    return unary_union(triangles)


def align_centroid(poly: Polygon) -> Polygon:
    cx, cy = poly.centroid.coords[0]
    return translate(poly, xoff=-cx, yoff=-cy)


def iou(poly1: Polygon, poly2: Polygon) -> float:
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return 0.0 if union == 0 else inter / union

###############################################################################
# Exhaustive search: translations × rotations
###############################################################################

def best_iou_over_grid(
    poly_fixed: Polygon,
    poly_movable: Polygon,
    angle_step: int = 1,
    jitter_range: float = 0.2,
    jitter_step: float = 0.005,
) -> Tuple[float, int, float, float]:
    """Return (best_iou, best_angle_deg, best_dx, best_dy)."""

    # Build translation offsets
    if jitter_range <= 0:
        offsets = [(0.0, 0.0)]
    else:
        coords = np.arange(-jitter_range, jitter_range + 1e-9, jitter_step)
        offsets = list(product(coords, coords))

    total_iters = len(offsets) * (360 // angle_step)
    pbar = tqdm(total=total_iters, desc="Searching", unit="conf")

    best_iou = 0.0
    best_angle = 0
    best_dx = best_dy = 0.0

    for dx, dy in offsets:
        for angle in range(0, 360, angle_step):
            rotated = rotate(poly_movable, angle, origin=(0, 0), use_radians=False)
            moved = translate(rotated, xoff=dx, yoff=dy)
            score = iou(poly_fixed, moved)
            if score > best_iou:
                best_iou, best_angle, best_dx, best_dy = score, angle, dx, dy
            pbar.update(1)
    pbar.close()
    return best_iou, best_angle, best_dx, best_dy

###############################################################################
# Main routine
###############################################################################

def choose_file(title: str) -> str | None:
    filetypes = [
        ("Mesh files", "*.obj *.ply"),
        ("OBJ files", "*.obj"),
        ("PLY files", "*.ply"),
        ("All files", "*.*"),
    ]
    return filedialog.askopenfilename(title=title, filetypes=filetypes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Maximum IoU between two planar shapes")
    parser.add_argument("--angle-step", type=int, default=1,
                        help="Rotation increment in degrees (default 1°; smaller ⇒ slower)")
    parser.add_argument("--jitter-range", type=float, default=0.0,
                        help="Translate movable shape in ±x,±y up to this value (default 0 = off)")
    parser.add_argument("--jitter-step", type=float, default=0.05,
                        help="Grid spacing for jitter translations (default 0.05)")
    args = parser.parse_args()

    root = Tk(); root.withdraw()  # hidden root window

    shape_a_path = choose_file("Select *FIRST* shape (OBJ/PLY)")
    if not shape_a_path:
        messagebox.showinfo("IoU Calculator", "No file selected – exiting.")
        return

    shape_b_path = choose_file("Select *SECOND* shape (OBJ/PLY)")
    if not shape_b_path:
        messagebox.showinfo("IoU Calculator", "Second file not selected – exiting.")
        return

    root.destroy()

    print("Loading meshes …")
    try:
        poly_a = align_centroid(mesh_to_polygon(shape_a_path))
        poly_b = align_centroid(mesh_to_polygon(shape_b_path))
    except Exception as err:
        print(f"Error loading meshes: {err}")
        return

    print(
        f"Searching translations ±{args.jitter_range} (step {args.jitter_step})"
        f" and rotations 0–359° (step {args.angle_step}°)…"
    )

    best_iou, best_angle, best_dx, best_dy = best_iou_over_grid(
        poly_fixed=poly_a,
        poly_movable=poly_b,
        angle_step=args.angle_step,
        jitter_range=args.jitter_range,
        jitter_step=args.jitter_step,
    )

    print(
        "\nRESULTS\n-------"
        f"\nBest IoU   : {best_iou:.6f}"
        f"\nAngle (°)  : {best_angle}"
        f"\nOffset x   : {best_dx}"
        f"\nOffset y   : {best_dy}"
    )


if __name__ == "__main__":
    main()
