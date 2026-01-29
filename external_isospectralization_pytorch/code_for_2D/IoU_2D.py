#!/usr/bin/env python3
"""
iou_calculator.py
=================
A *one‑step* GUI tool to find the **maximum Intersection‑over‑Union (IoU)**
between two planar shapes supplied as OBJ or PLY meshes.

What’s new
----------
* **Native file‑picker** – a Tk "open file" dialog pops up twice so you can
  browse to each file instead of typing paths.
* Command‑line still supports `--angle-step` to trade accuracy for speed.

Quick start
-----------
```bash
pip install trimesh shapely numpy tqdm  # tqdm optional – nice progress bar
python iou_calculator.py                # two dialogs will appear
```

Dependencies: only the Python std‑lib (**tkinter**) plus *trimesh*, *shapely*,
*numpy*, and optionally *tqdm*.

Assumptions
-----------
* The meshes are essentially 2‑D (z≈0). Faces that collapse in 2‑D are skipped.
* Works on Windows, macOS, Linux – tkinter is bundled with most Python installs.
"""

from __future__ import annotations

import argparse
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
        # degenerate if the 2×2 matrix of edge vectors is rank < 2
        if np.linalg.matrix_rank(pts - pts[0]) < 2:
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


def sweep(poly_fixed: Polygon, poly_movable: Polygon, step_deg: int = 1) -> Tuple[float, int]:
    best_iou, best_angle = 0.0, 0
    for angle in tqdm(range(0, 360, step_deg), desc="Rotating", unit="deg", leave=False):
        rotated = rotate(poly_movable, angle, origin=(0, 0), use_radians=False)
        score = iou(poly_fixed, rotated)
        if score > best_iou:
            best_iou, best_angle = score, angle
    return best_iou, best_angle

###############################################################################
# Main routine
###############################################################################

def choose_file(title: str) -> str | None:
    """Pop up a file‑open dialog and return the selected path or *None*."""
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
                        help="Rotation increment in degrees (default 1°; smaller ⇒ slower but finer)")
    args = parser.parse_args()

    # Spin up Tk root (hidden) for dialogs
    root = Tk()
    root.withdraw()  # keep root window invisible

    shape_a_path = choose_file("Select *FIRST* shape (OBJ/PLY)")
    if not shape_a_path:
        messagebox.showinfo("IoU Calculator", "No file selected – exiting.")
        return

    shape_b_path = choose_file("Select *SECOND* shape (OBJ/PLY)")
    if not shape_b_path:
        messagebox.showinfo("IoU Calculator", "Second file not selected – exiting.")
        return

    root.update()  # refresh GUI (optional but polite)
    root.destroy()  # close the hidden root window

    # Load & preprocess
    print("Loading meshes…")
    try:
        poly_a = align_centroid(mesh_to_polygon(shape_a_path))
        poly_b = align_centroid(mesh_to_polygon(shape_b_path))
    except Exception as err:
        print(f"Error loading meshes: {err}")
        return

    print(f"Sweeping {360 // args.angle_step} orientations (step = {args.angle_step}°)…")
    best_iou, best_angle = sweep(poly_a, poly_b, step_deg=args.angle_step)

    print(f"\nBest IoU  : {best_iou:.6f}\nAngle (°) : {best_angle}")


if __name__ == "__main__":
    main()
