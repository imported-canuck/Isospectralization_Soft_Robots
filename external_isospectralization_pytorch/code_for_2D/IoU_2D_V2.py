#!/usr/bin/env python3
"""
IoU_2D_V2.py
=================
A GUI-driven tool for maximum Intersection-over-Union (IoU) between two
planar meshes (OBJ / PLY), using an exhaustive grid search over rotations
and translations.

What’s new vs. the naive version
--------------------------------
• Centroid jittering: the movable shape is translated over a symmetric grid
  of offsets and rotated at each offset; we report the single best IoU.
• One overlay per IoU: union area is computed by inclusion–exclusion to avoid
  a second overlay call.
• Streaming grid: we iterate the (dx, dy) grid directly.
• Correct counts: angles use ceil(360/Δθ); translations use per-axis
  2*floor(R/Δt)+1 (symmetric, includes 0).
• AABB reject: skip expensive overlay when bounding boxes are disjoint.

Notes
-----
• Meshes must be 2-D (flat on the XY-plane).
• Search cost ~ N = Nθ * Nxy, with
    Nθ  = ceil(360 / angle_step),
    Nxy = (2 * floor(R / jitter_step) + 1)^2
• Typical usage:
    python IoU_2D_V2.py --angle-step 1 --jitter-range 1.0 --jitter-step 0.1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# tqdm is optional – fall back gracefully if unavailable
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, total=None, **kwargs):  # type: ignore
        # Minimal stand-in that behaves like range(total) if iterable is None
        if iterable is None and total is not None:
            return range(total)
        return iterable if iterable is not None else []

# ---------------------------------------------------------------------------
# GUI file dialog (tkinter)
# ---------------------------------------------------------------------------
try:
    from tkinter import Tk, filedialog, messagebox
except ImportError as exc:  # pragma: no cover
    raise ImportError("tkinter is required for the GUI file picker but is missing.") from exc


# =============================================================================
# Geometry helpers
# =============================================================================

def mesh_to_polygon(path: str | Path) -> BaseGeometry:
    """Load an OBJ/PLY mesh and convert it to a 2-D Shapely geometry (Polygon or MultiPolygon)."""
    mesh = trimesh.load(str(path), force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Mesh at '{path}' is empty or unsupported.")

    # Drop Z; assume planar on XY.
    vertices_2d = mesh.vertices[:, :2]

    triangles: list[Polygon] = []
    for face in mesh.faces:
        pts = vertices_2d[face]
        # Filter degenerate triangles in 2D
        if np.linalg.matrix_rank(pts - pts[0]) < 2:
            continue
        poly = Polygon(pts)
        if poly.is_valid and poly.area > 0:
            triangles.append(poly)

    if not triangles:
        raise ValueError(f"No valid planar faces found in '{path}'.")

    geom = unary_union(triangles)
    # Optional robustness nudge (uncomment if you hit topology errors):
    # geom = geom.buffer(0)
    return geom


def align_centroid(geom: BaseGeometry) -> BaseGeometry:
    """Translate geometry so its centroid is at the origin."""
    cx, cy = geom.centroid.coords[0]
    return translate(geom, xoff=-cx, yoff=-cy)


def iou(poly1: BaseGeometry, poly2: BaseGeometry, area1: float | None = None, area2: float | None = None) -> float:
    """
    IoU via a single overlay:
      IoU = area(intersection) / area(union)
          = I / (A + B - I)
    """
    if area1 is None:
        area1 = poly1.area
    if area2 is None:
        area2 = poly2.area
    inter_area = poly1.intersection(poly2).area  # one overlay
    union_area = area1 + area2 - inter_area
    return 0.0 if union_area == 0 else inter_area / union_area


def boxes_disjoint(a: BaseGeometry, b: BaseGeometry) -> bool:
    """Axis-aligned bounding-box disjointness test (cheap reject)."""
    ax1, ay1, ax2, ay2 = a.bounds
    bx1, by1, bx2, by2 = b.bounds
    return (ax2 < bx1) or (bx2 < ax1) or (ay2 < by1) or (by2 < ay1)


# =============================================================================
# Exhaustive search: translations × rotations
# =============================================================================

def best_iou_over_grid(
    poly_fixed: BaseGeometry,
    poly_movable: BaseGeometry,
    angle_step: int = 90,
    jitter_range: float = 3.0,
    jitter_step: float = 0.1,
) -> Tuple[float, int, float, float]:
    """Return (best_iou, best_angle_deg, best_dx, best_dy)."""

    if angle_step <= 0:
        raise ValueError("angle_step must be > 0")
    if jitter_range < 0:
        raise ValueError("jitter_range must be >= 0")
    if jitter_range > 0 and jitter_step <= 0:
        raise ValueError("jitter_step must be > 0 when jitter_range > 0")

    # Build symmetric coordinate vector per axis, includes 0
    if jitter_range <= 0:
        coords = np.array([0.0], dtype=float)
    else:
        kmax = int(math.floor(jitter_range / jitter_step))
        coords = jitter_step * np.arange(-kmax, kmax + 1, dtype=float)

    # Angle samples in [0, 360) with step angle_step
    nang = int(math.ceil(360.0 / float(angle_step)))

    total_iters = coords.size * coords.size * nang
    pbar = tqdm(total=total_iters, desc="Searching", unit="pose")

    best_iou = 0.0
    best_angle = 0
    best_dx = best_dy = 0.0

    area_fixed = poly_fixed.area
    area_movable = poly_movable.area

    for dx in coords:
        for dy in coords:
            for i in range(nang):
                angle = i * angle_step
                rotated = rotate(poly_movable, angle, origin=(0, 0), use_radians=False)
                moved = translate(rotated, xoff=dx, yoff=dy)

                # Cheap reject
                if boxes_disjoint(poly_fixed, moved):
                    score = 0.0
                else:
                    score = iou(poly_fixed, moved, area1=area_fixed, area2=area_movable)

                if score > best_iou:
                    best_iou, best_angle, best_dx, best_dy = score, angle, dx, dy
                if hasattr(pbar, "update"):
                    pbar.update(1)
    # Close tqdm if it’s real
    if hasattr(pbar, "close"):
        pbar.close()

    return best_iou, best_angle, best_dx, best_dy


# =============================================================================
# Main routine (GUI file picker + CLI args)
# =============================================================================

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
    parser.add_argument("--angle-step", type=int, default=90,
                        help="Rotation increment in degrees (default 1°; smaller ⇒ slower)")
    parser.add_argument("--jitter-range", type=float, default=1.5,
                        help="Translate movable shape in ±x,±y up to this value (default 0 = off)")
    parser.add_argument("--jitter-step", type=float, default=0.1,
                        help="Grid spacing for jitter translations (default 0.05)")
    args = parser.parse_args()

    # Hidden root window for file dialogs
    root = Tk()
    root.withdraw()

    shape_a_path = choose_file("Select *FIRST* shape (OBJ/PLY)")
    if not shape_a_path:
        messagebox.showinfo("IoU Calculator", "No file selected – exiting.")
        root.destroy()
        return

    shape_b_path = choose_file("Select *SECOND* shape (OBJ/PLY)")
    if not shape_b_path:
        messagebox.showinfo("IoU Calculator", "Second file not selected – exiting.")
        root.destroy()
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
        f"Searching translations ±{args.jitter_range} (step {args.jitter_step}) "
        f"and rotations 0–359° (step {args.angle_step}°)…"
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
