#!/usr/bin/env python3
"""
meshmaker2d.py

Reads a thin STL or OBJ of a flat 2D extrusion,
isolates the top face, and resamples it via a
SciPy-based constrained Delaunay pipeline that
mimics the shipped oval’s uniform Poisson-Delaunay style.

Outputs:
  - mesh.vert
  - mesh.triv
  - <name>.obj
"""

import os
import numpy as np
import meshio
import tkinter as tk
from tkinter import filedialog, simpledialog
from collections import Counter, defaultdict

from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay

from shape_library import prepare_mesh  # only need prepare_mesh for boundary extraction

############### New stuff 
def boundary_loops_from_tris(VERT, TRIV):
    """
    Return (outer_loop_idx, [hole_loop_idx_1, hole_loop_idx_2, ...]),
    where each is a 1D np.array of vertex indices ordered around the loop.
    """
    # Collect all triangle edges (unordered)
    e01 = TRIV[:, [0, 1]]
    e12 = TRIV[:, [1, 2]]
    e20 = TRIV[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges = np.sort(edges, axis=1)  # (i,j) with i<j

    # Boundary edges appear exactly once
    counts = Counter(map(tuple, edges))
    bedges = [e for e, c in counts.items() if c == 1]
    if not bedges:
        raise RuntimeError("No boundary edges found; is the mesh closed?")

    # Build adjacency on the boundary graph
    nbrs = defaultdict(list)
    for i, j in bedges:
        nbrs[i].append(j)
        nbrs[j].append(i)

    # Sanity: manifold boundary -> degree 2 at boundary verts
    # (We won't hard-fail; we'll still try to walk.)

    # Walk each loop
    loops = []
    visited = set()
    for start in list(nbrs.keys()):
        if start in visited:
            continue
        loop = [start]
        prev = None
        cur = start
        while True:
            neigh = nbrs[cur]
            # Pick the next neighbor that's not the previous
            nxt = neigh[0] if len(neigh) == 1 or neigh[0] != prev else neigh[1]
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
            if cur in visited:  # guard against non-manifold weirdness
                break
        visited.update(loop)
        loops.append(np.array(loop, dtype=int))

    if not loops:
        raise RuntimeError("Failed to assemble boundary loops.")

    # Choose outer loop as the one with largest |area| (shoelace)
    def signed_area(idx):
        P = VERT[idx]
        x, y = P[:, 0], P[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

    loops_sorted = sorted(loops, key=lambda L: abs(signed_area(L)), reverse=True)
    outer = loops_sorted[0]
    holes = loops_sorted[1:]
    return outer, holes

#################

def pick_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select your flat STL or OBJ",
        filetypes=[("STL/OBJ","*.stl *.obj")])
    if not path:
        raise SystemExit("No file selected.")
    return path

def load_top_face(path):
    """
    Reads an STL or OBJ, keeps only upward-facing triangles,
    and returns VERT0 (N×2) and TRIV0 (M×3, 0-based).
    """
    mesh = meshio.read(path)
    pts3d = mesh.points[:, :3]
    # pick the first triangle block
    if "triangle" in mesh.cells_dict:
        tris = mesh.cells_dict["triangle"]
    else:
        tris = mesh.cells[0].data

    # select only those triangles whose normal's z > 0
    good = []
    for tri in tris:
        p0, p1, p2 = pts3d[tri]
        n = np.cross(p1 - p0, p2 - p0)
        if n[2] > 0:
            good.append(tri)
    good = np.array(good, dtype=int)

    # keep only vertices used by those top triangles
    used = np.unique(good.flatten())
    VERT0 = pts3d[used, :2]  # drop Z

    # remap old indices -> new 0-based indices
    remap = -np.ones(pts3d.shape[0], dtype=int)
    remap[used] = np.arange(len(used), dtype=int)
    TRIV0 = remap[good]      # shape (M,3)

    return VERT0, TRIV0

def write_outputs(name, V, T):
    """
    Given V (N×2) and T (M×3, 0-based), writes:
      mesh.vert   (each line: x y 0)
      mesh.triv   (each line: i j k, 1-based)
      {name}.obj  (full OBJ for visualization)
    """
    # mesh.vert
    with open("mesh.vert", "w") as fv:
        for x, y in V:
            fv.write(f"{x:.6f} {y:.6f} 0\n")

    # mesh.triv (1-based)
    with open("mesh.triv", "w") as ft:
        for a, b, c in (T + 1):
            ft.write(f"{a} {b} {c}\n")

    # write OBJ
    with open(f"{name}.obj", "w") as fo:
        for x, y in V:
            fo.write(f"v {x:.6f} {y:.6f} 0\n")
        for a, b, c in (T + 1):
            fo.write(f"f {a} {b} {c}\n")

def resample_scipy(VERT, TRIV, npts):
    """
    Mimics shape_library.resample but uses SciPy Delaunay
    to avoid VisPy’s edge-splitting bug. Returns (V1, T1).
    """
    # 1) extract ALL boundary loops (outer + holes)
    outer_idx, hole_indices = boundary_loops_from_tris(VERT, TRIV)
    loop_pts = [VERT[outer_idx, :2]] + [VERT[h, :2] for h in hole_indices]
    outer_pts = loop_pts[0]
    hole_pts  = loop_pts[1:]

    def subdivide_ring(ring_pts, target_step=0.05):
        pts = []
        n = len(ring_pts)
        for i in range(n):
            a = ring_pts[i]
            b = ring_pts[(i + 1) % n]
            dist = np.linalg.norm(b - a)
            steps = max(1, int(dist / target_step))
            for j in range(steps):
                t = j / steps
                pts.append(a + t * (b - a))
        return np.array(pts)

    bound_pts = np.vstack([subdivide_ring(r) for r in loop_pts])

    # 2) jittered grid interior candidates
    minx, maxx = VERT[:,0].min(), VERT[:,0].max()
    miny, maxy = VERT[:,1].min(), VERT[:,1].max()
    d = max(2, int(npts / 5))
    xv = np.linspace(minx, maxx, d)
    yv = np.linspace(miny, maxy, d)
    grid = np.stack(np.meshgrid(xv, yv), axis=-1).reshape(-1,2)
    noise = (np.random.rand(*grid.shape) - 0.5) * 0.9 * max(maxx-minx, maxy-miny) / d
    grid += noise

    # 3) keep only those inside the polygon
    poly = Polygon(outer_pts, holes=[hp.tolist() for hp in hole_pts])  # holes OK
    inside_mask = np.array([poly.contains(Point(p)) for p in grid])
    interior_pts = grid[inside_mask]

    # 4) farthest-point sampling to get exactly npts
    pts = np.vstack([bound_pts, interior_pts])
    selected = list(range(bound_pts.shape[0]))
    while len(selected) < npts:
        dists = np.min(
            np.linalg.norm(pts[:,None,:2] - pts[selected][None,:,:2], axis=2),
            axis=1
        )
        idx = int(np.argmax(dists))
        selected.append(idx)
    pts = pts[selected,:]

    # 5) SciPy Delaunay
    tri = Delaunay(pts)
    TT = tri.simplices.copy()

    # 6) filter triangles whose centroid lies outside
    centroids = pts[TT].mean(axis=1)
    mask = np.array([poly.contains(Point(c)) for c in centroids])
    TT = TT[mask]

    return pts, TT

def main():
    path = pick_file()
    base = os.path.splitext(os.path.basename(path))[0]

    # prompt for desired vertex count
    root = tk.Tk()
    root.withdraw()
    npts = simpledialog.askinteger(
        "Vertices",
        "How many vertices in the resampled mesh?",
        initialvalue=500, minvalue=10, maxvalue=5000
    )
    if npts is None:
        raise SystemExit("No vertex count provided.")

    print(f"Loading top face from {path}…")
    V0, T0 = load_top_face(path)
    print(f"  extracted {V0.shape[0]} verts, {T0.shape[0]} tris")

    print(f"Resampling to ~{npts} vertices (SciPy Delaunay)…")
    V1, T1 = resample_scipy(V0, T0, npts)
    print(f"  result: {V1.shape[0]} verts, {T1.shape[0]} tris")

    print("Writing mesh.vert, mesh.triv, and", f"{base}.obj")
    write_outputs(base, V1, T1)
    print("Done.")

if __name__ == "__main__":
    main()
