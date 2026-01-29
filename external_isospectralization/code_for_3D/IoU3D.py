import numpy as np
import trimesh
import tkinter as tk
from tkinter import filedialog

def pick_mesh(title):
    return filedialog.askopenfilename(
        title=title,
        filetypes=[("mesh files", "*.ply *.obj *.stl *.off *.glb *.gltf"), ("all files", "*.*")]
    )

def load_as_trimesh(path):
    m = trimesh.load(path, force='mesh', process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)  # applies the scene/node transform(s)
    return m

def voxel_iou(mesh_a, mesh_b, resolution=128, padding=0.02):
    """
    Compute voxel IoU on a shared cubic grid.
    - resolution: number of voxels along the longest bbox side
    - padding: fraction of bbox size to pad on each side
    """
    # combined bounds
    bounds = np.vstack([mesh_a.bounds, mesh_b.bounds])
    bmin = bounds[:, 0:3].min(axis=0)
    bmax = bounds[:, 0:3].max(axis=0)

    size = bmax - bmin
    pad = padding * size.max()
    bmin = bmin - pad
    bmax = bmax + pad
    size = bmax - bmin

    # cubic voxels, pitch set by longest side / resolution
    pitch = size.max() / resolution

    # grid dimensions (ensure covers full box)
    dims = np.ceil(size / pitch).astype(int)
    # voxel centers
    xs = (np.arange(dims[0]) + 0.5) * pitch + bmin[0]
    ys = (np.arange(dims[1]) + 0.5) * pitch + bmin[1]
    zs = (np.arange(dims[2]) + 0.5) * pitch + bmin[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # inside tests (requires watertight for meaningful volume IoU)
    inside_a = mesh_a.contains(pts)
    inside_b = mesh_b.contains(pts)

    inter = np.logical_and(inside_a, inside_b).sum()
    union = np.logical_or(inside_a, inside_b).sum()
    iou = inter / union if union > 0 else 0.0

    # approximate voxel volumes (optional)
    voxel_vol = pitch ** 3
    vol_a = inside_a.sum() * voxel_vol
    vol_b = inside_b.sum() * voxel_vol
    vol_inter = inter * voxel_vol
    vol_union = union * voxel_vol

    return iou, pitch, dims, (vol_a, vol_b, vol_inter, vol_union)

def main():
    root = tk.Tk()
    root.withdraw()

    fA = pick_mesh("Select mesh A (e.g., GT)")
    if not fA:
        print("No mesh A selected.")
        return
    fB = pick_mesh("Select mesh B (e.g., reconstructed)")
    if not fB:
        print("No mesh B selected.")
        return

    A = load_as_trimesh(fA)
    B = load_as_trimesh(fB)

    print("Mesh A:", fA)
    print("Mesh B:", fB)
    print("A watertight:", A.is_watertight, "| B watertight:", B.is_watertight)

    # Try a couple resolutions to see stability
    for res in (5, 50, 55, 60, 64):
        iou, pitch, dims, vols = voxel_iou(A, B, resolution=res, padding=0.02)
        vol_a, vol_b, vol_inter, vol_union = vols
        print(f"\nResolution ~{res} (pitch={pitch:.6g}, dims={tuple(dims)})")
        print(f"IoU: {iou:.6f}")
        print(f"Approx vols: A={vol_a:.6g}, B={vol_b:.6g}, inter={vol_inter:.6g}, union={vol_union:.6g}")

if __name__ == "__main__":
    main()
