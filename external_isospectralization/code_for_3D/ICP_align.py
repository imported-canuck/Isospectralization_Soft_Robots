import argparse
import numpy as np
import open3d as o3d
import trimesh


def pick_mesh_path(title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Meshes (.obj, .ply)", "*.obj *.ply"),
            ("Wavefront OBJ", "*.obj"),
            ("Polygon File Format PLY", "*.ply"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def mesh_to_pcd(mesh: o3d.geometry.TriangleMesh, n_points: int) -> o3d.geometry.PointCloud:
    return mesh.sample_points_uniformly(number_of_points=n_points)


def estimate_base_voxel_from_sampling(pcd: o3d.geometry.PointCloud) -> float:
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 1000:
        raise ValueError("Too few sampled points to estimate scale. Increase --n_points.")

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    dists = np.empty(pts.shape[0], dtype=np.float64)

    for i, p in enumerate(pts):
        _, _, dist2 = kdtree.search_knn_vector_3d(p, 2)
        dists[i] = np.sqrt(dist2[1])

    med = float(np.median(dists))

    aabb = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(aabb.get_extent()))
    if diag <= 0:
        raise ValueError("Degenerate point cloud bounding box.")

    base = np.clip(3.0 * med, 1e-6 * diag, 0.05 * diag)
    return float(base)


def preprocess(pcd: o3d.geometry.PointCloud, voxel: float):
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5.0 * voxel, max_nn=100),
    )
    return pcd_down, fpfh


def load_as_trimesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def voxel_iou_shared_grid(mesh_a: trimesh.Trimesh,
                          mesh_b: trimesh.Trimesh,
                          pitch: float,
                          padding_frac: float = 0.02):
    bounds = np.vstack([mesh_a.bounds, mesh_b.bounds])
    bmin = bounds.min(axis=0)
    bmax = bounds.max(axis=0)

    size = bmax - bmin
    pad = padding_frac * size.max()
    bmin = bmin - pad
    bmax = bmax + pad
    size = bmax - bmin

    dims = np.ceil(size / pitch).astype(int)

    max_vox = int(dims[0]) * int(dims[1]) * int(dims[2])
    if max_vox > 250_000_000:
        raise MemoryError(
            f"Voxel grid too large: dims={tuple(dims)} (~{max_vox:,} voxels). "
            f"Increase pitch (coarser grid) via --iou_pitch_mult."
        )

    xs = (np.arange(dims[0]) + 0.5) * pitch + bmin[0]
    ys = (np.arange(dims[1]) + 0.5) * pitch + bmin[1]
    zs = (np.arange(dims[2]) + 0.5) * pitch + bmin[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    inside_a = mesh_a.contains(pts)
    inside_b = mesh_b.contains(pts)

    inter = np.logical_and(inside_a, inside_b).sum()
    union = np.logical_or(inside_a, inside_b).sum()
    iou = float(inter / union) if union > 0 else 0.0

    return iou, dims


def compute_icp_transform_open3d(target_path: str, source_path: str, n_points: int):
    tgt_mesh = o3d.io.read_triangle_mesh(target_path)
    src_mesh = o3d.io.read_triangle_mesh(source_path)
    if tgt_mesh.is_empty() or src_mesh.is_empty():
        raise ValueError("One of the meshes failed to load (empty).")

    tgt_pcd = mesh_to_pcd(tgt_mesh, n_points)
    src_pcd = mesh_to_pcd(src_mesh, n_points)

    base_voxel = estimate_base_voxel_from_sampling(tgt_pcd)

    voxel_ransac = 4.0 * base_voxel
    src_down, src_fpfh = preprocess(src_pcd, voxel_ransac)
    tgt_down, tgt_fpfh = preprocess(tgt_pcd, voxel_ransac)

    dist_coarse = 1.5 * voxel_ransac
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_coarse,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_coarse),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    T = ransac.transformation

    final_rmse = float("nan")
    for mult in [2.0, 1.0, 0.5]:
        voxel = base_voxel * mult
        max_corr = 1.5 * voxel

        src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel, max_nn=30))
        tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel, max_nn=30))

        loss = o3d.pipelines.registration.TukeyLoss(k=max_corr)
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        icp = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd,
            max_correspondence_distance=max_corr,
            init=T,
            estimation_method=est,
        )
        T = icp.transformation
        final_rmse = icp.inlier_rmse

    return T, final_rmse, base_voxel


def visualize_pair(target_path: str, source_path: str, n_points: int, T=None, title=""):
    """
    Visualizes sampled point clouds (fast):
      - target in blue
      - source in orange (optionally transformed by T)
    """
    tgt_mesh = o3d.io.read_triangle_mesh(target_path)
    src_mesh = o3d.io.read_triangle_mesh(source_path)

    tgt_pcd = mesh_to_pcd(tgt_mesh, n_points)
    src_pcd = mesh_to_pcd(src_mesh, n_points)

    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])   # blue
    src_pcd.paint_uniform_color([1, 0.706, 0])       # orange

    if T is not None:
        src_pcd.transform(np.asarray(T, dtype=float))

    o3d.visualization.draw_geometries([tgt_pcd, src_pcd], window_name=title or "Open3D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_points", type=int, default=100000, help="Points sampled per mesh for ICP/vis")
    ap.add_argument("--iou_pitch_mult", type=float, default=2.0,
                    help="IoU voxel pitch = (iou_pitch_mult * base_voxel). Larger => coarser IoU grid.")
    ap.add_argument("--padding", type=float, default=0.02, help="Padding fraction for IoU grid bbox")
    ap.add_argument("--vis_pre", action="store_true", help="Visualize pre-alignment (moving vs stable)")
    ap.add_argument("--vis_post", action="store_true", help="Visualize post-alignment (moving transformed vs stable)")
    args = ap.parse_args()

    target_path = pick_mesh_path("Select the STABLE (target) mesh (.obj/.ply)")
    if not target_path:
        return
    source_path = pick_mesh_path("Select the MOVING mesh to align onto the target (.obj/.ply)")
    if not source_path:
        return

    # Optional pre-visualization
    if args.vis_pre:
        visualize_pair(
            target_path, source_path,
            n_points=min(args.n_points, 80000),
            T=None,
            title="Pre-alignment (target=blue, source=orange)"
        )

    # Load trimesh versions for IoU (before/after)
    A = load_as_trimesh(target_path)
    B = load_as_trimesh(source_path)

    # ICP to get transform and scale
    T, rmse, base_voxel = compute_icp_transform_open3d(target_path, source_path, args.n_points)
    R = T[:3, :3]

    # Shared pitch for fair comparison
    pitch = float(args.iou_pitch_mult * base_voxel)

    # Naive IoU (no alignment)
    try:
        iou_naive, _ = voxel_iou_shared_grid(A, B, pitch=pitch, padding_frac=args.padding)
    except Exception as e:
        raise RuntimeError(
            f"Naive IoU computation failed. Common fix: `pip install rtree`.\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    # IoU after alignment
    B_aligned = B.copy()
    B_aligned.apply_transform(T)
    try:
        iou_aligned, _ = voxel_iou_shared_grid(A, B_aligned, pitch=pitch, padding_frac=args.padding)
    except Exception as e:
        raise RuntimeError(
            f"Aligned IoU computation failed. Common fix: `pip install rtree`.\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    # Optional post-visualization
    if args.vis_post:
        visualize_pair(
            target_path, source_path,
            n_points=min(args.n_points, 80000),
            T=T,
            title="Post-alignment (target=blue, source-aligned=orange)"
        )

    # Print data 
    print("Naive IoU (before):", iou_naive)
    print("IoU (after ICP):", iou_aligned)
    print("RMSE:", rmse)
    print("R (ideal rotation matrix):\n", R)


if __name__ == "__main__":
    main()