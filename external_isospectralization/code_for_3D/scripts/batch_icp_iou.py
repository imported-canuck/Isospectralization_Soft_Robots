# scripts/batch_icp_iou.py
import argparse
import csv
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt


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

    # trimesh.contains needs rtree installed (usually) and a watertight-ish mesh helps
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


def plot_from_csv(csv_path: Path, out_png: Path, title: str):
    rows = []
    with csv_path.open("r", newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError("No rows found in CSV; nothing to plot.")

    idx = np.array([int(r["mesh_index"]) for r in rows], dtype=int)
    iou_before = np.array([float(r["iou_naive"]) for r in rows], dtype=float)
    iou_after = np.array([float(r["iou_aligned"]) for r in rows], dtype=float)

    order = np.argsort(idx)
    idx = idx[order]
    iou_before = iou_before[order]
    iou_after = iou_after[order]

    plt.figure(figsize=(10, 5))
    plt.plot(idx, iou_before, marker="o", linewidth=2, label="Naive IoU (before)")
    plt.plot(idx, iou_after, marker="o", linewidth=2, label="IoU (after ICP)")
    plt.xlabel("Mesh index (mesh_000XX)")
    plt.ylabel("Voxel IoU")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def find_sources(bubble_root: Path, start: int, end: int):
    """
    Expects structure:
      bubble_root/
        mesh_00006/mesh_00006.obj
        mesh_00007/mesh_00007.obj
        ...
    """
    sources = []
    for k in range(start, end + 1):
        folder = bubble_root / f"mesh_{k:05d}"
        obj = folder / f"mesh_{k:05d}.obj"
        if obj.exists():
            sources.append((k, obj))
        else:
            # still record missing as skip
            sources.append((k, None))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True, help="Stable target mesh path (.obj/.ply)")
    ap.add_argument("--bubble_root", type=str, required=True, help="Bubble_Grasp_Deep_0.05x40 folder path")
    ap.add_argument("--start", type=int, default=6, help="Start mesh index (default 6)")
    ap.add_argument("--end", type=int, default=40, help="End mesh index (default 40)")
    ap.add_argument("--n_points", type=int, default=100000, help="Points sampled per mesh for ICP")
    ap.add_argument("--iou_pitch_mult", type=float, default=2.0,
                    help="IoU voxel pitch = (iou_pitch_mult * base_voxel). Larger => coarser grid.")
    ap.add_argument("--padding", type=float, default=0.02, help="Padding fraction for IoU bbox")
    ap.add_argument("--out_dir", type=str, default="out", help="Output folder (CSV + PNG)")
    ap.add_argument("--title", type=str, default="ICP + Voxel IoU vs Mesh Index", help="Plot title")
    args = ap.parse_args()

    target_path = Path(args.target)
    bubble_root = Path(args.bubble_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "icp_iou_results.csv"
    png_path = out_dir / "icp_iou_plot.png"

    if not target_path.exists():
        raise FileNotFoundError(f"Target mesh not found: {target_path}")
    if not bubble_root.exists():
        raise FileNotFoundError(f"bubble_root not found: {bubble_root}")

    # Load target once
    A = load_as_trimesh(str(target_path))

    sources = find_sources(bubble_root, args.start, args.end)

    fieldnames = [
        "mesh_index", "source_path",
        "iou_naive", "iou_aligned",
        "rmse", "base_voxel", "pitch",
        "status", "error"
    ]

    rows_out = []
    for k, src in sources:
        if src is None:
            print(f"[mesh_{k:05d}] SKIP (missing obj)")
            rows_out.append({
                "mesh_index": k,
                "source_path": "",
                "iou_naive": "",
                "iou_aligned": "",
                "rmse": "",
                "base_voxel": "",
                "pitch": "",
                "status": "missing",
                "error": "source obj not found"
            })
            continue

        print(f"[mesh_{k:05d}] running ICP/IoU...")
        try:
            B = load_as_trimesh(str(src))

            T, rmse, base_voxel = compute_icp_transform_open3d(
                str(target_path), str(src), args.n_points
            )
            pitch = float(args.iou_pitch_mult * base_voxel)

            # Naive
            iou_naive, _ = voxel_iou_shared_grid(A, B, pitch=pitch, padding_frac=args.padding)

            # Aligned
            B_aligned = B.copy()
            B_aligned.apply_transform(T)
            iou_aligned, _ = voxel_iou_shared_grid(A, B_aligned, pitch=pitch, padding_frac=args.padding)

            print(f"  naive IoU={iou_naive:.6f} | aligned IoU={iou_aligned:.6f} | rmse={rmse:.6g}")

            rows_out.append({
                "mesh_index": k,
                "source_path": str(src),
                "iou_naive": f"{iou_naive:.10f}",
                "iou_aligned": f"{iou_aligned:.10f}",
                "rmse": f"{rmse:.10g}",
                "base_voxel": f"{base_voxel:.10g}",
                "pitch": f"{pitch:.10g}",
                "status": "ok",
                "error": ""
            })

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows_out.append({
                "mesh_index": k,
                "source_path": str(src),
                "iou_naive": "",
                "iou_aligned": "",
                "rmse": "",
                "base_voxel": "",
                "pitch": "",
                "status": "error",
                "error": f"{type(e).__name__}: {e}"
            })

    # Write CSV cache
    with csv_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSaved results: {csv_path}")

    # Plot from CSV (so plot tweaks don't require recompute)
    plot_from_csv(csv_path, png_path, args.title)
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
