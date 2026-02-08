# external_isospectralization/code_for_3D/scripts/sphere_init_vs_cascade_plot.py
# Python 3.6 compatible

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---- make imports work regardless of launch location ----
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent  # .../code_for_3D
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# scripts in this repo assume cwd == code_for_3D
os.chdir(str(CODE_ROOT))

from shape_library import load_mesh, load_ply, prepare_mesh  # noqa
from spectrum_alignment import OptimizationParams, calc_evals, run_optimization  # noqa

import open3d as o3d  # noqa
import trimesh  # noqa


# -----------------------------
# small helpers
# -----------------------------
def _mesh_name(i):
    return "mesh_{:05d}".format(i)


def _find_obj_in_dir(mesh_dir, preferred_name=None):
    mesh_dir = Path(mesh_dir)
    if preferred_name:
        p = mesh_dir / preferred_name
        if p.exists():
            return p
    objs = sorted(mesh_dir.glob("*.obj"))
    if not objs:
        raise FileNotFoundError("No .obj found in {}".format(mesh_dir))
    return objs[0]


def find_recon_ply(out_dir, nevals, prefer_iter=299):
    out_dir = Path(out_dir)
    preferred = out_dir / "evals_{}_iter{}.ply".format(nevals, prefer_iter)
    if preferred.exists():
        return preferred, prefer_iter

    pat = re.compile(r"^evals_{}_iter(\d+)\.ply$".format(nevals))
    best_it = -1
    best_p = None
    for p in out_dir.glob("evals_{}_iter*.ply".format(nevals)):
        m = pat.match(p.name)
        if not m:
            continue
        it = int(m.group(1))
        if it > best_it:
            best_it = it
            best_p = p
    if best_p is None:
        raise FileNotFoundError("No reconstruction ply found in: {}".format(out_dir))
    return best_p, best_it


def load_csv_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(csv_path, fieldnames, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_csv_row(csv_path, row, write_header_if_new=True):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header_if_new and (not file_exists):
            w.writeheader()
        w.writerow(row)


# -----------------------------
# ICP + voxel IoU
# -----------------------------
def mesh_to_pcd(mesh, n_points):
    return mesh.sample_points_uniformly(number_of_points=int(n_points))


def estimate_base_voxel_from_sampling(pcd, max_samples=6000):
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 1000:
        raise ValueError("Too few sampled points to estimate scale. Increase --n_points.")

    if pts.shape[0] > max_samples:
        idx = np.random.choice(pts.shape[0], size=max_samples, replace=False)
        pts_sub = pts[idx]
        pcd_sub = o3d.geometry.PointCloud()
        pcd_sub.points = o3d.utility.Vector3dVector(pts_sub)
    else:
        pts_sub = pts
        pcd_sub = pcd

    kdtree = o3d.geometry.KDTreeFlann(pcd_sub)
    dists = np.empty((pts_sub.shape[0],), dtype=np.float64)
    for i, p in enumerate(pts_sub):
        _, _, dist2 = kdtree.search_knn_vector_3d(p, 2)
        dists[i] = np.sqrt(dist2[1])

    med = float(np.median(dists))

    aabb = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(aabb.get_extent()))
    if diag <= 0:
        raise ValueError("Degenerate point cloud bounding box.")
    base = float(np.clip(3.0 * med, 1e-6 * diag, 0.05 * diag))
    return base


def preprocess(pcd, voxel):
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0 * voxel, max_nn=100)
    )
    return pcd_down, fpfh


def load_as_trimesh(path):
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def voxel_iou_shared_grid(mesh_a, mesh_b, pitch, padding_frac=0.02, max_voxels=250_000_000):
    bounds = np.vstack([mesh_a.bounds, mesh_b.bounds])
    bmin = bounds.min(axis=0)
    bmax = bounds.max(axis=0)

    size = bmax - bmin
    pad = float(padding_frac) * float(size.max())
    bmin = bmin - pad
    bmax = bmax + pad
    size = bmax - bmin

    dims = np.ceil(size / float(pitch)).astype(int)
    max_vox = int(dims[0]) * int(dims[1]) * int(dims[2])
    if max_vox > int(max_voxels):
        raise MemoryError(
            "Voxel grid too large: dims={} (~{:,} voxels). Increase pitch.".format(tuple(dims), max_vox)
        )

    xs = (np.arange(dims[0]) + 0.5) * pitch + bmin[0]
    ys = (np.arange(dims[1]) + 0.5) * pitch + bmin[1]
    zs = (np.arange(dims[2]) + 0.5) * pitch + bmin[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    inside_a = mesh_a.contains(pts)
    inside_b = mesh_b.contains(pts)

    inter = int(np.logical_and(inside_a, inside_b).sum())
    union = int(np.logical_or(inside_a, inside_b).sum())
    iou = float(inter) / float(union) if union > 0 else 0.0
    return iou, (int(dims[0]), int(dims[1]), int(dims[2]))


def compute_icp_transform_open3d(target_mesh_path, source_mesh_path, n_points):
    tgt_mesh = o3d.io.read_triangle_mesh(str(target_mesh_path))
    src_mesh = o3d.io.read_triangle_mesh(str(source_mesh_path))
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
        final_rmse = float(icp.inlier_rmse)

    return T, final_rmse, float(base_voxel)


def eval_iou_pre_post_icp(target_obj, recon_ply, pitch, padding, max_voxels, n_points):
    A = load_as_trimesh(target_obj)
    B = load_as_trimesh(recon_ply)

    iou_naive, dims = voxel_iou_shared_grid(A, B, pitch=pitch, padding_frac=padding, max_voxels=max_voxels)

    T, rmse, base_voxel = compute_icp_transform_open3d(target_obj, recon_ply, n_points=n_points)

    B_aligned = B.copy()
    B_aligned.apply_transform(T)
    iou_aligned, _ = voxel_iou_shared_grid(A, B_aligned, pitch=pitch, padding_frac=padding, max_voxels=max_voxels)

    return float(iou_naive), float(iou_aligned), float(rmse), float(base_voxel), dims


# -----------------------------
# experiment: sphere init for each target (nevals=20)
# -----------------------------
def run_sphere_init_experiment(args, bubble_root, sphere_dir, results_root, out_dir):
    nevals = 20  # forced
    csv_path = out_dir / "sphere_init_ne20.csv"
    eval_cache_dir = out_dir / "evals_cache"
    if args.cache_target_evals:
        eval_cache_dir.mkdir(parents=True, exist_ok=True)

    params = OptimizationParams()
    params.min_eval_loss = 0.0001
    params.evals = [nevals]
    params.numsteps = int(args.numsteps)
    params.checkpoint = int(args.checkpoint)
    params.volume_reg = 1e1

    # sphere initial mesh (dir)
    V0, F0 = load_mesh(str(sphere_dir))
    mesh0 = prepare_mesh(V0, F0, "float32")

    for idx in range(int(args.start), int(args.end) + 1):
        name = _mesh_name(idx)
        target_dir = bubble_root / name
        if not target_dir.exists():
            raise FileNotFoundError("Missing target folder: {}".format(target_dir))

        target_obj = _find_obj_in_dir(target_dir, preferred_name="{}.obj".format(name))

        step_out = results_root / "sphere_init_ne20" / name
        step_out.mkdir(parents=True, exist_ok=True)

        # resume if row already exists AND recon ply exists
        if args.resume and csv_path.exists():
            rows = load_csv_rows(csv_path)
            already = [r for r in rows if (r.get("mesh_idx") == str(idx))]
            if already:
                # ensure recon exists
                maybe_ply = already[-1].get("recon_ply", "")
                if maybe_ply and Path(maybe_ply).exists():
                    print("[{}] sphere-init resume: already in CSV, skipping.".format(name))
                    continue

        # target evals (cached)
        eval_cache_path = eval_cache_dir / "{}_evals.npy".format(name)
        if args.cache_target_evals and eval_cache_path.exists():
            evals_t = np.load(str(eval_cache_path))
            print("[{}] target evals: loaded cache".format(name))
        else:
            print("[{}] target evals: computing...".format(name))
            Vt, Ft = load_mesh(str(target_dir))
            evals_t = calc_evals(Vt, Ft)
            if args.cache_target_evals:
                np.save(str(eval_cache_path), evals_t)

        # run optimization if needed
        recon_ply_expected = step_out / "evals_{}_iter{}.ply".format(nevals, int(args.prefer_iter))
        if args.resume and recon_ply_expected.exists():
            print("[{}] recon exists, skipping optimization.".format(name))
        else:
            print("[{}] sphere-init recon: running optimization...".format(name))
            t0 = time.time()
            run_optimization(mesh=mesh0, target_evals=evals_t, out_path=str(step_out), params=params)
            print("[{}] sphere-init recon: done in {:.1f}s".format(name, time.time() - t0))

        recon_ply, used_iter = find_recon_ply(step_out, nevals, prefer_iter=int(args.prefer_iter))

        # choose pitch based on target size (same approach as cascade)
        tgt_mesh_o3d = o3d.io.read_triangle_mesh(str(target_obj))
        tgt_pcd = mesh_to_pcd(tgt_mesh_o3d, n_points=int(args.n_points))
        base_voxel = estimate_base_voxel_from_sampling(tgt_pcd)
        pitch = float(args.iou_pitch_mult) * float(base_voxel)

        print("[{}] sphere-init ICP/IoU...".format(name))
        iou_naive, iou_aligned, rmse, base_voxel_icp, dims = eval_iou_pre_post_icp(
            target_obj=target_obj,
            recon_ply=recon_ply,
            pitch=pitch,
            padding=float(args.padding),
            max_voxels=int(args.max_voxels),
            n_points=int(args.n_points),
        )
        print("  naive IoU={:.6f} | aligned IoU={:.6f} | rmse={:.9f}".format(iou_naive, iou_aligned, rmse))

        row = {
            "mesh_idx": idx,
            "mesh_name": name,
            "target_obj": str(target_obj),
            "results_step_dir": str(step_out),
            "recon_ply": str(recon_ply),
            "used_iter": used_iter,
            "nevals": nevals,
            "numsteps": int(args.numsteps),
            "checkpoint": int(args.checkpoint),
            "n_points": int(args.n_points),
            "iou_pitch_mult": float(args.iou_pitch_mult),
            "padding": float(args.padding),
            "max_voxels": int(args.max_voxels),
            "pitch": float(pitch),
            "iou_naive": float(iou_naive),
            "iou_aligned": float(iou_aligned),
            "rmse": float(rmse),
            "grid_dim_x": int(dims[0]),
            "grid_dim_y": int(dims[1]),
            "grid_dim_z": int(dims[2]),
        }
        append_csv_row(csv_path, row)

    return csv_path


# -----------------------------
# plot: cascade(ne20) vs sphere-init(ne20)
# -----------------------------
def plot_overlay(cascade_csv_path, sphereinit_csv_path, out_dir):
    import matplotlib.pyplot as plt

    cascade_rows = load_csv_rows(cascade_csv_path)
    sph_rows = load_csv_rows(sphereinit_csv_path)

    def extract_xy(rows):
        xs = []
        pre = []
        post = []
        for r in rows:
            try:
                ne = int(float(r.get("nevals", "0")))
            except Exception:
                ne = 0
            if ne != 20:
                continue
            try:
                xs.append(int(r["mesh_idx"]))
                pre.append(float(r["iou_naive"]))
                post.append(float(r["iou_aligned"]))
            except Exception:
                pass
        # sort
        order = np.argsort(np.array(xs))
        xs = np.array(xs)[order]
        pre = np.array(pre)[order]
        post = np.array(post)[order]
        return xs, pre, post

    cx, cpre, cpost = extract_xy(cascade_rows)
    sx, spre, spost = extract_xy(sph_rows)

    if len(cx) == 0:
        raise RuntimeError("No nevals=20 rows found in cascade CSV: {}".format(cascade_csv_path))
    if len(sx) == 0:
        raise RuntimeError("No nevals=20 rows found in sphere-init CSV: {}".format(sphereinit_csv_path))

    plt.figure(figsize=(10.5, 5.8))

    # Cascade: same color pre/post, circular markers
    l1, = plt.plot(cx, cpre, "--o", linewidth=2, markersize=4, label="Cascade (ne=20) IoU pre-ICP")
    plt.plot(cx, cpost, "-o", linewidth=2.5, markersize=4, color=l1.get_color(), label="Cascade (ne=20) IoU post-ICP")

    # Sphere-init: different color, circular markers
    l2, = plt.plot(sx, spre, "--o", linewidth=2, markersize=4, label="Sphere-init (ne=20) IoU pre-ICP")
    plt.plot(sx, spost, "-o", linewidth=2.5, markersize=4, color=l2.get_color(), label="Sphere-init (ne=20) IoU post-ICP")

    xmin = int(min(cx.min(), sx.min()))
    xmax = int(max(cx.max(), sx.max()))
    plt.xlim(xmin, xmax)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Mesh index")
    plt.ylabel("Voxel IoU")
    plt.title("Cascade vs Sphere-init reconstruction (ne=20)")
    plt.legend()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "cascade_iou_plot_combined.png"
    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=220)
    plt.close()
    print("Saved overlay plot:", plot_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=6)
    ap.add_argument("--end", type=int, default=40)

    ap.add_argument("--sphere_dir", type=str, default="data/Drake_Sphere")
    ap.add_argument("--bubble_root", type=str, default="data/Bubble_Grasp_Deep_0.05x40")

    ap.add_argument("--results_root", type=str, default="results/Cascade_Recon")
    ap.add_argument("--out_dir", type=str, default="out/cascade")

    ap.add_argument("--numsteps", type=int, default=300)
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--prefer_iter", type=int, default=299)

    ap.add_argument("--n_points", type=int, default=100000)
    ap.add_argument("--iou_pitch_mult", type=float, default=2.0)
    ap.add_argument("--padding", type=float, default=0.02)
    ap.add_argument("--max_voxels", type=int, default=250_000_000)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cache_target_evals", action="store_true")

    ap.add_argument(
        "--cascade_csv",
        type=str,
        default="out/cascade/cascade_iou.csv",
        help="Path to your existing cascade CSV (will be read only).",
    )

    args = ap.parse_args()

    bubble_root = Path(args.bubble_root).resolve()
    sphere_dir = Path(args.sphere_dir).resolve()
    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    cascade_csv = Path(args.cascade_csv).resolve()
    if not cascade_csv.exists():
        raise FileNotFoundError("Cascade CSV not found: {}".format(cascade_csv))

    print("Working directory:", Path.cwd())
    print("Cascade CSV:", cascade_csv)
    print("Sphere init experiment CSV will be written to out_dir.")
    print()

    sphere_csv = run_sphere_init_experiment(args, bubble_root, sphere_dir, results_root, out_dir)
    plot_overlay(cascade_csv, sphere_csv, out_dir)


if __name__ == "__main__":
    main()

'''
RUN AS (compares 20eval cascade to always rcons from sphere):
.\external_isospectralization\tf-cpu\Scripts\python.exe `
  .\external_isospectralization\code_for_3D\scripts\sphere_init_vs_cascade_plot.py `
  --start 6 --end 40 `
  --sphere_dir "data/Drake_Sphere" `
  --bubble_root "data/Bubble_Grasp_Deep_0.05x40" `
  --results_root "results/Cascade_Recon" `
  --out_dir "out/cascade" `
  --cascade_csv "out/cascade/cascade_iou.csv" `
  --numsteps 300 --checkpoint 10 --prefer_iter 299 `
  --n_points 100000 --iou_pitch_mult 2.0 --padding 0.02 `
  --resume --cache_target_evals

'''