# external_isospectralization/code_for_3D/scripts/cascade_recon_iou.py

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make imports work no matter where you launch from:
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent  # .../code_for_3D
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Keep existing scripts happy (they assume relative paths from code_for_3D)
os.chdir(str(CODE_ROOT))

from shape_library import load_mesh, load_ply, prepare_mesh  # noqa: E402
from spectrum_alignment import OptimizationParams, calc_evals, run_optimization  # noqa: E402

import open3d as o3d  # noqa: E402
import trimesh  # noqa: E402


# -----------------------------
# Helpers: files / selection
# -----------------------------

def _mesh_name(i: int) -> str:
    return "mesh_{:05d}".format(i)


def find_recon_ply(out_dir: Path, nevals: int, prefer_iter: int = 299) -> Tuple[Path, int]:
    """Prefer evals_{nevals}_iter{prefer_iter}.ply; else take the max iter available."""
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


def load_existing_csv(csv_path: Path) -> Dict[int, Dict[str, str]]:
    """Return existing rows keyed by mesh index (if csv exists)."""
    if not csv_path.exists():
        return {}
    rows = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                idx = int(r["mesh_idx"])
            except Exception:
                continue
            rows[idx] = r
    return rows


def append_csv_row(csv_path: Path, row: Dict[str, object], write_header_if_new: bool = True) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header_if_new and (not file_exists):
            writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _find_obj_in_dir(mesh_dir: Path, preferred_name: str = None) -> Path:
    if preferred_name is not None:
        p = mesh_dir / preferred_name
        if p.exists():
            return p
    objs = list(mesh_dir.glob("*.obj"))
    if not objs:
        raise FileNotFoundError("No .obj found in {}".format(mesh_dir))
    return objs[0]


# -----------------------------
# ICP + IoU (voxel grid)
# -----------------------------

def mesh_to_pcd(mesh: o3d.geometry.TriangleMesh, n_points: int) -> o3d.geometry.PointCloud:
    return mesh.sample_points_uniformly(number_of_points=int(n_points))


def estimate_base_voxel_from_sampling(pcd: o3d.geometry.PointCloud, max_samples: int = 6000) -> float:
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 1000:
        raise ValueError("Too few sampled points to estimate scale. Increase --n_points.")

    # Subsample for speed
    if pts.shape[0] > max_samples:
        idx = np.random.choice(pts.shape[0], size=max_samples, replace=False)
        pts_sub = pts[idx]
        pcd_sub = o3d.geometry.PointCloud()
        pcd_sub.points = o3d.utility.Vector3dVector(pts_sub)
    else:
        pts_sub = pts
        pcd_sub = pcd

    kdtree = o3d.geometry.KDTreeFlann(pcd_sub)
    dists = np.empty(pts_sub.shape[0], dtype=np.float64)

    for i, p in enumerate(pts_sub):
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
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=5.0 * voxel, max_nn=100)
    )
    return pcd_down, fpfh


def load_as_trimesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def voxel_iou_shared_grid(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    pitch: float,
    padding_frac: float = 0.02,
    max_voxels: int = 250_000_000,
) -> Tuple[float, Tuple[int, int, int]]:
    bounds = np.vstack([mesh_a.bounds, mesh_b.bounds])
    bmin = bounds.min(axis=0)
    bmax = bounds.max(axis=0)

    size = bmax - bmin
    pad = padding_frac * float(size.max())
    bmin = bmin - pad
    bmax = bmax + pad
    size = bmax - bmin

    dims = np.ceil(size / float(pitch)).astype(int)
    max_vox = int(dims[0]) * int(dims[1]) * int(dims[2])
    if max_vox > max_voxels:
        raise MemoryError(
            "Voxel grid too large: dims={} (~{:,} voxels). Increase pitch via --iou_pitch_mult (or raise --max_voxels)."
            .format(tuple(dims), max_vox)
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
    iou = float(inter / union) if union > 0 else 0.0
    return iou, (int(dims[0]), int(dims[1]), int(dims[2]))


def compute_icp_transform_open3d(target_mesh_path: Path, source_mesh_path: Path, n_points: int):
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


def eval_iou_pre_post_icp(
    target_path_for_icp: Path,
    source_mesh_path: Path,
    pitch: float,
    padding: float,
    max_voxels: int,
    n_points: int,
) -> Tuple[float, float, float, float, Tuple[int, int, int]]:
    """
    Returns: (iou_naive, iou_aligned, rmse, base_voxel, dims)
    """
    A = load_as_trimesh(target_path_for_icp)
    B = load_as_trimesh(source_mesh_path)

    iou_naive, dims = voxel_iou_shared_grid(A, B, pitch=pitch, padding_frac=padding, max_voxels=max_voxels)

    T, rmse, base_voxel = compute_icp_transform_open3d(target_path_for_icp, source_mesh_path, n_points=n_points)

    B_aligned = B.copy()
    B_aligned.apply_transform(T)
    iou_aligned, _ = voxel_iou_shared_grid(A, B_aligned, pitch=pitch, padding_frac=padding, max_voxels=max_voxels)

    return float(iou_naive), float(iou_aligned), float(rmse), float(base_voxel), dims


# -----------------------------
# Baseline overlay (GT sphere vs GT targets)
# -----------------------------

def compute_or_load_gt_baseline(
    baseline_csv: Path,
    sphere_obj: Path,
    bubble_root: Path,
    start: int,
    end: int,
    n_points: int,
    iou_pitch_mult: float,
    padding: float,
    max_voxels: int,
    resume: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (or loads) baseline IoU curves between GT sphere and GT target meshes:
      - naive IoU (no ICP)
      - aligned IoU (after ICP)
    Uses a single pitch derived from the SPHERE (stable target) so the baseline is consistent across idx.
    Saves to baseline_csv so you don't recompute.

    Returns: xs, pre, post
    """
    baseline_csv.parent.mkdir(parents=True, exist_ok=True)

    # If baseline exists and resume requested, load it
    if resume and baseline_csv.exists():
        xs = []
        pre = []
        post = []
        with baseline_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                xs.append(int(r["mesh_idx"]))
                pre.append(float(r["iou_naive"]))
                post.append(float(r["iou_aligned"]))
        return np.array(xs), np.array(pre), np.array(post)

    # Derive pitch from sphere (stable) target once
    sphere_o3d = o3d.io.read_triangle_mesh(str(sphere_obj))
    sphere_pcd = mesh_to_pcd(sphere_o3d, n_points=int(n_points))
    base_voxel = estimate_base_voxel_from_sampling(sphere_pcd)
    pitch = float(iou_pitch_mult * base_voxel)

    # Write header fresh
    with baseline_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mesh_idx", "mesh_name", "sphere_obj", "target_obj",
                "n_points", "iou_pitch_mult", "padding", "max_voxels", "pitch",
                "iou_naive", "iou_aligned", "rmse",
                "grid_dim_x", "grid_dim_y", "grid_dim_z",
            ],
        )
        writer.writeheader()

    xs = []
    pre = []
    post = []

    print("[baseline] computing GT sphere vs GT targets ...")
    print("  sphere_obj:", sphere_obj)
    print("  pitch: {:.6g} (base_voxel~{:.6g})".format(pitch, base_voxel))
    print()

    for idx in range(start, end + 1):
        name = _mesh_name(idx)
        target_dir = bubble_root / name
        if not target_dir.exists():
            raise FileNotFoundError("Missing target folder: {}".format(target_dir))
        target_obj = _find_obj_in_dir(target_dir, preferred_name="{}.obj".format(name))

        print("[baseline:{}] running ICP/IoU ...".format(name))
        iou_naive, iou_aligned, rmse, base_voxel_icp, dims = eval_iou_pre_post_icp(
            target_path_for_icp=sphere_obj,
            source_mesh_path=target_obj,
            pitch=pitch,
            padding=float(padding),
            max_voxels=int(max_voxels),
            n_points=int(n_points),
        )
        print("  naive IoU={:.6f} | aligned IoU={:.6f} | rmse={:.9f}".format(iou_naive, iou_aligned, rmse))

        row = {
            "mesh_idx": idx,
            "mesh_name": name,
            "sphere_obj": str(sphere_obj),
            "target_obj": str(target_obj),
            "n_points": int(n_points),
            "iou_pitch_mult": float(iou_pitch_mult),
            "padding": float(padding),
            "max_voxels": int(max_voxels),
            "pitch": float(pitch),
            "iou_naive": float(iou_naive),
            "iou_aligned": float(iou_aligned),
            "rmse": float(rmse),
            "grid_dim_x": int(dims[0]),
            "grid_dim_y": int(dims[1]),
            "grid_dim_z": int(dims[2]),
        }
        append_csv_row(baseline_csv, row, write_header_if_new=False)

        xs.append(idx)
        pre.append(iou_naive)
        post.append(iou_aligned)

    print("[baseline] saved:", baseline_csv)
    print()
    return np.array(xs), np.array(pre), np.array(post)


# -----------------------------
# Main cascade
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=6)
    ap.add_argument("--end", type=int, default=40)

    ap.add_argument("--sphere_dir", type=str, default="data/Drake_Sphere",
                    help="Initial shape directory (must contain mesh.vert and mesh.triv). Also used for GT baseline overlay.")
    ap.add_argument("--bubble_root", type=str, default="data/Bubble_Grasp_Deep_0.05x40",
                    help="Root containing mesh_00006 ... mesh_00040 folders")

    # CHANGED: list of eigenvalue counts
    ap.add_argument("--nevals_list", type=str, default="20",
                    help="Comma-separated list of eigenvalue counts (e.g. 5,20,50).")

    # CHANGED: baseline overlay flag
    ap.add_argument("--overlay_gt_baseline", action="store_true",
                    help="Also compute/plot IoU between GT sphere and each GT target mesh (pre/post ICP).")

    ap.add_argument("--numsteps", type=int, default=300)
    ap.add_argument("--checkpoint", type=int, default=10)
    ap.add_argument("--prefer_iter", type=int, default=299)

    ap.add_argument("--results_root", type=str, default="results/Cascade_Recon",
                    help="Where per-step isospectralization outputs go (PLY checkpoints)")

    ap.add_argument("--out_dir", type=str, default="out/cascade",
                    help="Where CSV(s) + plot are written")

    # IoU/ICP knobs
    ap.add_argument("--n_points", type=int, default=100000)
    ap.add_argument("--iou_pitch_mult", type=float, default=2.0)
    ap.add_argument("--padding", type=float, default=0.02)
    ap.add_argument("--max_voxels", type=int, default=250_000_000)

    ap.add_argument("--resume", action="store_true",
                    help="Skip steps already present in the CSV(s) + existing recon ply")

    ap.add_argument("--cache_target_evals", action="store_true",
                    help="Cache target evals per mesh in out_dir/evals_cache to speed reruns")

    args = ap.parse_args()

    start, end = int(args.start), int(args.end)
    if end < start:
        raise ValueError("--end must be >= --start")

    nevals_list = [int(x.strip()) for x in args.nevals_list.split(",") if x.strip()]
    if not nevals_list:
        raise ValueError("Empty --nevals_list")

    bubble_root = Path(args.bubble_root).resolve()
    sphere_dir = Path(args.sphere_dir).resolve()
    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional eval cache (shared across nevals, since target evals depend on target mesh only)
    eval_cache_dir = out_dir / "evals_cache"
    if args.cache_target_evals:
        eval_cache_dir.mkdir(parents=True, exist_ok=True)

    print("Working directory set to:", Path.cwd())
    print("Bubble root: ", bubble_root)
    print("Sphere dir:  ", sphere_dir)
    print("Results root:", results_root)
    print("Out dir:     ", out_dir)
    print("nevals_list: ", nevals_list)
    print()

    # Find GT sphere obj for baseline overlay (and useful for sanity checks)
    sphere_obj = _find_obj_in_dir(sphere_dir, preferred_name="mesh_00000.obj")

    # CHANGED: baseline overlay computed once (and cached to CSV)
    baseline_xs = None
    baseline_pre = None
    baseline_post = None
    if args.overlay_gt_baseline:
        baseline_csv = out_dir / "baseline_gt_sphere_vs_targets.csv"
        baseline_xs, baseline_pre, baseline_post = compute_or_load_gt_baseline(
            baseline_csv=baseline_csv,
            sphere_obj=sphere_obj,
            bubble_root=bubble_root,
            start=start,
            end=end,
            n_points=int(args.n_points),
            iou_pitch_mult=float(args.iou_pitch_mult),
            padding=float(args.padding),
            max_voxels=int(args.max_voxels),
            resume=bool(args.resume),
        )

    # CHANGED: run cascades for each nevals, store curves for combined plotting
    curves = {}  # nevals -> dict with xs/pre/post and csv path

    for nevals in nevals_list:
        print("=" * 80)
        print("[cascade] Starting nevals={}".format(nevals))
        print("=" * 80)

        # CHANGED: separate results tree per nevals to avoid collisions
        results_root_nevals = results_root / "nevals_{}".format(nevals)
        csv_path = out_dir / "cascade_iou_nevals{}.csv".format(nevals)

        existing = load_existing_csv(csv_path) if args.resume else {}

        # Params (match your test.py defaults, but set evals to THIS nevals)
        params = OptimizationParams()
        params.min_eval_loss = 0.0001
        params.evals = [int(nevals)]  # CHANGED
        params.numsteps = int(args.numsteps)
        params.checkpoint = int(args.checkpoint)
        params.volume_reg = 0

        # Initial shape for this nevals run = Drake sphere dir
        current_initial_kind = "dir"
        current_initial = sphere_dir

        xs: List[int] = []
        iou_pre_list: List[float] = []
        iou_post_list: List[float] = []
        rmse_list: List[float] = []
        recon_plys: List[str] = []

        for idx in range(start, end + 1):
            name = _mesh_name(idx)

            if args.resume and idx in existing:
                print("[{} | nevals={}] resume: already in CSV, skipping.".format(name, nevals))
                try:
                    xs.append(idx)
                    iou_pre_list.append(float(existing[idx]["iou_naive"]))
                    iou_post_list.append(float(existing[idx]["iou_aligned"]))
                    rmse_list.append(float(existing[idx]["rmse"]))
                    recon_plys.append(existing[idx].get("recon_ply", ""))
                except Exception:
                    pass
                if existing[idx].get("recon_ply"):
                    current_initial_kind = "ply"
                    current_initial = Path(existing[idx]["recon_ply"]).resolve()
                continue

            target_dir = bubble_root / name
            if not target_dir.exists():
                raise FileNotFoundError("Missing target folder: {}".format(target_dir))

            target_obj = _find_obj_in_dir(target_dir, preferred_name="{}.obj".format(name))

            # CHANGED: per-nevals output dir
            step_out = results_root_nevals / name
            step_out.mkdir(parents=True, exist_ok=True)

            print("[{} | nevals={}] === Step {}/{} ===".format(name, nevals, idx, end))
            print("  target:", target_dir)
            print("  out:   ", step_out)

            # ---------
            # Load / compute target evals (optionally cached)
            # ---------
            eval_cache_path = eval_cache_dir / "{}_evals.npy".format(name)
            if args.cache_target_evals and eval_cache_path.exists():
                evals_t = np.load(str(eval_cache_path))
                print("  target evals: loaded from cache")
            else:
                print("  target evals: computing...")
                VERT_t, TRIV_t = load_mesh(str(target_dir))
                evals_t = calc_evals(VERT_t, TRIV_t)
                if args.cache_target_evals:
                    np.save(str(eval_cache_path), evals_t)
                    print("  target evals: cached")

            # ---------
            # Build initial mesh for this step
            # ---------
            if current_initial_kind == "dir":
                V0, F0 = load_mesh(str(current_initial))
                init_desc = str(current_initial)
            else:
                V0, F0 = load_ply(str(current_initial))
                init_desc = str(current_initial)

            mesh0 = prepare_mesh(V0, F0, "float32")

            # ---------
            # Run isospectralization unless recon ply already exists
            # ---------
            recon_ply_expected = step_out / "evals_{}_iter{}.ply".format(nevals, int(args.prefer_iter))
            if args.resume and recon_ply_expected.exists():
                print("  recon: found {}, skipping optimization.".format(recon_ply_expected.name))
            else:
                print("  recon: running optimization (init={}) ...".format(init_desc))
                t0 = time.time()
                run_optimization(mesh=mesh0, target_evals=evals_t, out_path=str(step_out), params=params)
                print("  recon: done in {:.1f}s".format(time.time() - t0))

            recon_ply, used_iter = find_recon_ply(step_out, int(nevals), prefer_iter=int(args.prefer_iter))
            print("  recon ply:", recon_ply.name, "(iter={})".format(used_iter))

            # ---------
            # ICP + IoU (recon vs GT target)
            # ---------
            print("  eval: running ICP/IoU ...")

            # Pitch derived per-target (same as before)
            tgt_mesh_o3d = o3d.io.read_triangle_mesh(str(target_obj))
            tgt_pcd = mesh_to_pcd(tgt_mesh_o3d, n_points=int(args.n_points))
            base_voxel = estimate_base_voxel_from_sampling(tgt_pcd)
            pitch = float(args.iou_pitch_mult * base_voxel)

            try:
                iou_naive, iou_aligned, rmse, base_voxel_icp, dims = eval_iou_pre_post_icp(
                    target_path_for_icp=target_obj,
                    source_mesh_path=recon_ply,
                    pitch=pitch,
                    padding=float(args.padding),
                    max_voxels=int(args.max_voxels),
                    n_points=int(args.n_points),
                )
            except Exception as e:
                raise RuntimeError(
                    "ICP/IoU failed.\n"
                    "Common fix if you see trimesh.contains errors: ensure `Rtree` is installed in the venv.\n"
                    "Original error: {}: {}".format(type(e).__name__, e)
                )

            print("  iou naive   = {:.6f}".format(iou_naive))
            print("  iou aligned = {:.6f}".format(iou_aligned))
            print("  rmse        = {:.9f}".format(rmse))
            print("  pitch       = {:.6g} (base_voxel~{:.6g})".format(pitch, base_voxel_icp))
            print("  grid dims   =", dims)
            print()

            row = {
                "mesh_idx": idx,
                "mesh_name": name,
                "nevals": int(nevals),
                "target_dir": str(target_dir),
                "target_obj": str(target_obj),
                "init_kind": current_initial_kind,
                "init_path": init_desc,
                "results_step_dir": str(step_out),
                "recon_ply": str(recon_ply),
                "used_iter": used_iter,
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

            xs.append(idx)
            iou_pre_list.append(iou_naive)
            iou_post_list.append(iou_aligned)
            rmse_list.append(rmse)
            recon_plys.append(str(recon_ply))

            # Cascade: next step's initial is this step's reconstruction ply
            current_initial_kind = "ply"
            current_initial = recon_ply

        if len(xs) == 0:
            print("[cascade] nevals={} produced no points (everything skipped?)".format(nevals))
            curves[nevals] = {"xs": np.array([]), "pre": np.array([]), "post": np.array([]), "csv": csv_path}
            continue

        order = np.argsort(np.array(xs))
        xs_arr = np.array(xs)[order]
        pre_arr = np.array(iou_pre_list)[order]
        post_arr = np.array(iou_post_list)[order]

        curves[nevals] = {"xs": xs_arr, "pre": pre_arr, "post": post_arr, "csv": csv_path}
        print("[cascade] saved CSV:", csv_path)
        print()

    # -----------------------------
    # Combined plot (CHANGED)
    #   - markers 'o'
    #   - 3 colors for nevals
    #   - overlay baseline GT if requested
    # -----------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10.5, 5.8))

    # Plot each nevals curve
    for nevals in nevals_list:
        d = curves.get(nevals, None)
        if d is None or d["xs"].size == 0:
            continue
        lpre, = plt.plot(
            d["xs"], d["pre"],
            linestyle="--", linewidth=2.0,
            marker="o", markersize=4.5,
            label="{} evals (pre-ICP)".format(nevals),
        )
        plt.plot(
            d["xs"], d["post"],
            linestyle="-", linewidth=2.6,
            marker="o", markersize=4.5,
            color=lpre.get_color(),
            label="{} evals (post-ICP)".format(nevals),
        )

    # Optional baseline overlay
    if args.overlay_gt_baseline and baseline_xs is not None:
        # Ensure ordered
        b_order = np.argsort(baseline_xs)
        bx = baseline_xs[b_order]
        bpre = baseline_pre[b_order]
        bpost = baseline_post[b_order]

        lbase, = plt.plot(
            bx, bpre,
            linestyle="--", linewidth=2.0,
            marker="o", markersize=4.5,
            color="0.35",
            label="GT sphere vs GT target (pre-ICP)",
        )
        plt.plot(
            bx, bpost,
            linestyle="-", linewidth=2.6,
            marker="o", markersize=4.5,
            color="0.35",
            label="GT sphere vs GT target (post-ICP)",
        )

    plt.ylim(0.0, 1.0)
    plt.xlim(start, end)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Mesh index")
    plt.ylabel("Voxel IoU")
    plt.title("Cascading reconstruction quality (naive vs ICP-aligned)")
    plt.legend(ncol=2, fontsize=9)

    plot_path = out_dir / "cascade_iou_plot_combined.png"
    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=200)
    plt.close()

    print("Saved combined plot:", plot_path)
    print("Per-nevals CSVs:")
    for nevals in nevals_list:
        if nevals in curves:
            print("  nevals={}: {}".format(nevals, curves[nevals]["csv"]))
    if args.overlay_gt_baseline:
        print("Baseline CSV:", out_dir / "baseline_gt_sphere_vs_targets.csv")


if __name__ == "__main__":
    main()


'''
RUN AS:
.\external_isospectralization\tf-cpu\Scripts\python.exe `
  .\external_isospectralization\code_for_3D\scripts\cascade_recon_iou.py `
  --start 6 --end 28 `
  --nevals_list "5,20,50,75" `
  --numsteps 300 --checkpoint 10 --prefer_iter 299 `
  --sphere_dir "data/Drake_Sphere" `
  --bubble_root "data/Bubble_Grasp_Deep_0.05x40" `
  --results_root "results/Cascade_Recon" `
  --out_dir "out/cascade" `
  --n_points 100000 --iou_pitch_mult 2.0 --padding 0.02 `
  --overlay_gt_baseline `
  --resume --cache_target_evals
'''