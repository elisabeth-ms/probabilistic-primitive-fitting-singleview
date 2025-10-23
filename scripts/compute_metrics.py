from __future__ import annotations

from pathlib import Path

from sklearn.cluster import DBSCAN
import open3d as o3d
from collections import defaultdict
import re, datetime as dt
import re, yaml, numpy as np
import torch
from mayavi import mlab
import plot_functions
import tools

import torch
import math

import numpy as np;
from typing import Tuple, Dict




def to_world(points_local, R, t):
    # world = local @ R + t
    return points_local @ R + t

# --- small helpers ---
def _signed_pow(x, p, eps=1e-12):
    # sign(x) * |x|^p, safe at 0
    return torch.sign(x) * torch.pow(torch.clamp(torch.abs(x), min=eps), p)

def _cos(x): 
  return torch.cos(x)

def _sin(x): 
  return torch.sin(x)

def superellipsoid_param_to_local(eta, omega, a, eps1, eps2):
    # Using standard param eq with signed powers of cos/sin
    c_eta = _cos(eta); s_eta = _sin(eta)
    c_om  = _cos(omega); s_om  = _sin(omega)

    x = a[0] * _signed_pow(c_eta, eps1/2.0) * _signed_pow(c_om, eps2/2.0)
    y = a[1] * _signed_pow(c_eta, eps1/2.0) * _signed_pow(s_om, eps2/2.0)
    z = a[2] * _signed_pow(s_eta, eps1/2.0)
    return torch.stack([x,y,z], dim=-1)  # (...,3)
  

# @torch.no_grad()
# def nearest_on_superquadric(
#     points_world: torch.Tensor,   # (N,3) float32, same device for best speed
#     samples_world: torch.Tensor,  # (K,3) float32, sampled points from THIS superquadric
# ):
#     """
#     Returns:
#       dmin : (N,)   nearest distance from each point to this superquadric's samples
#       idx  : (N,)   index into samples_world of the nearest sample for each point
#       qmin : (N,3)  the nearest sampled point coords (world) for each point
#     """
#     # Pairwise distances N x K
#     D = torch.cdist(points_world, samples_world, p=2)   # (N, K)

#     # Nearest per point
#     dmin, idx = D.min(dim=1)                            # (N,), (N,)
#     qmin = samples_world[idx]                           # (N,3)
#     return dmin, idx, qmin

  

  


import numpy as np

# ---------- SE(3) helpers ----------
def quat_to_R(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    q /= n
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)

def pose7_to_T(pose7):
    """pose7: [tx,ty,tz,qx,qy,qz,qw]  ->  4x4 T (world<-frame)"""
    tx, ty, tz, qx, qy, qz, qw = [float(v) for v in pose7]
    R = quat_to_R(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def T_inv(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def apply_T_points_normals(points, normals, T):
    R, t = T[:3,:3], T[:3,3]
    Pw = points @ R.T + t
    Nw = None if normals is None else (normals @ R.T)
    if Nw is not None:
        Nw /= (np.linalg.norm(Nw, axis=1, keepdims=True) + 1e-12)
    return Pw, Nw

# ---------- world -> camera conversion ----------
def object_pose_in_camera(T_w_o, pose7_cam_world):
    """
    T_w_o: 4x4 pose of object in world (world<-object)
    pose7_cam_world: [tx,ty,tz,qx,qy,qz,qw] of CAMERA in world (world<-camera)
    Returns: T_c_o (camera<-object)
    """
    T_w_c = pose7_to_T(pose7_cam_world)   # world <- camera
    T_c_w = T_inv(T_w_c)                  # camera <- world
    T_c_o = T_c_w @ T_w_o                 # camera <- object
    return T_c_o

# ---------- main convenience ----------
def transform_obj_cloud_into_camera(points_obj, normals_obj, T_c_o):
    """
    points_obj/normals_obj: OBJ cloud in object canonical frame (Nx3)
    pose7_obj_world: object pose in world [tx,ty,tz,qx,qy,qz,qw]
    pose7_cam_world: camera pose in world [tx,ty,tz,qx,qy,qz,qw]  (e.g., head_camera_rgb_optical_frame)
    Returns: points_cam, normals_cam  (cloud in camera frame)
    """
    pts_c, nrm_c = apply_T_points_normals(points_obj, normals_obj, T_c_o)
    return pts_c, nrm_c

# --- utilities ---
def np_to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pc

def auto_voxel(pts: np.ndarray) -> float:
    """
    Pick a voxel ~ median NN distance (clamped).
    Works better than fixed voxel when density varies.
    """
    if pts.shape[0] < 200:
        # sparse: use bbox-based heuristic
        diag = np.linalg.norm(pts.max(0) - pts.min(0))
        return max(0.0002, min(0.05, diag / 150.0))
    # fast kNN on a tiny subsample
    pcd = np_to_pcd(pts[np.random.choice(pts.shape[0], min(4000, pts.shape[0]), replace=False)])
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in range(min(1000, len(pcd.points))):
        _, idx, _ = kdt.search_knn_vector_3d(pcd.points[i], 2)
        if len(idx) == 2:
            d = np.linalg.norm(np.asarray(pcd.points[i]) - np.asarray(pcd.points[idx[1]]))
            dists.append(d)
    med = np.median(dists) if dists else 0.01
    return float(max(0.0002, min(0.05, med)))  # clamp to 2–50 mm



# ---------- helpers ----------
def np_to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pc

def auto_voxel(pts: np.ndarray) -> float:
    """Pick a voxel ~ median NN distance (clamped). Works when density varies."""
    if pts.shape[0] < 200:
        diag = np.linalg.norm(pts.max(0) - pts.min(0))
        return max(0.002, min(0.05, diag / 150.0))
    pcd = np_to_pcd(pts[np.random.choice(pts.shape[0], min(4000, pts.shape[0]), replace=False)])
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    upper = min(1000, len(pcd.points))
    for i in range(upper):
        _, idx, _ = kdt.search_knn_vector_3d(pcd.points[i], 2)
        if len(idx) == 2:
            d = np.linalg.norm(np.asarray(pcd.points[i]) - np.asarray(pcd.points[idx[1]]))
            dists.append(d)
    med = np.median(dists) if dists else 0.01
    return float(max(0.002, min(0.05, med)))  # clamp to 2–50 mm

def preprocess(pc: o3d.geometry.PointCloud, voxel: float, radius_factor=4.0, max_nn=60):
    p = pc.voxel_down_sample(voxel)
    if len(p.points) < 30:
        return pc  # too sparse → skip downsampling and normals here
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*radius_factor, max_nn=max_nn))
    try:
        p.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass
    return p

def ensure_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int = 120) -> bool:
    if len(pcd.points) < 30:
        return False
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass
    return pcd.has_normals()

# ---------- main ----------
def icp_point_to_plane_adaptive(src_pts: np.ndarray,
                                tgt_pts: np.ndarray,
                                T_init=np.eye(4)):
    """
    Adaptive multi-scale ICP with finer voxels and a final full-res pass.
    Returns: (T_refined [4x4], info [6x6])
    """
    assert src_pts.ndim == 2 and src_pts.shape[1] == 3
    assert tgt_pts.ndim == 2 and tgt_pts.shape[1] == 3
    if len(src_pts) < 10 or len(tgt_pts) < 10:
        raise ValueError("Not enough points for ICP.")

    base = 0.5 * (auto_voxel(src_pts) + auto_voxel(tgt_pts))

    # --- finer pyramid (more resolution) ---
    voxels = [base*1.25, base*0.7, max(base*0.4, 0.0015)]  # last ≈ very fine
    voxels = [float(max(0.001, min(0.03, v))) for v in voxels]  # clamp: 1–30 mm

    # per-level normal neighborhoods (bigger at fine scale)
    normal_radius_factor = (3.5, 5.0, 7.0)
    max_nn_levels = (40, 70, 90)

    # tighter correspondence per level
    mcd_mul = (2.0, 1.5, 1.2)
    iters = (80, 60, 50)

    src = np_to_pcd(src_pts)
    tgt = np_to_pcd(tgt_pts)

    T = T_init.copy()

    for lvl, (vox, rfac, nn, mul, nit) in enumerate(
        zip(voxels, normal_radius_factor, max_nn_levels, mcd_mul, iters), 1
    ):
        src_ds = preprocess(src, vox, rfac, nn)
        tgt_ds = preprocess(tgt, vox, rfac, nn)

        # Prefer point-to-plane; fall back if normals missing
        use_p2plane = src_ds.has_normals() and tgt_ds.has_normals() and \
                      len(src_ds.points) >= 30 and len(tgt_ds.points) >= 30
        if use_p2plane:
            try:
                rk = o3d.pipelines.registration.RobustKernel(
                    o3d.pipelines.registration.RobustKernelType.Tukey, 4.685
                )
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane(rk)
            except Exception:
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            mode = "p2plane"
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            mode = "p2point"

        max_corr = mul * vox
        reg = o3d.pipelines.registration.registration_icp(
            src_ds, tgt_ds, max_corr, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=nit),
        )
        T = reg.transformation
        print(f"[L{lvl}] voxel={vox:.4f} thr={max_corr:.4f}  fitness={reg.fitness:.3f} "
              f"rmse={reg.inlier_rmse:.5f} mode={mode}")

    # --- ultra-fine final pass on full-resolution clouds ---
    final_thresh = max(0.0015, 0.8 * voxels[-1])  # ~1–2 mm typical
    # ensure normals on target for point-to-plane; if not possible, fall back
    norm_radius = max(final_thresh * 8.0, 0.006)  # generous neighborhood
    tgt_has = ensure_normals(tgt, norm_radius, max_nn=150)
    src_has = ensure_normals(src, norm_radius, max_nn=150)

    if tgt_has:
        est_final = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        final_mode = "p2plane"
    else:
        est_final = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        final_mode = "p2point"

    reg_final = o3d.pipelines.registration.registration_icp(
        src, tgt, final_thresh, T, est_final,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    T = reg_final.transformation
    print(f"[Final] full-res thr={final_thresh:.4f} rmse={reg_final.inlier_rmse:.5f} mode={final_mode}")

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, tgt, max_correspondence_distance=final_thresh, transformation=T
    )
    return T, info
  
def quat_xyzw_to_R(qx, qy, qz, qw):
    # unit quaternion -> 3x3 rotation (x,y,z,w convention)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1-2*(yy+zz),   2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1-2*(xx+zz),     2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),     1-2*(xx+yy)]
    ], dtype=float)

def R_to_quat_xyzw(R):
    """
    Rotation matrix (3x3) -> quaternion in (x, y, z, w) order.
    Robust, handles small numerical issues, and normalizes.
    """
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)  # normalize
    return q

def pose7_from_T(T, prefer_quat_sign=None):
    """
    4x4 SE(3) -> [x, y, z, qx, qy, qz, qw]
    prefer_quat_sign: optional 4-vector (qx,qy,qz,qw) to keep sign continuity
                      (flips the result if dot < 0).
    """
    R = T[:3, :3]
    t = T[:3,  3]
    q = R_to_quat_xyzw(R)

    if prefer_quat_sign is not None:
        # keep quaternion sign consistent with a reference (avoids sudden flips)
        if np.dot(q, np.asarray(prefer_quat_sign, dtype=float)) < 0:
            q = -q
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=float)

import numpy as np
import open3d as o3d


def np_to_pcd(pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pc

def ensure_normals(pcd, radius, max_nn=120, towards=None):
    if len(pcd.points) < 30:
        return False
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    if towards is not None:
        pcd.orient_normals_towards_camera_location(np.asarray(towards, dtype=float))
    else:
        try:
            pcd.orient_normals_consistent_tangent_plane(50)
        except Exception:
            pass
    return pcd.has_normals()

import copy  # <-- add at top

def crop_target_around_transformed_source(src, tgt, T, radius):
    """Clone+transform source; do NOT mutate original."""
    if len(tgt.points) == 0 or len(src.points) == 0:
        return tgt
    src_tf = copy.deepcopy(src)   # <-- was: src.clone()
    src_tf.transform(T)           # transforms the copy only
    kdt = o3d.geometry.KDTreeFlann(tgt)
    mask = np.zeros(len(tgt.points), dtype=bool)
    tgt_np = np.asarray(tgt.points)
    step = max(1, len(src_tf.points)//4000)
    for p in np.asarray(src_tf.points)[::step]:
        _, idx, _ = kdt.search_radius_vector_3d(p, radius)
        mask[idx] = True
    if not mask.any():
        return tgt
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(tgt_np[mask])
    if tgt.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(tgt.normals)[mask])
    return cropped
  

  
def icp_refine_safe(src_pts: np.ndarray,
                    tgt_pts: np.ndarray,
                    T_init=np.eye(4),
                    cam_center=None,
                    crop_radius=0.035,     # loosen a bit so we have matches
                    max_corr=0.006,        # 6 mm (tight but workable)
                    max_iter=80):
    assert src_pts.ndim==2 and src_pts.shape[1]==3 and tgt_pts.ndim==2 and tgt_pts.shape[1]==3
    src = np_to_pcd(src_pts)
    tgt = np_to_pcd(tgt_pts)

    # normals
    ensure_normals(src, radius=0.02, max_nn=100, towards=cam_center)
    ensure_normals(tgt, radius=0.02, max_nn=150)

    # crop to overlap
    tgt_crop = crop_target_around_transformed_source(src, tgt, T_init, radius=crop_radius)
    if len(tgt_crop.points) < 50:
        tgt_crop = tgt  # if we cropped too aggressively

    # baseline metrics
    base_eval = o3d.pipelines.registration.evaluate_registration(src, tgt_crop, max_corr, T_init)
    print(f"Baseline: fitness={base_eval.fitness:.4f}, rmse={base_eval.inlier_rmse:.6f}")

    # run ICP
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt_crop, max_corr, T_init, est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    print(f"Refined : fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.6f}")

    # accept if RMSE decreased OR fitness improved meaningfully
    improved = (reg.inlier_rmse < base_eval.inlier_rmse * 0.995) or \
               (reg.fitness > base_eval.fitness * 1.02)

    if not improved:
        # as a second try, allow a bit looser threshold with GICP
        try:
            reg_g = o3d.pipelines.registration.registration_generalized_icp(
                src, tgt_crop, max_corr*1.5, T_init,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
            )
            print(f"GICP    : fitness={reg_g.fitness:.4f}, rmse={reg_g.inlier_rmse:.6f}")
            improved = (reg_g.inlier_rmse < base_eval.inlier_rmse * 0.995) or \
                       (reg_g.fitness > base_eval.fitness * 1.02)
            if improved:
                return reg_g.transformation, reg_g
        except Exception:
            pass
        # no real improvement → keep the original pose
        return T_init, base_eval

    return reg.transformation, reg

def export_like_loadPointCloud(points_np: np.ndarray):
    """Return (N, [POINT3D,...]) exactly like loadPointCloud()."""
    n = int(points_np.shape[0])
    p3dlist = [POINT3D(x, y, z) for x, y, z in points_np]
    return n, p3dlist

# --- If your points are in an Open3D PointCloud `pcd` ---
def export_o3d_like_loadPointCloud(pcd):
    pts = np.asarray(pcd.points, dtype=float)
    return export_like_loadPointCloud(pts)

def goicp_to_T(R_list, t_list):
    """
    Go-ICP -> 4x4 SE(3) transform.
    R_list: python list (3x3) from goicp.optimalRotation()
    t_list: python list (len 3 or 1x3/3x1) from goicp.optimalTranslation()
    Returns T such that X_target ≈ R * X_source + t  (i.e., source -> target)
    """
    R = np.array(R_list, dtype=float).reshape(3, 3)
    t = np.array(t_list, dtype=float).reshape(-1)
    if t.size != 3:
        t = t.reshape(3)

    # (optional but recommended) enforce a proper rotation numerically
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # fix possible reflection
        U[:, -1] *= -1
        R = U @ Vt

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

FX = 554.254691191187
FY = 554.254691191187
CX = 320.5
CY = 240.5

scene_ = "scene_25"
method_ = "ems"
number_samples_per_ray_ = 300
sampled_by_shape = []
device = 'cuda'
base_path = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/results/check"
base_path = "/home/elisabeth/repos/EMS-superquadric_fitting/results_EMS/check"
scene_dir = Path(base_path) / scene_




point_cloud = tools.read_with_open3d("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/pcds/"+scene_+"/cloud.pcd")
point_cloud = tools.remove_close_points(point_cloud, 0.003)
filtered_points, plane_points, plane_model = tools.remove_largest_plane(point_cloud, distance_threshold=0.003)
filtered_points, plane_points1, plane_model1 = tools.remove_largest_plane(filtered_points, distance_threshold=0.003)
point_cloud = tools.filter_by_z(point_cloud, -np.inf, 1.5)

all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor


labels, id2color = tools.load_seg_as_labels("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/segmasks/"+scene_+"/gtseg_ord-nearest_first_step-0.png", inflate_px=5)
point_labels = tools.label_points_from_seg(filtered_points, labels, FX, FY, CX, CY)
clusters = tools.split_points_by_label(filtered_points, point_labels)

clusters_pruned, rep = tools.prune_clusters_like(
    clusters,
    dbscan_min_samples=20,
    keep_quantile=0.95,
    min_points_after=30,
    gap_min=0.01,        # 5 cm gap to drop tiny islands
    rel_size_max=0.30    # drop components <20% of main if also far
)

clusters = clusters_pruned
fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
plot_functions.showPoints(filtered_points, scale_factor=0.0025)
mlab.show()
all_params_modeled = {}
idx = 0

tau_10 = 0.010
tau_5 = 0.005

voxel_scales = (0.01, 0.005, 0.002)              # 10mm → 5mm → 2mm
max_corr_multipliers = (2.0, 1.5, 1.2)           # tighter thresholds
iters = (80, 60, 50)                             # a few more iterations

# when estimating normals at each level, use a bigger neighborhood at fine scale
normal_radius_factor = (3.0, 4.0, 6.0)           # radius = voxel * factor
max_nn_levels = (40, 60, 80)

ap,bp,cp,dp = plane_model

    
scene_data = tools.load_scene_file_cam("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/scene_data/"+scene_+"_wrt_cam.json")
for lid, cluster in clusters.items():
    if lid ==1:
        continue
    if scene_ == "scene_104" and lid ==5:
        continue
    all_dists = []
    all_dists_object = []
    device = 'cuda'
    thetas_sq = []           # list[torch.Tensor] where each is shape (11,)
    sq_indices = []          # keep mapping to shapes_list indices if you need it
    
    # Pre-allocate collectors
    dmins_list   = []   # list of (N,) tensors       — per-shape nearest distances
    qmins_list   = []   # list of (N,3) tensors     — per-shape nearest sampled points (world)
    idx_list     = []   # list of (N,) long tensors — per-shape indices into that shape's samples
    shape_ids    = []   # list of ints              — index in shapes_list (or your own id)

    dmins_list_object   = []   # list of (N,) tensors       — per-shape nearest distances
    qmins_list_object   = []   # list of (N,3) tensors     — per-shape nearest sampled points (world)
    idx_list_object     = []   # list of (N,) long tensors — per-shape indices into that shape's samples
    shape_ids_object    = []   # list of ints              — index in shapes_list (or your own id)
    free = 0
    table_loss = 0
    penn_table = 0
    c = tools.read_cluster_yaml(base_path, scene_, cluster_id=lid, which="latest")
    # ---- save top-level fields into variables ----
    scene_val       = c["scene"]
    test_val        = c["test"]
    cluster_id_val  = c["cluster_id"]
    object_val = c["object"]
    timestamp_val   = c["timestamp"]
    n_shapes_val    = c["n_shapes"]
    shapes_list     = c["shapes"]   # list of dicts
    
    # ---- print top-level ----
    print("##################################################################################################################################################################")
    print(f"scene: {scene_val}")
    print(f"test: {test_val}")
    print(f"cluster_id: {cluster_id_val}")
    print(f"object: {object_val}")
    print(f"timestamp: {timestamp_val}")
    print(f"n_shapes: {n_shapes_val}")
    
    pcd, pts_np, nrm_np = tools.mesh_to_point_cloud("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/models/"+object_val+"/textured.obj",
    n_points=15000,
    method="poisson",
    scale=1.0,      
    voxel_size=0.004,
    estimate_normals=True,
    noise_std=0.0,     
    )
    
    
    pose7_obj_cam = tools.get_object_pose_cam(scene_data, object_val)


    T_refined = pose7_to_T(pose7_obj_cam)
    # T_c_o = object_pose_in_camera(T_w_o, pose7_cam_world)
    
    # pts_cam_before, nrm_cam = transform_obj_cloud_into_camera(pts_np, nrm_np, T_c_o)

    
    # Nm, a_points = export_like_loadPointCloud(cluster)
    # Nd, b_points = export_like_loadPointCloud(pts_cam_before)
    # goicp = GoICP();
    # goicp.loadModelAndData(Nm, a_points, Nd, b_points);
    # goicp.setDTSizeAndFactor(50, 1.0);
    # goicp.MSEThresh = 0.00005;
    # goicp.BuildDT();
    # goicp.Register();
    # Rot = goicp.optimalRotation()
    # t = goicp.optimalTranslation()
    # print(goicp.optimalRotation()); # A python list of 3x3 is returned with the optimal rotation
    # print(goicp.optimalTranslation());# A python list of 1x3 is returned with the optimal translation


    # T_refined = goicp_to_T(Rot,t)

    pts_cam, nrm_cam = apply_T_points_normals(pts_np, nrm_np, T_refined)
    



    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    mlab.view(azimuth=108.51, elevation=168.97, distance=0.5805, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.864720847838999), roll=-177.93)

    plot_functions.showPoints(pts_cam, scale_factor=0.0025, color=(0, 1, 0), figure=fig)
    plot_functions.showPoints(point_cloud, scale_factor=0.0025, color=(0.702, 0.702, 0.702), figure=fig)
    plot_functions.showPoints(cluster, scale_factor=0.0025, color=(1, 0, 0.0), figure=fig)
    mlab.show()
    
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    mlab.view(azimuth=108.51, elevation=168.97, distance=0.5805, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.864720847838999), roll=-177.93)
    plot_functions.showPoints(point_cloud, scale_factor=0.0025, color=(0.702, 0.702, 0.702), figure=fig)
    plot_functions.showPoints(pts_cam, scale_factor=0.0025, color=(0, 1, 0), figure=fig)

    # ---- iterate shapes, save per-shape variables, and print ----
    for i, s in enumerate(shapes_list):
    
        shape_type            = s["type"]
        theta                 = s["theta"]            # np.ndarray shape (11,)
        k                     = s["k"]                # float or None
        b                     = s["b"]                # float or None
        alpha                 = s["alpha"]            # float or None
        free_space_penalty    = s["free_space_penalty"]
        indices_good          = s["indices_good"]     # np.ndarray(int64)
        indices_bad           = s["indices_bad"]      # np.ndarray(int64)
        indices_remaining     = s["indices_remaining"]# np.ndarray(int64)

        print(f"\n  [shape {i}]")
        print(f"    type: {shape_type}")
        print(f"    theta (11): {theta.tolist()}")
        print(f"    k: {k}")
        print(f"    b: {b}")
        print(f"    alpha: {alpha}")
        print(f"    free_space_penalty: {free_space_penalty}")
        if shape_type == "superquadric":
            #plot_functions.showSuperquadrics(theta,b, alpha)
            points = torch.tensor(cluster, dtype=torch.float32, device='cuda')
            points_object = torch.tensor(pts_cam, dtype=torch.float32, device='cuda')
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device='cuda')
            if method_ =="psqf":
                distances = tools.sq_distances(points, theta_tensor).unsqueeze(1)
                distances_object = tools.sq_distances(points_object, theta_tensor).unsqueeze(1)

                all_dists.append(distances)
                all_dists_object.append(distances_object)
                
                th = torch.as_tensor(s["theta"], dtype=torch.float32, device=device)
                thetas_sq.append(th)
                sq_indices.append(i)
                free += tools.compute_free_space(number_samples_per_ray_, all_points, points, theta_tensor, None, None, shape_type)
                
                plane_normal = torch.tensor([ap,bp,cp], dtype=torch.float32, device=theta_tensor.device)
    
                table_pts = tools.make_table_grid_points(
                        theta=theta_tensor,
                        plane_normal=plane_normal,
                        plane_d=torch.tensor(float(dp), device=theta_tensor.device),
                        half_size=1.0,      # ±1 m in both in-plane directions
                        step=0.01,          # 2 cm spacing; adjust as you like
                        offset_above=0.01,    # or e.g. 0.005 to sit 5 mm above the plane
                    )
                
                table_loss+= tools.table_transverse_loss(table_pts, theta_tensor)
                penn_table_val = tools.table_violation_metrics(table_pts, theta_tensor)
                penn_table+= penn_table_val
            else:
                th = torch.as_tensor(s["theta"], dtype=torch.float32, device=device)
                thetas_sq.append(th)
                sq_indices.append(i)
                
                plane_normal = torch.tensor([ap,bp,cp], dtype=torch.float32, device=theta_tensor.device)
    
                table_pts = tools.make_table_grid_points(
                        theta=theta_tensor,
                        plane_normal=plane_normal,
                        plane_d=torch.tensor(float(dp), device=theta_tensor.device),
                        half_size=1.0,      # ±1 m in both in-plane directions
                        step=0.01,          # 2 cm spacing; adjust as you like
                        offset_above=0.01,    # or e.g. 0.005 to sit 5 mm above the plane
                    )
                
                free += tools.compute_free_space(number_samples_per_ray_, all_points, points, theta_tensor, None, None, shape_type)
                table_loss+= tools.table_transverse_loss(table_pts, theta_tensor)
                
                penn_table_val = tools.table_violation_metrics(table_pts, theta_tensor)
                print("penn_table_val: ", penn_table_val)
                penn_table+= penn_table_val
                
                pts_world, normals_world_t = tools.sample_superquadric_points_torch(theta, b=s.get("b", 0.0) or 0.0, alpha=s.get("alpha", 0.0) or 0.0,
                                                    threshold=1e-2, num_limit=1000, arclength=0.003, device='cuda')
                print('norm range before call:',
                normals_world_t.norm(dim=1).min().item(),
                normals_world_t.norm(dim=1).max().item(),
                normals_world_t.norm(dim=1).mean().item())
                normals_world_t = normals_world_t / (normals_world_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
)

                dmin, idx, qmin,_ , debug= tools.nearest_on_superquadric_mixed(points, pts_world, normals_world_t)
                print("debug: ", debug)
                # 3) store
                dmins_list.append(dmin)
                qmins_list.append(qmin)
                idx_list.append(idx)
                shape_ids.append(i)   # or s["cluster_id"], etc.

                
                print("dmins_list: ", dmins_list)
                
                sampled_by_shape.append(pts_world)
                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                #plot_functions.showPoints(pts_world_np, scale_factor=0.0025, color=(0, 0, 1.0), figure=fig)
                
                
                dmin_object, idx_object, qmin_object, n_q_object, debug = tools.nearest_on_superquadric_mixed(points_object, pts_world, normals_world_t)
                print("debug: ", debug)

                # 3) store
                dmins_list_object.append(dmin_object)
                qmins_list_object.append(qmin_object)
                idx_list_object.append(idx_object)
                shape_ids_object.append(i)
                sampled_by_shape.append(pts_world)
                
                print("dmins_list_object: ", dmins_list_object)
                plot_functions.show_vectors(
                qmin_object,
                qmin_object + 0.02 * n_q_object,   # 2 cm arrows (adjust as needed)
                    color=(1,0,0), mode='arrow', every=10, scale_factor=1.0, figure=fig
                )

                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
            # plot_functions.showPoints(pts_world_np, scale_factor=0.0025, color=(0, 1, 0.0), figure=fig)

            
        elif shape_type == "supertoroid":
            plot_functions.showSupertoroid(theta)
            points = torch.tensor(cluster, dtype=torch.float32, device='cuda')
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device='cuda')
            distances = tools.st_distances(points, theta_tensor).unsqueeze(1)

            
            all_dists.append(distances)
            free+= tools.compute_free_space(number_samples_per_ray_, all_points, points, theta_tensor, None, None, shape_type)
            
            points_object = torch.tensor(pts_cam, dtype=torch.float32, device='cuda')
            distances_object = tools.st_distances(points_object, theta_tensor).unsqueeze(1)
            all_dists_object.append(distances_object)
            plane_normal = torch.tensor([ap,bp,cp], dtype=torch.float32, device=theta_tensor.device)
    
            table_pts = tools.make_table_grid_points(
                        theta=theta_tensor,
                        plane_normal=plane_normal,
                        plane_d=torch.tensor(float(dp), device=theta_tensor.device),
                        half_size=1.0,      # ±1 m in both in-plane directions
                        step=0.01,          # 2 cm spacing; adjust as you like
                        offset_above=0.01,    # or e.g. 0.005 to sit 5 mm above the plane
                    )
                
            table_loss += tools.table_transverse_loss_toroid(table_pts, theta_tensor)
            penn_table_val = tools.table_violation_metrics(table_pts, theta_tensor)
            penn_table+= penn_table_val
            
            if method_ =='ems':
                
                pts_world = tools.sample_supertoroid_points_torch(theta, b=s.get("b", 0.0) or 0.0, alpha=s.get("alpha", 0.0) or 0.0,
                                                    threshold=1e-2, num_limit=10000, arclength=0.003, device='cuda')
                
                dmin, idx, qmin = tools.nearest_on_superquadric(points, pts_world)
                
                # 3) store
                dmins_list.append(dmin)
                qmins_list.append(qmin)
                idx_list.append(idx)
                shape_ids.append(i)   # or s["cluster_id"], etc.

                
                print("dmin: ", dmin)
                
                sampled_by_shape.append(pts_world)
                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                dmin_object, idx_object, qmin_object = tools.nearest_on_superquadric(points_object, pts_world)
                
                # 3) store
                dmins_list_object.append(dmin_object)
                qmins_list_object.append(qmin_object)
                idx_list_object.append(idx_object)
                shape_ids_object.append(i)   # or s["cluster_id"], etc.

                                
                sampled_by_shape.append(pts_world)
                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                thetas_sq.append(theta_tensor)
                sq_indices.append(i)


        else:
            plot_functions.showTaperedSuperparaboloidWithBase(theta,k)
            points = torch.tensor(cluster, dtype=torch.float32, device='cuda')
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device='cuda')
            k_tensor = torch.tensor(k, dtype=torch.float32, device='cuda')
            distances = tools.spb_distances_autograd(points, theta_tensor, k_tensor).unsqueeze(1)
            all_dists.append(distances)
            
            points_object = torch.tensor(pts_cam, dtype=torch.float32, device='cuda')
            distances_object = tools.spb_distances_autograd(points_object, theta_tensor, k_tensor).unsqueeze(1)
            all_dists_object.append(distances_object)
            
            if method_ == 'ems':
                pts_world = tools.sample_tapered_superparaboloid_points_torch(theta, k, b=s.get("b", 0.0) or 0.0, alpha=s.get("alpha", 0.0) or 0.0,
                                                    threshold=1e-2, num_limit=8000, arclength=0.005, device='cuda')
                
                dmin, idx, qmin = tools.nearest_on_superquadric(points, pts_world)
                
                # 3) store
                dmins_list.append(dmin)
                qmins_list.append(qmin)
                idx_list.append(idx)
                shape_ids.append(i)   # or s["cluster_id"], etc.

                
                print("dmin: ", dmin)
                
                sampled_by_shape.append(pts_world)
                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                dmin_object, idx_object, qmin_object = tools.nearest_on_superquadric(points_object, pts_world)
                
                # 3) store
                dmins_list_object.append(dmin_object)
                qmins_list_object.append(qmin_object)
                idx_list_object.append(idx_object)
                shape_ids_object.append(i)   # or s["cluster_id"], etc.

                
                print("dmin: ", dmin)
                
                sampled_by_shape.append(pts_world)
                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                thetas_sq.append(theta_tensor)
                sq_indices.append(i)
                

        points = torch.tensor(cluster, dtype=torch.float32, device=device)
        # chosen_dist, chosen_q, chosen_idx = closest_points_and_outer_distance(points, thetas_sq, device=device)
        # print("chosen_dist: ", chosen_dist)
        # print(f"    indices_good (n={indices_good.size}): {indices_good.tolist()}")
        # print(f"    indices_bad (n={indices_bad.size}): {indices_bad.tolist()}")
        # print(f"    indices_remaining (n={indices_remaining.size}): {indices_remaining.tolist()}")
        # Compute per-shape nearest (no global selection)

        #plot_functions.showPoints(cluster, scale_factor=0.0025, color=(1, 0, 0.0), figure=fig)
        print(len(cluster))
        
    if method_=="psqf":
        d_all = torch.cat(all_dists, dim=1)  # (N, M)
        d_min, _ = torch.min(d_all, dim=1)   # (N,)
        coverage_5 =  (d_min < tau_5).float().mean()
        coverage_10 =  (d_min < tau_10).float().mean()
        rmse = torch.sqrt(torch.mean(d_min**2)).item()
        mae = torch.mean(torch.abs(d_min)).item()
        penn_table=penn_table_val/len(shapes_list)
        
        d_all_object = torch.cat(all_dists_object, dim=1)  # (N, M)
        d_min_object, _ = torch.min(d_all_object, dim=1)   # (N,)
        coverage_5_object =  (d_min_object < tau_5).float().mean()
        coverage_10_object =  (d_min_object < tau_10).float().mean()
        rmse_object = torch.sqrt(torch.mean(d_min_object**2)).item()
        mae_object = torch.mean(torch.abs(d_min_object)).item()
        
    else:
        D = torch.stack(dmins_list, dim=1)   # (N, M)
        Q = torch.stack(qmins_list, dim=1)   # (N, M, 3)
        chosen_d, chosen_q, chosen_m = tools.select_outside_nearest_from_samples(D, Q, thetas_sq)
        tau_5, tau_10 = 0.005, 0.010
        coverage_5  = (chosen_d < tau_5 ).float().mean().item()
        coverage_10 = (chosen_d < tau_10).float().mean().item()
        rmse = torch.sqrt((chosen_d**2).mean()).item()
        mae  = torch.mean(torch.abs(chosen_d)).item()
        penn_table=penn_table/len(shapes_list)

        Dobject = torch.stack(dmins_list_object, dim=1)   # (N, M)
        Qobject = torch.stack(qmins_list_object, dim=1)   # (N, M, 3)
        chosen_d_object, chosen_q_object, chosen_m_object = tools.select_outside_nearest_from_samples(Dobject, Qobject, thetas_sq)
        tau_5, tau_10 = 0.005, 0.010
        coverage_5_object  = (chosen_d_object < tau_5 ).float().mean().item()
        coverage_10_object = (chosen_d_object < tau_10).float().mean().item()
        rmse_object = torch.sqrt((chosen_d_object**2).mean()).item()
        mae_object  = torch.mean(torch.abs(chosen_d_object)).item()
        
    # d_all = torch.cat(all_dists, dim=1)  # (N, M)
    # d_min, _ = torch.min(d_all, dim=1)   # (N,)
        
    # coverage_5 =  (d_min < tau_5).float().mean()
    # coverage_10 =  (d_min < tau_10).float().mean()

    # rmse = torch.sqrt(torch.mean(d_min**2)).item()
    # mae = torch.mean(torch.abs(d_min)).item()
    
    # d_use = torch.nan_to_num(chosen_dist, nan=1e9, posinf=1e9, neginf=1e9)  # safety

    # coverage_5  = (d_use < tau_5).float().mean().item()
    # coverage_10 = (d_use < tau_10).float().mean().item()

    # rmse = torch.sqrt(torch.mean(d_use**2)).item()
    # mae  = torch.mean(torch.abs(d_use)).item()

    print("rmse: ", rmse)
    print("mae: ", mae)
    print("coverage5mm: ", coverage_5)
    print("coverage5+10mm: ", coverage_10)
    print("free: ", free)
    print("table loss:", table_loss)
    print("penn_table: ", penn_table)
    print("------------------------------ FULL OBJECT CLOUD-----------------------------")
    print("rmse: ", rmse_object)
    print("mae: ", mae_object)
    print("coverage5mm: ", coverage_5_object)
    print("coverage5+10mm: ", coverage_10_object)
    mlab.show()