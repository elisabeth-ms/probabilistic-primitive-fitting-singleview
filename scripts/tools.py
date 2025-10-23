from __future__ import annotations
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import cv2
from pathlib import Path
from typing import Dict, Any, List, Union
import yaml
import re
from scipy.spatial.transform import Rotation as R
import torch
import json, numpy as np
import torch.nn.functional as F
import plot_functions
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from numpy.linalg import pinv

def get_translation_numpy(theta_np):
    """
    theta_np: numpy array of shape (11,)
    returns: (3,) translation vector
    """
    return theta_np[8:11]
  


def build_rotation_matrix_numpy(theta_np):
    """
    theta_np: numpy array of shape (11,)
    returns: 3x3 rotation matrix
    """
    rz, ry, rx = theta_np[5:8]
    rot = R.from_euler('ZYX', [rz, ry, rx])  # careful with order!!
    return rot.as_matrix()


def build_rotation_matrix(euler_angles):
    rz, ry, rx = euler_angles

    cosx = torch.cos(rx)
    sinx = torch.sin(rx)
    cosy = torch.cos(ry)
    siny = torch.sin(ry)
    cosz = torch.cos(rz)
    sinz = torch.sin(rz)

    # Rotation around x-axis
    Rx = torch.stack([
        torch.stack([torch.tensor(1., device=euler_angles.device), torch.tensor(0., device=euler_angles.device), torch.tensor(0., device=euler_angles.device)]),
        torch.stack([torch.tensor(0., device=euler_angles.device), cosx, -sinx]),
        torch.stack([torch.tensor(0., device=euler_angles.device), sinx,  cosx])
    ])

    # Rotation around y-axis
    Ry = torch.stack([
        torch.stack([cosy, torch.tensor(0., device=euler_angles.device), siny]),
        torch.stack([torch.tensor(0., device=euler_angles.device), torch.tensor(1., device=euler_angles.device), torch.tensor(0., device=euler_angles.device)]),
        torch.stack([-siny, torch.tensor(0., device=euler_angles.device), cosy])
    ])

    # Rotation around z-axis
    Rz = torch.stack([
        torch.stack([cosz, -sinz, torch.tensor(0., device=euler_angles.device)]),
        torch.stack([sinz,  cosz, torch.tensor(0., device=euler_angles.device)]),
        torch.stack([torch.tensor(0., device=euler_angles.device), torch.tensor(0., device=euler_angles.device), torch.tensor(1., device=euler_angles.device)])
    ])

    R = Rz @ Ry @ Rx
    return R

def read_with_open3d(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts
  

def remove_close_points(points, min_dist=0.01):
    """
    Removes points that are closer than min_dist to each other.

    Parameters:
        points (np.ndarray): Nx3 array of 3D points.
        min_dist (float): minimum allowed distance between any two points.

    Returns:
        np.ndarray: Filtered points.
    """
    tree = cKDTree(points)
    mask = np.ones(len(points), dtype=bool)

    for i in range(len(points)):
        if not mask[i]:
            continue
        # Find neighbors within min_dist (excluding self)
        indices = tree.query_ball_point(points[i], r=min_dist)
        indices = [j for j in indices if j > i]
        mask[indices] = False  # Mark close duplicates for removal

    return points[mask]

def _knn_eps(points: np.ndarray, k: int = 8, q: float = 0.60, scale: float = 1.0) -> float:
    """Distance to k-th NN, take a lower quantile q (0.5–0.7 works), times scale."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)      # [:,0] is 0 (self)
    dk = dists[:, -1]
    return float(np.quantile(dk, q) * scale)

def _largest_component_adaptive(points: np.ndarray,
                                min_samples: int = 20,
                                start_q: float = 0.70,
                                min_q: float = 0.40,
                                shrink: float = 0.75,
                                max_iter: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try DBSCAN with progressively smaller eps (via kNN quantile) until
    the largest component dominates. Returns (labels, uniq_non_noise).
    """
    q = start_q
    uniq = []
    labels = np.full(points.shape[0], -1, dtype=int)

    for _ in range(max_iter):
        eps = _knn_eps(points, k=8, q=q, scale=1.0)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
        uniq = [l for l in np.unique(labels) if l != -1]
        if not uniq:
            # only noise -> shrink eps and try again
            q = max(min_q, q * shrink)
            continue

        # check dominance of largest component
        sizes = [np.sum(labels == l) for l in uniq]
        sizes_sorted = np.sort(sizes)[::-1]
        if len(sizes_sorted) == 1 or sizes_sorted[0] >= 3 * sizes_sorted[1]:
            break
        q = max(min_q, q * shrink)

    return labels, uniq
  
def _prune_one(points: np.ndarray,
               *,
               dbscan_min_samples: int = 20,
               keep_quantile: float = 0.98,
               min_points_after: int = 30,
               gap_min: float = 0.05,          # min centroid gap to drop tiny extra blobs
               rel_size_max: float = 0.20) -> np.ndarray:
    if points.shape[0] == 0:
        return points

    # --- 1) keep dominant connected component (adaptive eps) ---
    labels, uniq = _largest_component_adaptive(points, min_samples=dbscan_min_samples)
    if uniq:
        # largest by size
        sizes = {l: int(np.sum(labels == l)) for l in uniq}
        main = max(uniq, key=lambda l: sizes[l])
        comp = points[labels == main]

        # drop extra tiny far components (robust gap rule)
        main_c = comp.mean(axis=0)
        for l in uniq:
            if l == main:
                continue
            comp_l = points[labels == l]
            if comp_l.size == 0:
                continue
            if comp_l.shape[0] <= rel_size_max * sizes[main]:
                gap = np.linalg.norm(comp_l.mean(axis=0) - main_c)
                if gap >= gap_min:
                    # mark as noise (dropped)
                    labels[labels == l] = -1
        comp = points[labels == main]
    else:
        comp = points  # fallback if everything was noise

    if comp.shape[0] < min_points_after:
        return comp

    # --- 2) Mahalanobis tail trim on the main blob ---
    X = comp
    center = np.median(X, axis=0)
    Xc = X - center
    C = np.cov(Xc.T) + 1e-6 * np.eye(3)
    Cinv = pinv(C)
    d2 = np.einsum('ni,ij,nj->n', Xc, Cinv, Xc)
    thr = np.quantile(d2, keep_quantile)
    keep = d2 <= thr
    X_kept = X[keep]
    if X_kept.shape[0] < min_points_after:
        return comp
    return X_kept
  
def prune_clusters_like(clusters: Dict[int, np.ndarray],
                        *,
                        dbscan_min_samples: int = 20,
                        keep_quantile: float = 0.98,
                        min_points_after: int = 30,
                        gap_min: float = 0.05,
                        rel_size_max: float = 0.20) -> Tuple[Dict[int, np.ndarray], Dict]:
    """
    One pruned cluster per original cluster. Same keys preserved.
    """
    cleaned: Dict[int, np.ndarray] = {}
    report: Dict[int, dict] = {}
    for lab, pts in clusters.items():
        orig = int(pts.shape[0])
        pts_new = _prune_one(pts,
                             dbscan_min_samples=dbscan_min_samples,
                             keep_quantile=keep_quantile,
                             min_points_after=min_points_after,
                             gap_min=gap_min,
                             rel_size_max=rel_size_max)
        kept = int(pts_new.shape[0])
        cleaned[lab] = pts_new
        report[lab] = {"orig": orig, "kept": kept, "dropped": orig - kept}
    return cleaned, report
  
def remove_largest_plane(points_np, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Removes the largest plane (e.g., a flat table) from a point cloud using RANSAC.

    Args:
        points_np (np.ndarray): Nx3 numpy array of points.
        distance_threshold (float): Max distance from the plane to consider as inlier.
        ransac_n (int): Number of points to sample per RANSAC iteration.
        num_iterations (int): Number of RANSAC iterations.

    Returns:
        np.ndarray: Point cloud with the largest plane removed.
        np.ndarray: Points on the plane (optional, for visualization/debugging).
    """
    # # Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Fit plane using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

    # Separate inliers (plane) and outliers (rest)
    plane_cloud = pcd.select_by_index(inliers)
    remaining_cloud = pcd.select_by_index(inliers, invert=True)

    # Convert back to numpy
    remaining_points = np.asarray(remaining_cloud.points)
    plane_points = np.asarray(plane_cloud.points)

    return remaining_points, plane_points, plane_model

def filter_by_z(points, z_min=-np.inf, z_max=np.inf):
    """
    Removes points with Z outside the specified range.

    Args:
        points (np.ndarray): Nx3 array of points.
        z_min (float): Minimum allowed Z value.
        z_max (float): Maximum allowed Z value.

    Returns:
        np.ndarray: Filtered point cloud.
    """
    z = points[:, 2]
    mask = (z >= z_min) & (z <= z_max)
    return points[mask]

def _inflate_into_background(labels: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Expand each label up to radius_px pixels into background (-1) only.
    Uses nearest-label assignment (no order bias, no crossing into other labels).
    """
    if radius_px <= 0:
        return labels

    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        raise RuntimeError("scipy is required for inflate_px. Install with: pip install scipy")

    fg = labels >= 0               # foreground (labeled)
    bg = ~fg                       # background (== -1)

    if not np.any(bg):
        return labels.copy()

    # Distance from each background pixel to the nearest foreground pixel,
    # plus the indices (row,col) of that nearest foreground pixel.
    dist, idx = distance_transform_edt(bg, return_indices=True)

    # For each pixel, nearest foreground coordinates:
    rr, cc = idx[0], idx[1]
    nearest_lab = labels[rr, cc]

    out = labels.copy()
    # Fill only background pixels within radius with the nearest label
    fill = bg & (dist <= float(radius_px))
    out[fill] = nearest_lab[fill]
    return out

def load_seg_as_labels(seg_png_path, bg_colors=[(0,0,0)], inflate_px: int = 0):
    """
    Reads a color segmentation PNG and returns:
      labels (H,W) int32 with -1 for background,
      id2color: dict[label_id] = (R,G,B)
    Optional: inflate_px grows labels into background by N pixels.
    """
    seg_bgr = cv2.imread(seg_png_path, cv2.IMREAD_COLOR)
    if seg_bgr is None:
        raise FileNotFoundError(seg_png_path)
    seg = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = seg.shape

    rgb = seg.reshape(-1,3).astype(np.int32)
    col_hash = (rgb[:,0] << 16) | (rgb[:,1] << 8) | rgb[:,2]
    uniq, inv = np.unique(col_hash, return_inverse=True)
    labels = inv.reshape(H, W).astype(np.int32)

    id2color = {i: ((c>>16)&255, (c>>8)&255, c&255) for i, c in enumerate(uniq)}
    bg_hashes = {(r<<16)|(g<<8)|b for (r,g,b) in bg_colors}
    bg_ids = {i for i, c in enumerate(uniq) if c in bg_hashes}
    if bg_ids:
        labels[np.isin(labels, list(bg_ids))] = -1

    # Inflate a bit into background (if requested)
    if inflate_px > 0:
        labels = _inflate_into_background(labels, inflate_px)

    return labels, id2color

def label_points_from_seg(points_np, labels, fx, fy, cx, cy):
    """
    points_np: (N,3) XYZ in camera optical frame (meters)
    labels: (H,W) int32 from load_seg_as_labels
    returns: point_labels (N,) int32, -1 for unlabeled/out of bounds
    """
    Z = points_np[:, 2]
    # project (X,Y,Z) -> (u,v)
    u = np.round(fx * (points_np[:,0] / Z) + cx).astype(np.int32)
    v = np.round(fy * (points_np[:,1] / Z) + cy).astype(np.int32)

    H, W = labels.shape
    valid = (Z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    point_labels = np.full(points_np.shape[0], -1, dtype=np.int32)
    point_labels[valid] = labels[v[valid], u[valid]]
    return point_labels


def split_points_by_label(points_np, point_labels, ignore_label=-1):
    """
    Returns dict: {label_id: (M_i,3) points}
    """
    clusters = {}
    for lab in np.unique(point_labels):
        if lab == ignore_label:
            continue
        clusters[lab] = points_np[point_labels == lab]
    return clusters

def _pick_test(scene_dir: Path, which: Union[int, str]) -> int:
    if isinstance(which, int):
        return which
    if which != "latest":
        raise ValueError("which must be an int or 'latest'")
    tests = []
    for p in scene_dir.iterdir():
        m = re.match(r"^test(\d+)$", p.name)
        if p.is_dir() and m:
            tests.append(int(m.group(1)))
    if not tests:
        raise FileNotFoundError(f"No test* folders in {scene_dir}")
    return max(tests)

def _as_float(x):
    return None if x is None else float(x)

def _as_float_arr11(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size != 11:
        raise ValueError(f"theta must have 11 elements, got {arr.size}")
    return arr

def _as_int_arr(x):
    if x is None:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(x, dtype=np.int64).reshape(-1)
  
def read_cluster_yaml(
    base_path: Union[str, Path],
    scene_: str,
    cluster_id: int,
    which: Union[int, str] = "latest"
) -> Dict[str, Any]:
    """
    Load ONE cluster{cluster_id}.yaml and return a dict:
      {
        'scene': str,
        'test': int,
        'cluster_id': int,
        'object': str,
        'timestamp': str,
        'n_shapes': int,
        'shapes': [
          {
            'type': str,
            'theta': np.ndarray(11,),
            'k': Optional[float],
            'b': Optional[float],
            'alpha': Optional[float],
            'free_space_penalty': float,
            'indices_good': np.ndarray(int64),
            'indices_bad': np.ndarray(int64),
            'indices_remaining': np.ndarray(int64),
          }, ...
        ]
      }
    """
    base_path = Path(base_path)
    scene_dir = base_path / scene_
    
    test_n = _pick_test(scene_dir, which)
    f = scene_dir / f"test{test_n}" / f"cluster{cluster_id}"/f"cluster{cluster_id}.yaml"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    data = yaml.safe_load(f.read_text()) or {}

    # normalize shapes
    shapes_out: List[Dict[str, Any]] = []
    for s in data.get("shapes", []) or []:
        shapes_out.append({
            "type": str(s.get("type", "superquadric")),
            "theta": _as_float_arr11(s.get("theta", [])),
            "k": _as_float(s.get("k")),
            "b": _as_float(s.get("b")),
            "alpha": _as_float(s.get("alpha")),
            "free_space_penalty": float(s.get("free_space_penalty", 0.0)),
            "indices_good": _as_int_arr(s.get("indices_good")),
            "indices_bad": _as_int_arr(s.get("indices_bad")),
            "indices_remaining": _as_int_arr(s.get("indices_remaining")),
        })

    return {
        "scene": str(data.get("scene", "")),
        "test": int(data.get("test", test_n)),
        "cluster_id": int(data.get("cluster_id", cluster_id)),
        "object": str(data.get("object", "")),
        "timestamp": str(data.get("timestamp", "")),
        "n_shapes": int(data.get("n_shapes", len(shapes_out))),
        "shapes": shapes_out,
    }

def sq_distances(points, theta):
    """
    Compute distances from points to the superquadric surface.
    This version is differentiable if called with grad enabled.
    """


    # Rotation + translation
    R = build_rotation_matrix(theta[5:8])
    t = theta[8:11]
    points_local = points @ R - t @ R


    # parameters
    a1 = theta[2].abs().clamp_min(1e-6)
    a2 = theta[3].abs().clamp_min(1e-6)
    a3 = theta[4].abs().clamp_min(1e-6)
    e1, e2 = theta[0], theta[1]

    # normalize
    x_ = points_local[:, 0] / a1
    y_ = points_local[:, 1] / a2
    z_ = points_local[:, 2] / a3

    # superquadric inside-outside function
    term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
    term2 = (torch.abs(z_)**(2/e1))
    inside_outside = term1 + term2

    # radial norm
    r_norm = torch.norm(points_local, dim=1)

    # distance formula
    distances = r_norm * torch.abs(inside_outside**(-e1/2) - 1)

    return distances  # (N,) tensor

def st_distances(points, theta, eps=1e-6):
    """
    Supertoroid distance surrogate:  d ≈ |F| / ||∇F||.
    theta = [e_eta, e_omega, Rmaj, a_r, a_z, rz, ry, rx, tx, ty, tz]
    F = ((|rho|/a_r)^(2/e_eta) + (|z|/a_z)^(2/e_eta))^(e_eta/2) - 1,
    where rho = r_xy(e_omega) - Rmaj and r_xy = (|x|^p + |y|^p)^(1/p), p=2/e_omega.
    """
    tiny = 1e-12

    # pose
    Rm = build_rotation_matrix(theta[5:8])   # (3,3)
    t  = theta[8:11]                         # (3,)
    pl = points @ Rm - t @ Rm

    x, y, z = pl[:,0], pl[:,1], pl[:,2]

    # params
    e_eta   = theta[0].abs().clamp_min(0.05)
    e_omega = theta[1].abs().clamp_min(0.05)
    Rmaj    = theta[2].abs().clamp_min(tiny)
    a_r     = theta[3].abs().clamp_min(tiny)
    a_z     = theta[4].abs().clamp_min(tiny)

    # implicit F
    p = 2.0 / e_omega
    q = 2.0 / e_eta

    r_xy = (x.abs().clamp_min(tiny).pow(p) + y.abs().clamp_min(tiny).pow(p)).pow(1.0/p)
    rho  = r_xy - Rmaj
    u = (rho.abs() / a_r).clamp_min(tiny).pow(q)
    v = (z.abs()   / a_z).clamp_min(tiny).pow(q)
    G = (u + v).clamp_min(tiny).pow(1.0/q)       # ==1 on surface
    phi = G - 1.0

    # grad norm via autograd w.r.t. points only (pose detached for efficiency)
    Rm_d = Rm.detach()
    t_d  = t.detach()
    pts = points.detach().requires_grad_(True)
    pl2 = (pts - t_d) @ Rm_d
    x2, y2, z2 = pl2[:,0], pl2[:,1], pl2[:,2]
    r_xy2 = (x2.abs().clamp_min(tiny).pow(p) + y2.abs().clamp_min(tiny).pow(p)).pow(1.0/p)
    rho2  = r_xy2 - Rmaj
    u2 = (rho2.abs() / a_r).clamp_min(tiny).pow(q)
    v2 = (z2.abs()   / a_z).clamp_min(tiny).pow(q)
    G2 = (u2 + v2).clamp_min(tiny).pow(1.0/q)
    phi2 = G2 - 1.0

    g = torch.autograd.grad(phi2.sum(), pts, create_graph=False, retain_graph=False)[0]
    gradnorm = g.norm(dim=1).clamp_min(eps).detach()

    d = phi.abs() / gradnorm
    return d


def spb_distances_autograd(points, theta, k, eps=1e-9):
    tiny = 1e-12
    Rm = build_rotation_matrix(theta[5:8])
    t  = theta[8:11]

    e1 = theta[0].abs().clamp_min(0.05)
    e2 = theta[1].abs().clamp_min(0.05)
    a1 = theta[2].abs().clamp_min(tiny)
    a2 = theta[3].abs().clamp_min(tiny)
    a3 = theta[4].abs().clamp_min(tiny)
    k  = torch.as_tensor(k, device=points.device, dtype=points.dtype).clamp_min(0.0)

    # local coords for F (no grad on pose/params for gradnorm)
    Rm_d, t_d = Rm.detach(), t.detach()
    pts = points.detach().requires_grad_(True)
    pl = (pts - t_d) @ Rm_d
    x, y, z = pl[:,0], pl[:,1], pl[:,2]

    p = 2.0 / e2
    u = (x.abs().clamp_min(tiny)/a1).pow(p) + (y.abs().clamp_min(tiny)/a2).pow(p)
    term1 = u.clamp_min(tiny).pow(e2/2.0)
    # keep one–sided opening but smooth: use softplus instead of hard clamp if desired
    zplus = torch.clamp(z, min=0.0)
    term2 = (zplus.clamp_min(tiny)/a3).pow(1.0/e1)

    phi = term1 - term2 - k

    g = torch.autograd.grad(phi.sum(), pts, create_graph=False, retain_graph=False)[0]
    gradnorm = g.norm(dim=1).clamp_min(eps).detach()

    d = phi.abs().detach() / gradnorm
    return d


def active_idx_ellipsoid(points, theta, factor=2.0, pad=0.2):
    # θ sin gradiente para no meter la selección en el grafo
    th = theta.detach()
    euler, t = th[5:8], th[8:11]
    R = build_rotation_matrix(euler)
    pts_local = points @ R - t @ R

    # semiejes seguros
    a1 = th[2].abs().clamp_min(1e-6)
    a2 = th[3].abs().clamp_min(1e-6)
    a3 = th[4].abs().clamp_min(1e-6)

    # métrica elipsoidal p=2 (barata)
    x = pts_local[:,0] / a1
    y = pts_local[:,1] / a2
    z = pts_local[:,2] / a3
    q = x*x + y*y + z*z

    thr = (factor + pad)**2  # margen para evitar “parpadeo”
    idx = torch.nonzero(q <= thr, as_tuple=False).squeeze(-1)
    return idx

# Differentiable inverse using a short fixed-point loop
def unbend_points_torch(P_def, b, alpha, iters=5):
    """
    Approximate D_b^{-1}. Inputs/outputs are (N,3) in the *local* frame.
    """
    device = P_def.device
    b = torch.clamp(torch.as_tensor(b, device=device, dtype=P_def.dtype), min=1e-6)
    alpha = torch.as_tensor(alpha, device=device, dtype=P_def.dtype)
    invb = 1.0 / b
    x_p, y_p, z_p = P_def[:, 0], P_def[:, 1], P_def[:, 2]

    # init guess: no bending
    x = x_p.clone()
    y = y_p.clone()
    # reasonable z from z' ≈ (invb - r0) * sin(b z)  -> start with z'=z
    z = z_p.clone()

    for _ in range(iters):
        # current r from current (x,y)
        rho  = torch.sqrt(x * x + y * y + 1e-8)
        ang  = torch.atan2(y, x)
        r    = torch.cos(alpha - ang) * rho                  # r(x,y)

        # infer gamma (=> z) from z' = (invb - r) * sin(gamma)
        denom = torch.clamp(invb - r, min=1e-6)
        sin_g = torch.clamp(z_p / denom, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        gamma = torch.asin(sin_g)
        # choose principal branch; small b keeps things stable
        z = gamma / b

        # with gamma and r, compute shift s = R - r = (invb - r)(1 - cos gamma)
        s = denom * (1.0 - torch.cos(gamma))

        # recover (x,y) from x' = x + s cos α, y' = y + s sin α
        x = x_p - s * torch.cos(alpha)
        y = y_p - s * torch.sin(alpha)

    out = torch.stack([x, y, z], dim=1)
    return out


def sq_inside_near_only(points, theta, b, alpha, far_const=20.0):
    # máscara dinámica en *cada* iteración
    idx = active_idx_ellipsoid(points, theta, factor=2.0, pad=0.3)

    F = torch.full((points.shape[0],), float(far_const), device=points.device)
    if idx.numel() == 0:
        return F

    # --- evalúas tu supercuádrica SOLO en idx ---
    e1, e2 = theta[0], theta[1]
    a1 = theta[2].abs().clamp_min(1e-6)
    a2 = theta[3].abs().clamp_min(1e-6)
    a3 = theta[4].abs().clamp_min(1e-6)
    R  = build_rotation_matrix(theta[5:8])
    t  = theta[8:11]

    pl = points[idx] @ R - t @ R
    
    if b is not None and alpha is not None:
      pl = unbend_points_torch(pl, b, alpha)
    x_ = pl[:,0] / a1; y_ = pl[:,1] / a2; z_ = pl[:,2] / a3

    # potencias estables
    def spow(u, p, eps=1e-8, umax=1e2, max_log=100.0):
        u = torch.clamp(u.abs(), min=eps, max=umax)
        return torch.exp(torch.clamp(p * torch.log(u), -max_log, max_log))

    px = spow(x_, 2.0 / e2)
    py = spow(y_, 2.0 / e2)
    s  = torch.clamp(px + py, min=1e-8)
    term1 = torch.exp(torch.clamp((e2 / e1) * torch.log(s), -1000.0, 1000.0))
    term2 = spow(z_, 2.0 / e1)

    F[idx] = term1 + term2
    
    return F

def active_idx_toroid(points, theta, factor=2.0, pad=0.3):
    """
    Keep points near the ring tube:
      |r_xy(eω) - Rmaj| <= factor*a_r + pad   and   |z| <= factor*a_z + pad
    """
    th = theta.detach()
    R  = build_rotation_matrix(th[5:8])
    t  = th[8:11]
    pl = points @ R - t @ R

    e_omega = th[1].abs().clamp_min(0.05)
    Rmaj    = th[2].abs().clamp_min(1e-6)
    a_r     = th[3].abs().clamp_min(1e-6)
    a_z     = th[4].abs().clamp_min(1e-6)

    # Lp-radius in xy with p = 2/e_omega (when e_omega=1 -> Euclidean)
    p = 2.0 / e_omega
    r_xy = (pl[:,0].abs().pow(p) + pl[:,1].abs().pow(p)).pow(1.0/p)

    dr = (r_xy - Rmaj).abs()
    mr = factor * a_r + pad
    mz = factor * a_z + pad

    idx = torch.nonzero((dr <= mr) & (pl[:,2].abs() <= mz), as_tuple=False).squeeze(-1)
    return idx
  
def st_inside_near_only(points, theta, far_const=20.0):
    """
    Torus inside function G(ρ,z) in the tube cross-section:
      ρ = r_xy(eω) - Rmaj
      G = ( (|ρ|/a_r)^{2/eη} + (|z|/a_z)^{2/eη} )^{eη/2}
    G==1 on the surface, G<1 inside, G>1 outside.
    """
    idx = active_idx_toroid(points, theta, factor=2.0, pad=0.3)
    F = torch.full((points.shape[0],), float(far_const), device=points.device)
    if idx.numel() == 0:
        return F

    e_eta  = theta[0].abs().clamp_min(0.05)
    e_omega= theta[1].abs().clamp_min(0.05)
    Rmaj   = theta[2].abs().clamp_min(1e-6)
    a_r    = theta[3].abs().clamp_min(1e-6)
    a_z    = theta[4].abs().clamp_min(1e-6)

    R = build_rotation_matrix(theta[5:8])
    t = theta[8:11]
    pl = points[idx] @ R - t @ R

    p = 2.0 / e_omega
    r_xy = (pl[:,0].abs().pow(p) + pl[:,1].abs().pow(p)).pow(1.0/p)
    rho  = r_xy - Rmaj

    u = (rho.abs() / a_r).clamp_min(1e-12)
    v = (pl[:,2].abs() / a_z).clamp_min(1e-12)

    q = 2.0 / e_eta
    G = (u.pow(q) + v.pow(q)).clamp_min(1e-12).pow(1.0 / q)  # ==1 on surface

    F[idx] = G
    return F

def free_space_loss(ray_samples_flat, theta, b, alpha, number_of_rays):
    inside_score = sq_inside_near_only(ray_samples_flat, theta, b, alpha)
    soft_inside = torch.sigmoid(-(inside_score - 1.0) * 20.0)
    return soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()

def free_space_loss_toroid(ray_samples_flat, theta, number_of_rays, sharpness=20.0):
    inside_score = st_inside_near_only(ray_samples_flat, theta)  # 1 on surface
    soft_inside  = torch.sigmoid(-(inside_score - 1.0) * sharpness)
    return soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()
  
def compute_free_space(number_samples_per_ray, all_points, points, theta, b, alpha, shape):
  
    camera_position = torch.tensor([0.0, 0.0, 0.0], device=all_points.device)
    cluster_vecs = points - camera_position  # shape (N, 3)
    cluster_vecs = torch.nn.functional.normalize(cluster_vecs, dim=1)
    
    center_dir = torch.mean(cluster_vecs, dim=0)
    center_dir = center_dir / torch.norm(center_dir)
    
    cos_angles = (cluster_vecs @ center_dir)
    max_angle = torch.acos(torch.clamp(cos_angles.min(), -1.0, 1.0))  # in radians
    
    margin = 15 * torch.pi / 180  # radians
    final_cone_angle = max_angle + margin
    cos_thresh = torch.cos(final_cone_angle)    

    all_vecs = all_points - camera_position
    all_vecs = torch.nn.functional.normalize(all_vecs, dim=1)

    camera_origin = torch.zeros_like(all_points)
    directions = all_points - camera_origin  # or just points if origin is (0,0,0)

    t_vals = torch.linspace(0.5, 0.99, number_samples_per_ray, device=all_points.device)  
    ray_points = camera_origin[:, None, :] + t_vals[None, :, None] * directions[:, None, :]
    print("Number of rays:", len(directions))

    
    current_ray_samples_flat = ray_points.reshape(-1, 3)
    if shape == "superquadric":
        free = free_space_loss(current_ray_samples_flat, theta, b, alpha, directions.numel())
    elif shape == "supertoroid":
        free = free_space_loss_toroid(current_ray_samples_flat, theta, directions.numel())

    return free

import open3d as o3d
from pathlib import Path

def mesh_to_point_cloud(
    mesh_path,
    n_points=50000,
    method="poisson",             # "poisson" or "uniform"
    scale=1.0,                    # e.g., 0.001 if model is in mm and you want meters
    voxel_size=None,              # e.g., 0.002 for 2 mm downsample
    estimate_normals=True,
    noise_std=None,               # e.g., 0.0005 for 0.5 mm Gaussian noise
    out_path=None,                # save to .pcd or .ply if provided
    to_torch=False,               # return torch tensors too
):
    mesh_path = Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Could not read mesh: {mesh_path}")

    # Apply scale if needed (be mindful of YCB mesh units)
    if scale != 1.0:
        mesh.scale(scale, center=(0, 0, 0))

    mesh.compute_vertex_normals()

    # Sample points on the surface
    if method == "poisson":
        # Poisson-disk (blue noise) on surface
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:
        raise ValueError("method must be 'poisson' or 'uniform'")

    # Add small Gaussian noise if desired (to simulate sensor)
    if noise_std and noise_std > 0:
        pts = np.asarray(pcd.points)
        pts = pts + np.random.normal(scale=noise_std, size=pts.shape)
        pcd.points = o3d.utility.Vector3dVector(pts)

    # Estimate/Orient normals if desired
    if estimate_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.normalize_normals()

    # Voxel downsample for uniform density
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if estimate_normals:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            pcd.normalize_normals()

    # Save if requested
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Choose .pcd or .ply based on extension
        o3d.io.write_point_cloud(str(out_path), pcd)

    # Return Open3D and arrays (and torch if asked)
    pts_np = np.asarray(pcd.points)
    nrm_np = np.asarray(pcd.normals) if pcd.has_normals() else None

    if to_torch:
        import torch
        pts_t = torch.from_numpy(pts_np).float()
        nrm_t = torch.from_numpy(nrm_np).float() if nrm_np is not None else None
        return pcd, pts_np, nrm_np, pts_t, nrm_t

    return pcd, pts_np, nrm_np

import numpy as np

def quat_to_R(qx, qy, qz, qw):
    # normalize just in case
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:  # identity if degenerate
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)],
    ], dtype=np.float64)
    return R

def apply_pose_to_points_normals(pts_local, nrms_local, pose7):
    """
    pts_local: (N,3) local coords from the object model (your mesh_to_point_cloud output)
    nrms_local: (N,3) corresponding normals (can be None)
    pose7: [tx, ty, tz, qx, qy, qz, qw]
    returns: pts_world, nrms_world (normals normalized)
    """
    tx, ty, tz, qx, qy, qz, qw = [float(v) for v in pose7]
    R = quat_to_R(qx, qy, qz, qw)
    t = np.array([tx, ty, tz], dtype=np.float64)

    pts_world = pts_local @ R.T + t
    if nrms_local is not None:
        nrms_world = nrms_local @ R.T
        # (normals rotate only; no translation)
        nrms_world /= (np.linalg.norm(nrms_world, axis=1, keepdims=True) + 1e-12)
    else:
        nrms_world = None
    return pts_world, nrms_world



def _coerce_floats(seq):
    """Coerce list/tuple of numbers or numeric strings to float np.array."""
    return np.array([float(x) for x in seq], dtype=np.float64)

def _normalize_quat(q):
    """Normalize quaternion [qx,qy,qz,qw]; returns same shape."""
    n = np.linalg.norm(q)
    return q if n < 1e-12 else q / n

def load_scene_file(scene_json_path):
    """
    Load scene file and return a dict with:
      - 'obj_poses': {name: np.array([tx,ty,tz,qx,qy,qz,qw])}
      - 'raw': original JSON dict
    If 'obj_poses' missing, falls back to 'gz_obj_poses'.
    """
    path = Path(scene_json_path)
    with path.open("r") as f:
        raw = json.load(f)

    # prefer obj_poses; fallback to gz_obj_poses
    poses_src = None
    if "obj_poses" in raw and isinstance(raw["obj_poses"], dict):
        poses_src = raw["obj_poses"]
    elif "gz_obj_poses" in raw and isinstance(raw["gz_obj_poses"], dict):
        poses_src = raw["gz_obj_poses"]
    else:
        raise KeyError("No 'obj_poses' or 'gz_obj_poses' found in scene JSON.")

    obj_poses = {}
    for name, pose7 in poses_src.items():
        pose = _coerce_floats(pose7)
        if pose.shape != (7,):
            raise ValueError(f"Pose for '{name}' must have 7 values, got {pose.shape}.")
        # normalize quaternion
        t = pose[:3]
        q = _normalize_quat(pose[3:])
        obj_poses[name] = np.concatenate([t, q], axis=0)

    return {"obj_poses": obj_poses, "raw": raw}

def load_scene_file_cam(scene_json_path):
    """
    Load scene file and return a dict with:
      - 'obj_poses': {name: np.array([tx,ty,tz,qx,qy,qz,qw])}
      - 'raw': original JSON dict
    If 'obj_poses' missing, falls back to 'gz_obj_poses'.
    """
    path = Path(scene_json_path)
    with path.open("r") as f:
        raw = json.load(f)

    # prefer obj_poses; fallback to gz_obj_poses
    poses_src = None
    poses_src = raw["obj_poses_wrt_camera"]


    obj_poses = {}
    for name, pose7 in poses_src.items():
        pose = _coerce_floats(pose7)
        if pose.shape != (7,):
            raise ValueError(f"Pose for '{name}' must have 7 values, got {pose.shape}.")
        # normalize quaternion
        t = pose[:3]
        q = _normalize_quat(pose[3:])
        obj_poses[name] = np.concatenate([t, q], axis=0)

    return {"obj_poses_wrt_camera": obj_poses, "raw": raw}
  
def get_object_pose(scene_data, object_name):
    """
    scene_data: output of load_scene_file
    Returns np.array([tx,ty,tz,qx,qy,qz,qw]) for 'object_name'.
    """
    try:
        return scene_data["obj_poses"][object_name]
    except KeyError:
        avail = ", ".join(scene_data["obj_poses"].keys())
        raise KeyError(f"Object '{object_name}' not found. Available: {avail}")
      

def get_object_pose_cam(scene_data, object_name):
    """
    scene_data: output of load_scene_file
    Returns np.array([tx,ty,tz,qx,qy,qz,qw]) for 'object_name'.
    """
    try:
        return scene_data["obj_poses_wrt_camera"][object_name]
    except KeyError:
        avail = ", ".join(scene_data["obj_poses_wrt_camera"].keys())
        raise KeyError(f"Object '{object_name}' not found. Available: {avail}")

def sq_F(points, theta):
    R = build_rotation_matrix(theta[5:8])
    t = theta[8:11]
    pl = points @ R - t @ R

    a1 = theta[2].abs().clamp_min(1e-6)
    a2 = theta[3].abs().clamp_min(1e-6)
    a3 = theta[4].abs().clamp_min(1e-6)
    e1, e2 = theta[0], theta[1]

    def spow(u, p, eps=1e-8, umax=1e3, max_log=100.0):
        u = torch.clamp(u.abs(), min=eps, max=umax)
        return torch.exp(torch.clamp(p * torch.log(u), -max_log, max_log))

    x_ = pl[:,0]/a1; y_ = pl[:,1]/a2; z_ = pl[:,2]/a3
    px = spow(x_, 2.0/e2); py = spow(y_, 2.0/e2)
    s  = torch.clamp(px + py, min=1e-8)
    term1 = torch.exp(torch.clamp((e2/e1) * torch.log(s), -100.0, 100.0))
    term2 = spow(z_, 2.0/e1)
    return term1 + term2   # <1 inside, =1 surface, >1 outside



def table_transverse_loss(table_pts, theta, tol=0.005, reduction="mean"):
    # print("table transverse loss")
    # print("table_pts:", tuple(table_pts.shape))

    if table_pts.numel() == 0:
        return torch.tensor(0.0, device=theta.device)

    F = sq_F(table_pts, theta)
    
    inside = F < (1.0 - tol)
    vals   = (1.0 - F)[inside]

    # print("vals: ", vals)
    
    inside = F < (1.0 - tol)
    # print("table_pts1:", tuple(table_pts.shape))
    # print("min/max F1:", float(F.min()), float(F.max()))
    # print("inside_count1:", int(inside.sum()))
    
    if not inside.any():
        return torch.tensor(0.0, device=theta.device)
      


        # choose how you want to aggregate the violations:
    vals = (1.0 - F)[inside]                       # penetration amount in F-space
    if reduction == "mean":
        return vals.mean()
    elif reduction == "max":
        return vals.max()
    elif reduction == "sum":
        return vals.sum()
    else:
        raise ValueError("reduction must be 'mean'|'max'|'sum'")

def _plane_project_point(p, n, d):
    """
    Project point p onto plane n·x + d = 0.
    p: (3,), n: (3,) unit vector, d: scalar
    """
    return p - (torch.dot(n, p) + d) * n

def _plane_basis(n):
    """
    Given a unit normal n (3,), return two orthonormal in-plane axes (u,v).
    """
    # pick any vector not parallel to n
    a = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)
    if torch.allclose(torch.abs(torch.dot(a, n)), torch.tensor(1.0, device=n.device, dtype=n.dtype), atol=1e-4):
        a = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)
    u = F.normalize(torch.cross(n, a), dim=0)
    v = torch.cross(n, u)
    return u, v

def make_table_grid_points(theta,
                           plane_normal,      # torch (3,), not necessarily unit
                           plane_d,           # scalar (from plane model [a,b,c,d])
                           half_size=1.0,     # meters -> grid spans [-1, +1] in both axes
                           step=0.02,         # grid spacing in meters
                           offset_above=0.0,  # small lift along normal if you want (e.g., 0.005)
                           max_points=None):  # optional cap
    """
    Build a (2*half_size/step + 1)^2 grid on the plane, centered below the SQ.
    Returns a (M,3) torch tensor on the same device as theta.
    """
    with torch.no_grad():
        device = theta.device
        dtype = theta.dtype

        # unit normal
        n = F.normalize(plane_normal, dim=0)

        # project SQ center t onto the plane to define grid center
        t = theta[8:11]
        c = _plane_project_point(t, n, plane_d)

        # in-plane basis
        u, v = _plane_basis(n)

        # ranges
        # NOTE: use torch.arange on correct device/dtype to avoid host->device copies
        rng = torch.arange(-half_size, half_size + 1e-8, step, device=device, dtype=dtype)
        # meshgrid
        UU, VV = torch.meshgrid(rng, rng, indexing='ij')  # (K,K)
        # points = c + u*UU + v*VV  (+ small offset above plane if desired)
        pts = c[None, :] + UU.reshape(-1, 1) * u[None, :] + VV.reshape(-1, 1) * v[None, :]
        if offset_above != 0.0:
            pts = pts + (offset_above * n)[None, :]

        if max_points is not None and pts.shape[0] > max_points:
            idx = torch.randperm(pts.shape[0], device=device)[:max_points]
            pts = pts[idx]

    return pts.detach()


# --------- Inside/outside function for a supertoroid ----------
def st_F(points, theta):
    """
    F == 1 on the torus surface, <1 inside the tube, >1 outside.
    theta = [e_eta, e_omega, Rmaj, a_r, a_z, rz, ry, rx, tx, ty, tz]
    """
    # pose to local frame
    R = build_rotation_matrix(theta[5:8])
    t = theta[8:11]
    pl = points @ R - t @ R

    # parameters (clamped for stability)
    e_eta   = theta[0].abs().clamp_min(0.05)
    e_omega = theta[1].abs().clamp_min(0.05)
    Rmaj    = theta[2].abs().clamp_min(1e-6)
    a_r     = theta[3].abs().clamp_min(1e-6)
    a_z     = theta[4].abs().clamp_min(1e-6)
    
    # helper: safe power
    def spow(u, p, eps=1e-12):
        return u.abs().clamp_min(eps).pow(p)
    
    # L^p radius in xy with p = 2/e_omega  (when e_omega=1 -> Euclidean)
    p_xy = 2.0 / e_omega
    r_xy = spow(pl[:,0], p_xy) + spow(pl[:,1], p_xy)
    r_xy = r_xy.clamp_min(1e-12).pow(1.0 / p_xy)
    
    # tube cross-section coords (rho, z)
    rho = (r_xy - Rmaj)
    u = spow(rho / a_r, 2.0 / e_eta)
    v = spow(pl[:,2] / a_z, 2.0 / e_eta)
    
    # superellipse in the (rho,z) plane
    F = (u + v).clamp_min(1e-12).pow(e_eta / 2.0)
    return F

def table_transverse_loss_toroid(table_pts, theta, tol=0.005, reduction="mean"):
    """
    Penalizes table samples that lie strictly *inside* the torus tube (by tol margin).
    """
    if table_pts.numel() == 0:
        return torch.tensor(0.0, device=theta.device)

    F = st_F(table_pts, theta)          # <1 inside, =1 on surface
    inside = F < (1.0 - tol)
    if not inside.any():
        return torch.tensor(0.0, device=theta.device)

    vals = (1.0 - F[inside])            # penetration amount in F-space

    if reduction == "mean":
        return vals.mean()
    elif reduction == "max":
        return vals.max()
    elif reduction == "sum":
        return vals.sum()
    else:
        raise ValueError("reduction must be 'mean'|'max'|'sum'")

def sample_supertoroid_points_torch(
    x,                      # theta vector; layout matches your showSupertoroid()
    b=0.0,                  # optional bend amount (same convention as your code)
    alpha=0.0,              # bend axis/orientation param (your convention)
    threshold=1e-2,
    num_limit=10000,
    arclength=0.02,
    device='cuda'
):
    """
    Returns: torch.Tensor of shape (K,3) on `device` with sampled surface points.

    x layout (same as your showSupertoroid comment + implementation):
      x[0] = eps_eta   (tube/cross-section exponent)
      x[1] = eps_omega (ring exponent)
      x[2] = Rmaj?     (major ring radius)         <-- see note below
      x[3] = a_r?      (tube radius in r)
      x[4] = a_z       (tube radius in z)
      x[5:8] = Euler
      x[8:11] = translation

    NOTE: Your showSupertoroid() uses:
        a_r   = max(float(x[3]), 1e-4)
        Rmaj  = max(float(x[2]), a_r + 1e-4)
    even though the docstring says x[2]=a_r and x[3]=R. This sampler follows the
    IMPLEMENTATION (not the docstring) so it matches your plots exactly.
    """
    # --- copy & clamp like your plotting code ---
    x_np = np.array(x, dtype=float).copy()

    # exponents (avoid tiny values -> numerical issues)
    e_eta   = max(float(x_np[0]), 0.05)
    e_omega = max(float(x_np[1]), 0.05)

    # radii with same guard rails as showSupertoroid()
    a_r  = max(float(x_np[3]), 1e-4)         # tube radius in r (rho)
    Rmaj = max(float(x_np[2]), a_r + 1e-4)   # major ring radius; keep hole open
    a_z  = max(float(x_np[4]), 1e-4)         # tube radius in z

    # --- sample the two driving superellipses using your existing numpy samplers ---
    # cross-section (rho,z): [ a_r * cos^{e_eta}(η),  a_z * sin^{e_eta}(η) ]
    se_eta   = plot_functions.uniformSampledSuperellipse(e_eta,   [a_r, a_z],
                                                         threshold, num_limit, arclength)   # (2, Ne)
    # ring direction (unit): [ cos^{e_omega}(ω), sin^{e_omega}(ω) ]
    se_omega = plot_functions.uniformSampledSuperellipse(e_omega, [1.0, 1.0],
                                                         threshold, num_limit, arclength)   # (2, Nw)

    Ne = se_eta.shape[1]
    Nw = se_omega.shape[1]

    # --- vectorized surface construction (canonical frame) ---
    # ring unit superellipse (xy) -> shape (2, Nw, 1)
    xy_ring = se_omega[:, :, None]                               # (2, Nw, 1)

    # tube cross-section -> rho (radial in xy), z (vertical)
    rho = se_eta[0, :][None, None, :]                            # (1, 1, Ne)
    z   = se_eta[1, :][None, None, :]                            # (1, 1, Ne)

    # radial distance of the ring centerline plus tube offset
    radial = (Rmaj + rho)                                        # (1, 1, Ne)

    # x,y from ring * radial; broadcast to (2, Nw, Ne)
    xy = xy_ring * radial                                        # (2, Nw, Ne)
    # z broadcast to (1, Nw, Ne)
    z  = np.broadcast_to(z, (1, Nw, Ne))                         # (1, Nw, Ne)

    pts_local = np.concatenate([xy, z], axis=0)                  # (3, Nw, Ne)
    pts_local = pts_local.reshape(3, -1).T                       # (Nw*Ne, 3)

    # --- optional bend in local frame (reuse your numpy util) ---
    if b and b > 0.0:
        pts_local = plot_functions.bend_points_numpy(pts_local, b, alpha)

    # --- pose to world (reuse your numpy utils) ---
    RotM = build_rotation_matrix_numpy(x_np)               # (3, 3)
    t    = get_translation_numpy(x_np)                     # (3,)
    pts_world = pts_local @ RotM.T + t[None, :]

    # --- to torch ---
    pts_world_t = torch.from_numpy(np.ascontiguousarray(pts_world)).to(
        device=device, dtype=torch.float32
    )
    return pts_world_t

def sample_tapered_superparaboloid_points_torch(
    x,
    r_offset=0.01,
    b=0.0,
    alpha=0.0,
    threshold=1e-2,
    num_limit=10000,
    arclength=0.02,
    device='cuda'
):
    # --- params & clamps (mirror your plotting code) ---
    x_np = np.array(x, dtype=float).copy().reshape(-1)
    e1 = max(float(x_np[0]), 0.007)
    e2 = max(float(x_np[1]), 0.007)
    a1, a2, a3 = float(x_np[2]), float(x_np[3]), max(float(x_np[4]), 1e-6)

    # === ω sampling (unit superellipse) ===
    se_omega = plot_functions.uniformSampledSuperellipse(
        e2, [1.0, 1.0], threshold, num_limit, arclength
    )  # (2, Nw)
    if se_omega.ndim != 2 or se_omega.shape[0] != 2:
        raise ValueError(f"uniformSampledSuperellipse returned shape {se_omega.shape}, expected (2, Nw)")
    Nw = se_omega.shape[1]
    xy_omega = se_omega[:, :, None]                 # (2, Nw, 1)

    # === z sampling (arc-length along generating superparabola) ===
    dense_n = max(400, min(20000, int(np.ceil(a3 / max(arclength*0.25, 1e-4)))))
    z_dense = np.linspace(0.0, a3, dense_n, dtype=float)
    z_safe  = np.maximum(z_dense, 1e-12)
    dr_dz   = (1.0 / (e1 * a3)) * (z_safe / a3) ** (1.0 / e1 - 1.0)
    ds_dz   = np.sqrt(1.0 + dr_dz * dr_dz)
    s = np.zeros_like(z_dense)
    s[1:] = np.cumsum(0.5 * (ds_dz[1:] + ds_dz[:-1]) * (z_dense[1:] - z_dense[:-1]))
    S = float(s[-1])

    step = max(arclength, threshold)
    Nz = max(2, min(num_limit, int(np.ceil(S / step)) + 1))
    z_vals = np.interp(np.linspace(0.0, S, Nz, dtype=float), s, z_dense)  # (Nz,)
    r_vals = r_offset + (z_vals / a3) ** (1.0 / e1)                        # (Nz,)

    # --- build canonical points (vectorized) ---
    # scale_xy MUST be (2,1,1) — use reshape to avoid accidental extra axis
    scale_xy = np.array([a1, a2], dtype=float).reshape(2, 1, 1)   # (2,1,1)
    r_b = r_vals.reshape(1, 1, Nz)                                # (1,1,Nz)
    z_b = z_vals.reshape(1, 1, Nz)                                # (1,1,Nz)

    xy = (xy_omega * scale_xy) * r_b                              # (2, Nw, Nz)
    zc = np.broadcast_to(z_b, (1, Nw, Nz))                        # (1, Nw, Nz)

    # sanity checks to prevent the 4D concat bug
    assert xy.ndim == 3 and xy.shape[0] == 2, f"xy shape {xy.shape}"
    assert zc.ndim == 3 and zc.shape[0] == 1 and zc.shape[1] == Nw and zc.shape[2] == Nz, f"zc shape {zc.shape}"

    pts_local = np.concatenate([xy, zc], axis=0)                  # (3, Nw, Nz)
    pts_local = pts_local.reshape(3, -1).T                        # (Nw*Nz, 3)

    # optional bend in local frame
    if b and b > 0.0:
        pts_local = plot_functions.bend_points_numpy(pts_local, b, alpha)

    # pose to world
    RotM = build_rotation_matrix_numpy(x_np)
    t    = get_translation_numpy(x_np)
    pts_world = pts_local @ RotM.T + t[None, :]

    # to torch
    return torch.from_numpy(np.ascontiguousarray(pts_world)).to(device=device, dtype=torch.float32)

def estimate_normals_torch(points, k=20):
    """
    Estimate normals with torch GPU ops (brute-force kNN).
    """
    N = points.shape[0]
    dists = torch.cdist(points, points)  # (N,N)
    knn = dists.topk(k, largest=False).indices  # (N,k)

    normals = torch.zeros_like(points)
    for i in range(N):
        neigh = points[knn[i]]
        mean = neigh.mean(0, keepdim=True)
        X = neigh - mean
        cov = X.T @ X
        eigvals, eigvecs = torch.linalg.eigh(cov)
        n = eigvecs[:, 0]
        normals[i] = n / (n.norm() + 1e-12)

    return normals
  
def sample_superquadric_points_torch(
    x,                      # theta in your layout
    b=0.0,
    alpha=0.0,
    threshold=1e-2,
    num_limit=10000,
    arclength=0.02,
    device='cuda'
):
    """
    Returns:
        pts_world_t:    (K,3) torch.Tensor of sampled surface points (world frame)
        normals_world_t:(K,3) torch.Tensor of corresponding unit normals (world frame)
    """
    import numpy as np
    from math import isfinite

    # --- helper: safe |x|^p ---
    def pow_abs_safe(base, p, eps=1e-12):
        return np.power(np.clip(np.abs(base), eps, None), p)

    # --- helper: PCA normals (for bent geometry) ---
    def normals_pca(pts, k=20):
        # lazy import; sklearn is already in your environment elsewhere
        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as e:
            raise RuntimeError("normals_pca requires scikit-learn (NearestNeighbors).") from e
        n = pts.shape[0]
        k = min(k, max(3, n))  # at least 3
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(pts)
        idxs = nbrs.kneighbors(pts, return_distance=False)

        normals = np.zeros_like(pts)
        for i in range(n):
            X = pts[idxs[i]]
            Xc = X - X.mean(axis=0, keepdims=True)
            # smallest singular vector of covariance
            _, _, vh = np.linalg.svd(Xc, full_matrices=False)
            nrm = vh[-1]
            # normalize
            ln = np.linalg.norm(nrm)
            if ln > 0:
                nrm = nrm / ln
            normals[i] = nrm
        return normals

    # --- numpy path for your existing sampler ---
    x_np = np.array(x, dtype=float).copy()

    # avoid numerical instability in superellipse sampling (as in your code)
    if x_np[0] < 0.007: x_np[0] = 0.007
    if x_np[1] < 0.007: x_np[1] = 0.007

    # unpack parameters (adapt if your layout differs)
    a1, a2, a3 = x_np[0:3]
    eps1, eps2 = x_np[4:6]     # IMPORTANT: eps1, eps2 > 0

    # sample meridians/latitudes in local frame (your routines)
    point_eta   = plot_functions.uniformSampledSuperellipse(x_np[0], [1.0, x_np[4]], threshold, num_limit, arclength)
    point_omega = plot_functions.uniformSampledSuperellipse(x_np[1], [x_np[2], x_np[3]], threshold, num_limit, arclength)

    Ne = int(point_eta.shape[1])
    Nw = int(point_omega.shape[1])

    xy_scale = point_eta[0, :][None, :]      # (1, Ne)
    z_vals   = point_eta[1, :][None, :]      # (1, Ne)
    xy_omega = point_omega[:, :, None]       # (2, Nw, 1)

    xy = xy_omega * xy_scale                 # (2, Nw, Ne)
    z  = np.broadcast_to(z_vals, (1, Nw, Ne))
    pts_local = np.concatenate([xy, z], axis=0).reshape(3, -1).T  # (Nw*Ne, 3)

    # --- compute ANALYTIC normals in local frame BEFORE bending ---
    xL, yL, zL = pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]

    two_over_e1 = 2.0 / max(eps1, 1e-12)
    two_over_e2 = 2.0 / max(eps2, 1e-12)

    term_xy = pow_abs_safe(xL / a1, two_over_e2) + pow_abs_safe(yL / a2, two_over_e2)
    # exponents
    px  = two_over_e2 - 1.0
    py  = px
    pxy = (eps2 / max(eps1, 1e-12)) - 1.0
    pz  = two_over_e1 - 1.0

    # CORRECT gradient: includes 1/eps1 in all components
    gx = (2.0 / (eps1 * a1)) * np.sign(xL) * pow_abs_safe(xL / a1, px) * pow_abs_safe(term_xy, pxy)
    gy = (2.0 / (eps1 * a2)) * np.sign(yL) * pow_abs_safe(yL / a2, py) * pow_abs_safe(term_xy, pxy)
    gz = (2.0 / (eps1 * a3)) * np.sign(zL) * pow_abs_safe(zL / a3, pz)

    normals_local = np.stack([gx, gy, gz], axis=-1)
    nrm_len = np.linalg.norm(normals_local, axis=1, keepdims=True)
    normals_local = normals_local / (nrm_len + 1e-12)

    

    # recompute normals on the BENT surface with PCA (robust fallback)
    normals_local = normals_pca(pts_local, k=20)

    # --- to world frame ---
    RotM = build_rotation_matrix_numpy(x_np)  # (3,3)
    t    = get_translation_numpy(x_np)        # (3,)
    pts_world = pts_local @ RotM.T + t[None, :]
    normals_world = normals_local @ RotM.T
    normals_world /= (np.linalg.norm(normals_world, axis=1, keepdims=True) + 1e-12)

    # --- convert to torch ---
    pts_world_t     = torch.from_numpy(np.ascontiguousarray(pts_world)).to(device=device, dtype=torch.float32)
    normals_world_t = torch.from_numpy(np.ascontiguousarray(normals_world)).to(device=device, dtype=torch.float32)
    return pts_world_t, normals_world_t


# def sample_superquadric_points_torch(
#     x,                      # your theta in the SAME layout you already use in plot_functions/tools
#     b=0.0,
#     alpha=0.0,
#     threshold=1e-2,
#     num_limit=10000,
#     arclength=0.02,
#     device='cuda'
# ):
#     """
#     Returns: torch.Tensor of shape (K,3) on `device` with sampled surface points.
#     This mirrors your NumPy sampling (uniformSampledSuperellipse + optional bend + pose).
#     """
#     # --- use numpy for your existing sampler, then convert to torch ---
#     x_np = np.array(x, dtype=float).copy()

#     # avoid numerical instability in superellipse sampling (as in your code)
#     if x_np[0] < 0.007: x_np[0] = 0.007
#     if x_np[1] < 0.007: x_np[1] = 0.007

#     # sample superellipse along eta (vertical meridian) and omega (equatorial)
#     # NOTE: these scales/exponents match your snippet exactly
#     point_eta   = plot_functions.uniformSampledSuperellipse(x_np[0], [1.0, x_np[4]], threshold, num_limit, arclength)  # (2, Ne)
#     point_omega = plot_functions.uniformSampledSuperellipse(x_np[1], [x_np[2], x_np[3]], threshold, num_limit, arclength)  # (2, Nw)

#     Ne = point_eta.shape[1]
#     Nw = point_omega.shape[1]

#     # make all local points (before bend/pose): vectorized instead of double for-loop
#     # point_omega contributes (x,y) ring; point_eta contributes radial scale for XY and z for Z
#     xy_scale = point_eta[0, :][None, :]            # (1, Ne)
#     z_vals   = point_eta[1, :][None, :]            # (1, Ne)
#     xy_omega = point_omega[:, :, None]             # (2, Nw, 1)

#     xy = xy_omega * xy_scale                       # (2, Nw, Ne)
#     z  = np.broadcast_to(z_vals, (1, Nw, Ne))      # (1, Nw, Ne)

#     pts_local = np.concatenate([xy, z], axis=0)    # (3, Nw, Ne)
#     pts_local = pts_local.reshape(3, -1).T         # (Nw*Ne, 3)

    
#     # optional bend (your bending is in local frame)
#     if b > 0.0:
#         pts_local = plot_functions.bend_points_numpy(pts_local, b, alpha)

#     # pose to world
#     RotM = build_rotation_matrix_numpy(x_np)         # (3,3)
#     t    = get_translation_numpy(x_np)               # (3,)
#     pts_world = pts_local @ RotM.T + t[None, :]

#     # convert to torch on the requested device
#     pts_world_t = torch.from_numpy(np.ascontiguousarray(pts_world)).to(device=device, dtype=torch.float32)
#     return pts_world_t


# def nearest_on_superquadric(points_world: torch.Tensor,
#                             samples_world: torch.Tensor,
#                             points_block: int = 50_000,
#                             samples_block: int = 10_000,
#                             compute_precision: str = "fp32",   # "fp32", "fp64", "fp16"
#                             return_squared: bool = False,
#                             dedup_samples: bool = False):
#     """
#     Memory-safe nearest neighbor with controllable precision.

#     - compute_precision="fp32" (default) is robust and fast on GPU.
#     - Use "fp64" once to verify; if results match fp32, you're good.
#     - Avoid "fp16" unless you must save memory (it can produce extra zeros).

#     Returns:
#       dmin : (N,) nearest Euclidean distance (or squared if return_squared=True)
#       idx  : (N,) argmin indices into *original* samples_world
#       qmin : (N,3) nearest sampled points
#     """
#     # sanitize shapes
#     P = points_world.reshape(-1, 3).contiguous()
#     S = samples_world.reshape(-1, 3).contiguous()
#     if S.numel() == 0:
#         raise ValueError("nearest_on_superquadric: samples_world is empty.")

#     device = P.device
#     # outputs remain fp32; internals use compute_dtype
#     dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}
#     compute_dtype = dtype_map.get(compute_precision.lower(), torch.float32)

#     # optional dedup to reduce duplicates causing zero distances
#     if dedup_samples:
#         # round to 1e-6 grid in world units before unique, to be stable
#         S_rounded = (S.to(torch.float64) * 1e6).round().to(torch.int64)
#         S_key = S_rounded.view(-1, 3)
#         uniq, inv = torch.unique(S_key, dim=0, return_inverse=True)
#         S = S[uniq.new_empty(uniq.size(0), dtype=torch.long).scatter_(0, torch.arange(uniq.size(0), device=uniq.device), torch.arange(uniq.size(0), device=uniq.device))]
#         # map back indices after selection: inv gives, for each original row, the index in uniq
#         # We’ll rebuild a mapping from dedup’d index to first occurrence in original:
#         # (simple approach) get first index for each unique key:
#         first_idx = torch.full((uniq.size(0),), -1, dtype=torch.long, device=device)
#         for i, u in enumerate(inv):
#             if first_idx[u] == -1:
#                 first_idx[u] = i
#         samples_to_original = first_idx
#     else:
#         samples_to_original = torch.arange(S.shape[0], device=device, dtype=torch.long)

#     # move to device/dtype
#     P = P.to(device=device, dtype=compute_dtype)
#     S = S.to(device=device, dtype=compute_dtype)

#     N, K = P.shape[0], S.shape[0]

#     # outputs in fp32
#     dmin = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
#     imin = torch.full((N,), -1, device=device, dtype=torch.long)

#     # tiled computation of squared distances: ||p||^2 + ||s||^2 - 2 p·s
#     for p0 in range(0, N, points_block):
#         p1 = min(p0 + points_block, N)
#         P_blk = P[p0:p1]                                    # (Bp,3)
#         pp = (P_blk * P_blk).sum(dim=1)                     # (Bp,)

#         best_d2 = torch.full((p1 - p0,), float("inf"), device=device, dtype=torch.float64)
#         best_j  = torch.full((p1 - p0,), -1, device=device, dtype=torch.long)

#         # compute in compute_dtype but accumulate/compare in fp64 to be safe
#         pp64 = pp.to(torch.float64)

#         for s0 in range(0, K, samples_block):
#             s1 = min(s0 + samples_block, K)
#             S_blk = S[s0:s1]                                # (Bs,3)
#             ss = (S_blk * S_blk).sum(dim=1)                 # (Bs,)
#             dot = P_blk @ S_blk.t()                         # (Bp,Bs)

#             # cast to fp64 before combining to minimize cancellation
#             d2 = (pp64[:, None] +
#                   ss[None, :].to(torch.float64) -
#                   2.0 * dot.to(torch.float64))              # (Bp,Bs)

#             tile_min, tile_arg = d2.min(dim=1)              # (Bp,), (Bp,)
#             mask = tile_min < best_d2
#             best_d2[mask] = tile_min[mask]
#             best_j[mask]  = tile_arg[mask] + s0

#         # guard tiny negative due to round-off
#         best_d2 = torch.clamp(best_d2, min=0.0)

#         if return_squared:
#             dmin[p0:p1] = best_d2.to(torch.float32)
#         else:
#             dmin[p0:p1] = torch.sqrt(best_d2).to(torch.float32)
#         imin[p0:p1] = samples_to_original[best_j]

#     qmin = samples_world.reshape(-1, 3)[imin.to(samples_world.device)]

#     return dmin, imin, qmin

def nearest_on_superquadric(
    points_world: torch.Tensor,
    samples_world: torch.Tensor,
    samples_normals_world: torch.Tensor,   # (K,3)
    points_block: int = 50_000,
    samples_block: int = 10_000,
    compute_precision: str = "fp32",   # "fp32", "fp64", "fp16"
    return_squared: bool = False,
    dedup_samples: bool = False,
):
    """
    Memory-safe nearest neighbor using normal-projected (unsigned) distances.

    Returns:
      dmin : (N,) unsigned distance along surface normal  |(p - q)·n|
      idx  : (N,) indices into samples_world
      qmin : (N,3) nearest sampled points
      n_q  : (N,3) corresponding surface normals
    """
    # --- sanitize inputs ---
    P = points_world.reshape(-1, 3).contiguous()
    S = samples_world.reshape(-1, 3).contiguous()
    Nrm = samples_normals_world.reshape(-1, 3).contiguous()
    if S.numel() == 0:
        raise ValueError("nearest_on_superquadric: samples_world is empty.")
    assert S.shape[0] == Nrm.shape[0], "Normals and samples must have same length."

    device = P.device
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}
    compute_dtype = dtype_map.get(compute_precision.lower(), torch.float32)

    # optional dedup
    if dedup_samples:
        S_rounded = (S.to(torch.float64) * 1e6).round().to(torch.int64)
        uniq, inv = torch.unique(S_rounded, dim=0, return_inverse=True)
        S = S[uniq]
        Nrm = Nrm[uniq]
        samples_to_original = torch.arange(S.shape[0], device=device, dtype=torch.long)
    else:
        samples_to_original = torch.arange(S.shape[0], device=device, dtype=torch.long)

    P = P.to(device=device, dtype=compute_dtype)
    S = S.to(device=device, dtype=compute_dtype)
    Nrm = Nrm.to(device=device, dtype=compute_dtype)

    N, K = P.shape[0], S.shape[0]

    # --- initialize outputs ---
    dmin = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
    imin = torch.full((N,), -1, device=device, dtype=torch.long)

    # --- blockwise NN search ---
    for p0 in range(0, N, points_block):
        p1 = min(p0 + points_block, N)
        P_blk = P[p0:p1]
        pp = (P_blk * P_blk).sum(dim=1)

        best_d2 = torch.full((p1 - p0,), float("inf"), device=device, dtype=torch.float64)
        best_j  = torch.full((p1 - p0,), -1, device=device, dtype=torch.long)
        pp64 = pp.to(torch.float64)

        for s0 in range(0, K, samples_block):
            s1 = min(s0 + samples_block, K)
            S_blk = S[s0:s1]
            ss = (S_blk * S_blk).sum(dim=1)
            dot = P_blk @ S_blk.t()

            d2 = (pp64[:, None] + ss[None, :].to(torch.float64) - 2.0 * dot.to(torch.float64))
            tile_min, tile_arg = d2.min(dim=1)
            mask = tile_min < best_d2
            best_d2[mask] = tile_min[mask]
            best_j[mask]  = tile_arg[mask] + s0

        best_d2 = torch.clamp(best_d2, min=0.0)
        if return_squared:
            dmin[p0:p1] = best_d2.to(torch.float32)
        else:
            dmin[p0:p1] = torch.sqrt(best_d2).to(torch.float32)
        imin[p0:p1] = samples_to_original[best_j]

    # --- gather results ---
    qmin = samples_world[imin.to(samples_world.device)]
    n_q  = samples_normals_world[imin.to(samples_world.device)]

    # --- compute unsigned normal-projected distance |(p - q)·n| ---
    error_vec = points_world - qmin
    d_signed = torch.sum(error_vec * n_q, dim=1)
    d_abs = torch.abs(d_signed).to(torch.float32)

    return d_abs, imin, qmin, n_q

def nearest_on_superquadric_normalproj(
    points_world: torch.Tensor,          # (N,3)
    samples_world: torch.Tensor,         # (K,3)
    samples_normals_world: torch.Tensor, # (K,3) unit normals
    points_block: int = 50_000,
    samples_block: int = 10_000,
    compute_precision: str = "fp32",
    dedup_samples: bool = False,
    topk_preselect: int = 0,             # opcional: 0 = sin preselección; >0 = refina con top-k euclídeo
):
    """
    Devuelve:
      d_abs : (N,)  = min_s | (p - s)·n_s |   (distancia proyectada sobre la normal, NO firmada)
      idx   : (N,)  índice del sample que minimiza la proyección
      qmin  : (N,3) sample ganador
      n_q   : (N,3) normal del sample ganador
    """
    # --- sanity ---
    P = points_world.reshape(-1,3).contiguous()
    S = samples_world.reshape(-1,3).contiguous()
    Nrm = samples_normals_world.reshape(-1,3).contiguous()
    if S.numel() == 0:
        raise ValueError("nearest_on_superquadric_normalproj: samples_world is empty.")
    assert S.shape[0] == Nrm.shape[0], "samples and normals must match in length."

    device = P.device
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}
    compute_dtype = dtype_map.get(compute_precision.lower(), torch.float32)

    # opcional: deduplicar samples (reduce puntos coincidentes que pueden dar proyecciones 0 repetidas)
    if dedup_samples:
        key = (S.to(torch.float64) * 1e6).round().to(torch.int64)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        S = S[uniq]
        Nrm = Nrm[uniq]

    P = P.to(device=device, dtype=compute_dtype)
    S = S.to(device=device, dtype=compute_dtype)
    Nrm = Nrm.to(device=device, dtype=compute_dtype)

    N, K = P.shape[0], S.shape[0]

    # salidas (fp32)
    d_abs = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
    imin  = torch.full((N,), -1, device=device, dtype=torch.long)

    # precompute p·(·) on the fly por bloques (no cabe todo si K es grande)
    # si activas preselección, primero top-k en distancia euclídea para recortar columnas
    use_topk = int(topk_preselect) > 0

    for p0 in range(0, N, points_block):
        p1 = min(p0 + points_block, N)
        P_blk = P[p0:p1]                    # (Bp,3)

        if not use_topk:
            # Full projected distance tile: | P_blk @ Nrm^T  -  (S·Nrm)[None,:] |
            for s0 in range(0, K, samples_block):
                s1 = min(s0 + samples_block, K)
                N_blk = Nrm[s0:s1]                          # (Bs,3)
                S_blk = S[s0:s1]                            # (Bs,3)

                # p·n  and  s·n
                p_dot_n = P_blk @ N_blk.t()                 # (Bp,Bs)
                s_dot_n = (S_blk * N_blk).sum(dim=1)        # (Bs,)
                proj = (p_dot_n - s_dot_n.unsqueeze(0)).abs().to(torch.float64)

                tile_min, tile_arg = proj.min(dim=1)        # (Bp,), (Bp,)
                # actualizar mejores
                better = tile_min < d_abs[p0:p1].to(torch.float64)
                d_abs[p0:p1][better] = tile_min[better].to(torch.float32)
                imin[p0:p1][better]  = (tile_arg[better] + s0)
        else:
            # 1) top-k euclídeo para candidatos por punto
            #    Nota: hacemos esto en tiles de samples también para memoria controlada
            k = min(topk_preselect, K)
            # inicializa buffers de top-k (valores grandes)
            top_vals = torch.full((p1-p0, k), float("inf"), device=device, dtype=torch.float64)
            top_idx  = torch.full((p1-p0, k), -1, device=device, dtype=torch.long)

            pp64 = (P_blk*P_blk).sum(dim=1).to(torch.float64)  # (Bp,)

            for s0 in range(0, K, samples_block):
                s1 = min(s0 + samples_block, K)
                S_blk = S[s0:s1]
                ss64 = (S_blk*S_blk).sum(dim=1).to(torch.float64)        # (Bs,)
                dot = (P_blk @ S_blk.t()).to(torch.float64)              # (Bp,Bs)
                d2 = pp64[:,None] + ss64[None,:] - 2.0*dot               # (Bp,Bs)
                # concat con lo acumulado y quedarnos con top-k mínimos
                cand = torch.cat([top_vals, d2], dim=1)                  # (Bp, k+Bs)
                idxs = torch.cat([top_idx, torch.arange(s0, s1, device=device).repeat(P_blk.size(0),1)], dim=1)
                top_vals, top_pos = torch.topk(cand.neg(), k, dim=1)     # neg para topk => mínimos
                top_vals = top_vals.neg()
                top_idx = idxs.gather(1, top_pos)

            # 2) ahora, para cada punto, evaluamos proyección solo en sus k candidatos
            #    (convertimos a matriz bloque por eficiencia)
            # Construye Nrm y S correspondientes
            unique_cols = torch.unique(top_idx.clamp_min(0))
            N_cand = Nrm[unique_cols]                       # (Kc,3)
            S_cand = S[unique_cols]                         # (Kc,3)

            # p·n  and  s·n para candidatos
            p_dot_n = P_blk @ N_cand.t()                    # (Bp,Kc)
            s_dot_n = (S_cand * N_cand).sum(dim=1)          # (Kc,)
            proj_all = (p_dot_n - s_dot_n.unsqueeze(0)).abs().to(torch.float64)  # (Bp,Kc)

            # Para cada fila, de entre sus candidatos (indices en unique_cols), coger el mínimo
            # map each row's candidate set to column positions in unique_cols
            # construimos una máscara/índices
            # (solución simple: para cada fila, tomamos el mínimo sobre TODA Kc pero
            #  restringimos a sus k candidatos poniendo +inf en el resto)
            mask_inf = torch.full_like(proj_all, float("inf"))
            # crea un dict col->pos para mapear rápido
            col2pos = {int(c.item()): j for j,c in enumerate(unique_cols)}
            rows = torch.arange(P_blk.size(0), device=device)
            for r in range(P_blk.size(0)):
                cols_r = top_idx[r]
                pos_r = torch.tensor([col2pos[int(c.item())] for c in cols_r if c >= 0], device=device, dtype=torch.long)
                mask_inf[r, pos_r] = proj_all[r, pos_r]

            tile_min, tile_arg = mask_inf.min(dim=1)
            better = tile_min < d_abs[p0:p1].to(torch.float64)
            d_abs[p0:p1][better] = tile_min[better].to(torch.float32)
            imin[p0:p1][better]  = unique_cols[tile_arg[better]]

    # recoge q y n
    qmin = samples_world[imin.to(samples_world.device)]
    n_q  = samples_normals_world[imin.to(samples_world.device)]

    return d_abs, imin, qmin, n_q

def nearest_on_superquadric_mixed(
    points_world: torch.Tensor,          # (N,3)
    samples_world: torch.Tensor,         # (K,3)
    samples_normals_world: torch.Tensor, # (K,3) unit normals, 1:1 with samples_world
    points_block: int = 50_000,
    samples_block: int = 10_000,
    compute_precision: str = "fp32",
    dedup_samples: bool = False,
    topk_preselect: int = 32,            # Euclidean preselection per point (set 0 to disable)
    mu: float = 0.1,                     # tangential penalty
):
    """
    Returns:
      d_mix : (N,)  mixed distance sqrt(d_perp^2 + mu * d_para^2)
      idx   : (N,)  argmin index into samples_world
      qmin  : (N,3) matched sample point
      n_q   : (N,3) matched sample normal (unit)
      debug : dict  sanity stats
    """
    # --- sanitize ---
    P = points_world.reshape(-1,3).contiguous()
    S = samples_world.reshape(-1,3).contiguous()
    Nrm = samples_normals_world.reshape(-1,3).contiguous()
    if S.numel() == 0:
        raise ValueError("nearest_on_superquadric_mixed: empty samples.")
    assert S.shape[0] == Nrm.shape[0], "samples and normals must have same length."

    device = P.device
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}
    compute_dtype = dtype_map.get(compute_precision.lower(), torch.float32)

    if dedup_samples:
        key = (S.to(torch.float64) * 1e6).round().to(torch.int64)
        uniq, inv = torch.unique(key, dim=0, return_inverse=False, sorted=True)
        S = S[uniq]
        Nrm = Nrm[uniq]

    P = P.to(device=device, dtype=compute_dtype)
    S = S.to(device=device, dtype=compute_dtype)
    Nrm = Nrm.to(device=device, dtype=compute_dtype)

    # sanity: normals unit?
    nlen = torch.linalg.norm(Nrm, dim=1)
    # if you want, enforce normalization
    Nrm = Nrm / (nlen.clamp_min(1e-12).unsqueeze(1))

    N, K = P.shape[0], S.shape[0]
    d_best = torch.full((N,), float("inf"), device=device, dtype=torch.float32)
    i_best = torch.full((N,), -1, device=device, dtype=torch.long)

    # optional Euclidean top-k preselection (strongly recommended for speed+robustness)
    if topk_preselect and topk_preselect > 0:
        k = min(int(topk_preselect), K)
        # compute Euclidean NN via tiling to get candidate indices
        top_vals = torch.full((N, k), float("inf"), device=device, dtype=torch.float64)
        top_idx  = torch.full((N, k), -1, device=device, dtype=torch.long)
        for p0 in range(0, N, points_block):
            p1 = min(p0 + points_block, N)
            P_blk = P[p0:p1]
            pp = (P_blk*P_blk).sum(dim=1).to(torch.float64)
            best_vals = torch.full((p1-p0, k), float("inf"), device=device, dtype=torch.float64)
            best_idx  = torch.full((p1-p0, k), -1, device=device, dtype=torch.long)
            for s0 in range(0, K, samples_block):
                s1 = min(s0 + samples_block, K)
                S_blk = S[s0:s1]
                ss = (S_blk*S_blk).sum(dim=1).to(torch.float64)
                dot = (P_blk @ S_blk.t()).to(torch.float64)
                d2  = pp[:,None] + ss[None,:] - 2.0*dot          # (Bp, Bs)
                # merge into running top-k minima
                cand = torch.cat([best_vals, d2], dim=1)          # (Bp, k+Bs)
                idxs = torch.cat([
                    best_idx,
                    torch.arange(s0, s1, device=device).repeat(P_blk.size(0),1)
                ], dim=1)
                # topk over negative => mins
                vals, pos = torch.topk(-cand, k, dim=1)
                best_vals = -vals
                best_idx  = idxs.gather(1, pos)
            top_vals[p0:p1] = best_vals
            top_idx[p0:p1]  = best_idx

        # evaluate mixed distance only on candidates
        unique_cols = torch.unique(top_idx.clamp_min(0))
        S_c   = S[unique_cols]           # (Kc,3)
        Nrm_c = Nrm[unique_cols]         # (Kc,3)

        # precompute s·n for candidates
        s_dot_n = (S_c * Nrm_c).sum(dim=1)                   # (Kc,)
        for p0 in range(0, N, points_block):
            p1 = min(p0 + points_block, N)
            P_blk = P[p0:p1]                                  # (Bp,3)
            p_dot_n = P_blk @ Nrm_c.t()                       # (Bp,Kc)
            # normal distance
            d_perp = (p_dot_n - s_dot_n.unsqueeze(0)).abs()   # (Bp,Kc)
            # tangential component against each candidate normal
            # (p - s) part: do it by tiles to save mem
            Bp, Kc = p_dot_n.shape
            d_best_blk = torch.full((Bp,), float("inf"), device=device, dtype=torch.float32)
            i_best_blk = torch.full((Bp,), -1, device=device, dtype=torch.long)
            # tile candidates if needed
            tile = 4096
            for c0 in range(0, Kc, tile):
                c1 = min(c0+tile, Kc)
                Sc = S_c[c0:c1]                                # (Ct,3)
                Nc = Nrm_c[c0:c1]                              # (Ct,3)
                # (p - s)
                diff = P_blk.unsqueeze(1) - Sc.unsqueeze(0)    # (Bp,Ct,3)
                # d_perp on same slice
                dperp_slice = d_perp[:, c0:c1].unsqueeze(-1)   # (Bp,Ct,1)
                # parallel component
                para = dperp_slice * Nc.unsqueeze(0)           # (Bp,Ct,3)
                # tangential component length
                dpara = torch.linalg.norm(diff - para, dim=-1) # (Bp,Ct)
                # mixed distance
                dmix = torch.sqrt(dperp_slice.squeeze(-1)**2 + mu * dpara**2).to(torch.float32)  # (Bp,Ct)
                # keep minima
                val, arg = dmix.min(dim=1)                     # (Bp,)
                better = val < d_best_blk
                d_best_blk[better] = val[better]
                i_best_blk[better] = (c0 + arg[better])

            # map local candidate column to original index
            col2orig = unique_cols[i_best_blk.clamp_min(0)]
            d_best[p0:p1] = d_best_blk
            i_best[p0:p1] = col2orig

    else:
        # No preselection: full tiles
        for p0 in range(0, N, points_block):
            p1 = min(p0 + points_block, N)
            P_blk = P[p0:p1]
            # we’ll accumulate best over sample tiles
            best_val = torch.full((p1-p0,), float("inf"), device=device, dtype=torch.float32)
            best_j   = torch.full((p1-p0,), -1, device=device, dtype=torch.long)
            for s0 in range(0, K, samples_block):
                s1 = min(s0 + samples_block, K)
                S_blk   = S[s0:s1]            # (Bs,3)
                N_blk   = Nrm[s0:s1]          # (Bs,3)
                s_dot_n = (S_blk*N_blk).sum(dim=1)                    # (Bs,)
                p_dot_n = (P_blk @ N_blk.t())                          # (Bp,Bs)
                d_perp  = (p_dot_n - s_dot_n.unsqueeze(0)).abs()       # (Bp,Bs)

                # compute tangential part
                diff  = P_blk.unsqueeze(1) - S_blk.unsqueeze(0)        # (Bp,Bs,3)
                para  = (d_perp.unsqueeze(-1)) * N_blk.unsqueeze(0)    # (Bp,Bs,3)
                dpara = torch.linalg.norm(diff - para, dim=-1)         # (Bp,Bs)

                dmix  = torch.sqrt(d_perp**2 + mu * dpara**2).to(torch.float32)  # (Bp,Bs)
                val, arg = dmix.min(dim=1)
                better = val < best_val
                best_val[better] = val[better]
                best_j[better]   = arg[better] + s0
            d_best[p0:p1] = best_val
            i_best[p0:p1] = best_j

    qmin = samples_world[i_best.to(samples_world.device)]
    n_q  = samples_normals_world[i_best.to(samples_world.device)]

    debug = {
        "max_normal_dev": float((torch.linalg.norm(nlen - 1.0).abs().max()).cpu()),
        "units_tip": "Distances are in the same units as input (meters). Check your scales."
    }
    return d_best, i_best, qmin, n_q, debug

# --- implicit F and its gradient (for normals if you want them) ---
def F_superellipsoid_local(x_local, a, eps1, eps2, eps=1e-12):
    ax, ay, az = a
    e1, e2 = eps1, eps2
    X = torch.pow(torch.clamp(torch.abs(x_local[:,0]/ax), min=eps), 2.0/e2)
    Y = torch.pow(torch.clamp(torch.abs(x_local[:,1]/ay), min=eps), 2.0/e2)
    Z = torch.pow(torch.clamp(torch.abs(x_local[:,2]/az), min=eps), 2.0/e1)
    return torch.pow(torch.clamp(X+Y, min=eps), e2/e1) + Z

# --- SE(3) transforms ---
def to_local(points_world, R, t):
    # if R maps local->world, local = (points_world - t) @ R.T
    return (points_world - t) @ R.T

# --- utilities to unpack theta ---
def _Rx(a):
    ca, sa = torch.cos(a), torch.sin(a)
    R = torch.zeros((3,3), dtype=a.dtype, device=a.device)
    R[0,0] = 1
    R[1,1] =  ca; R[1,2] = -sa
    R[2,1] =  sa; R[2,2] =  ca
    return R

def _Ry(a):
    ca, sa = torch.cos(a), torch.sin(a)
    R = torch.zeros((3,3), dtype=a.dtype, device=a.device)
    R[1,1] = 1
    R[0,0] =  ca; R[0,2] =  sa
    R[2,0] = -sa; R[2,2] =  ca
    return R

def _Rz(a):
    ca, sa = torch.cos(a), torch.sin(a)
    R = torch.zeros((3,3), dtype=a.dtype, device=a.device)
    R[2,2] = 1
    R[0,0] =  ca; R[0,1] = -sa
    R[1,0] =  sa; R[1,1] =  ca
    return R

def unpack_sq(theta_tensor: torch.Tensor):
    """
    theta: shape (11,) on the correct device/dtype
      [tx, ty, tz, rx, ry, rz, ax, ay, az, eps1, eps2]
    Returns:
      R  : (3,3) rotation matrix, local->world
      t  : (3,)  translation (world)
      a  : (3,)  semi-axes (positive)
      e1 : ()    epsilon1
      e2 : ()    epsilon2
    """
    assert theta_tensor.numel() == 11 and theta_tensor.ndim == 1, "theta must be a (11,) tensor"
    tx, ty, tz, rx, ry, rz, ax, ay, az, e1, e2 = theta_tensor

    # Rotation (Euler XYZ: apply Rx, then Ry, then Rz) -> R = Rz @ Ry @ Rx
    Rx = _Rx(rx)
    Ry = _Ry(ry)
    Rz = _Rz(rz)
    R  = Rz @ Ry @ Rx   # local -> world

    # Translation
    t = torch.stack([tx, ty, tz], dim=0)

    # Scales and exponents (clamp to avoid degeneracy)
    a  = torch.stack([ax, ay, az], dim=0)
    a  = torch.clamp(a, min=1e-6)
    e1 = torch.clamp(e1, min=1e-6)
    e2 = torch.clamp(e2, min=1e-6)

    return R, t, a, e1, e2
  
@torch.no_grad()
# def select_outside_nearest_from_samples(
#     D: torch.Tensor,               # (N,M)
#     Q: torch.Tensor,               # (N,M,3)
#     thetas_sq: list[torch.Tensor], # len M
# ):
#     """
#     For each point:
#       - ranks shapes by D
#       - picks first whose projection Q[n,m] is NOT inside any other shape (F<0)
#       - fallback to nearest if all are buried
#     """
#     device = D.device
#     dtype  = D.dtype
#     N, M = D.shape

#     # pre-decode params
#     R_list, t_list, a_list, e_list = [], [], [], []
#     for th in thetas_sq:
#         th = torch.as_tensor(th, dtype=dtype, device=device)
#         R, t, a, e1, e2 = unpack_sq(th)     # ensure consistent unpack
#         R_list.append(R); t_list.append(t); a_list.append(a); e_list.append((e1, e2))

#     # check "inside any" correctly
#     inside_any = torch.zeros((N, M), device=device, dtype=torch.bool)
#     Q_flat = Q.reshape(-1, 3)

#     for j in range(M):
#         Rj, tj, aj = R_list[j], t_list[j], a_list[j]
#         e1j, e2j = e_list[j]
#         # transform all candidates to local frame of shape j
#         Qloc = to_local(Q_flat, Rj, tj).reshape(N, M, 3)
#         Fj = F_superellipsoid_local(Qloc.reshape(-1,3), aj, e1j, e2j).reshape(N, M)
#         # Correct test: inside ⇔ F < 0
#         inside_j = Fj < 1.0
#         inside_j[:, j] = False
#         inside_any |= inside_j  # accumulate
#     # ---------------------------------------------

#     order = torch.argsort(D, dim=1)  # ascending
#     chosen_m = torch.full((N,), -1, device=device, dtype=torch.long)
#     chosen_d = torch.full((N,), float('inf'), device=device, dtype=dtype)
#     chosen_q = torch.zeros((N,3), device=device, dtype=dtype)
#     unresolved = torch.ones((N,), device=device, dtype=torch.bool)

#     for r in range(M):
#         if not unresolved.any():
#             break
#         cand_m = order[:, r]
#         idx_rows = unresolved.nonzero(as_tuple=False).squeeze(1)
#         cand_m_r = cand_m[idx_rows]
#         good = ~inside_any[idx_rows, cand_m_r]
#         if good.any():
#             rows_ok = idx_rows[good]
#             ms_ok   = cand_m_r[good]
#             chosen_m[rows_ok] = ms_ok
#             chosen_d[rows_ok] = D[rows_ok, ms_ok]
#             chosen_q[rows_ok] = Q[rows_ok, ms_ok]
#             unresolved[rows_ok] = False

#     # fallback (if all buried)
#     if unresolved.any():
#         rows = unresolved.nonzero(as_tuple=False).squeeze(1)
#         m0 = order[rows, 0]
#         chosen_m[rows] = m0
#         chosen_d[rows] = D[rows, m0]
#         chosen_q[rows] = Q[rows, m0]
#     return chosen_d, chosen_q, chosen_m

def select_outside_nearest_from_samples(
    D: torch.Tensor,               # (N,M) distances per point per shape (nearest sampled within that shape)
    Q: torch.Tensor,               # (N,M,3) corresponding nearest sampled points (world coords)
    thetas_sq: list[torch.Tensor], # length M, each (11,) on proper device/dtype
):
    """
    For each point n:
      - sort shapes by D[n, :]
      - pick the first shape whose Q[n, m, :] is NOT inside any other superquadric
      - if all candidates are 'inside', fall back to the nearest (rank 0)
    Returns:
      chosen_d : (N,)   distance
      chosen_q : (N,3)  sampled point (world)
      chosen_m : (N,)   chosen shape index (0..M-1)
    """
    device = D.device
    dtype  = D.dtype
    N, M = D.shape

    # --- precompute per-shape transforms/params
    R_list, t_list, a_list, e_list = [], [], [], []
    for th in thetas_sq:
        th = torch.as_tensor(th, dtype=dtype, device=device)
        R, t, a, e1, e2 = unpack_sq(th)
        R_list.append(R); t_list.append(t); a_list.append(a); e_list.append((e1, e2))

    # --- compute "inside any other shape" mask for each candidate Q[:, m, :]
    # inside_any[n, m] == True  iff Q[n,m] is inside *some* shape j != m
    inside_any = torch.zeros((N, M), device=device, dtype=torch.bool)

    # Flatten Q once for reuse
    Q_flat = Q.reshape(-1, 3)  # (N*M, 3)

    # For each shape j, test all Q[n,m] against it
    for j in range(M):
        Rj, tj, aj = R_list[j], t_list[j], a_list[j]
        e1j, e2j = e_list[j]
        # transform to local of shape j
        Qloc = to_local(Q_flat, Rj, tj).reshape(N, M, 3)
        # implicit F_j; inside if F < 1
        Fj = F_superellipsoid_local(Qloc.reshape(-1, 3), aj, e1j, e2j).reshape(N, M)
        inside_j = Fj < 1.0  # (N, M)

        # do NOT count being inside itself: zero-out column m=j
        inside_j[:, j] = False

        # accumulate OR over j
        inside_any |= inside_j

    # --- per-point ranked selection (small loop over ranks)
    order = torch.argsort(D, dim=1)  # (N, M) ascending by distance per row

    chosen_m = torch.full((N,), -1, device=device, dtype=torch.long)
    chosen_d = torch.zeros((N,), device=device, dtype=dtype)
    chosen_q = torch.zeros((N,3), device=device, dtype=dtype)

    unresolved = torch.ones((N,), device=device, dtype=torch.bool)

    for r in range(M):
        if not unresolved.any():
            break
        cand_m = order[:, r]             # (N,)
        idx_rows = torch.nonzero(unresolved, as_tuple=False).squeeze(1)
        cand_m_r = cand_m[idx_rows]      # candidates for unresolved rows

        good = ~inside_any[idx_rows, cand_m_r]  # pick those whose Q is not inside others
        if good.any():
            rows_ok = idx_rows[good]
            ms_ok   = cand_m_r[good]
            chosen_m[rows_ok] = ms_ok
            chosen_d[rows_ok] = D[rows_ok, ms_ok]
            chosen_q[rows_ok] = Q[rows_ok, ms_ok]
            unresolved[rows_ok] = False

    # Fallback: if some points were inside for all shapes, assign nearest (rank 0)
    if unresolved.any():
        rows = torch.nonzero(unresolved, as_tuple=False).squeeze(1)
        m0 = order[rows, 0]
        chosen_m[rows] = m0
        chosen_d[rows] = D[rows, m0]
        chosen_q[rows] = Q[rows, m0]
        unresolved[rows] = False

    return chosen_d, chosen_q, chosen_m

@torch.no_grad()
def table_violation_fraction(table_pts, theta, tol=0.005):
    """Fracción de puntos de mesa que violan (F < 1 - tol)."""
    if table_pts.numel() == 0:
        return 0.0
    F = sq_F(table_pts, theta)                      # F==1 en superficie
    frac = (F < (1.0 - tol)).float().mean()
    return float(frac)

def table_violation_metrics(table_pts, theta, tol=0.005):
    """
    Métricas de violación de mesa:
      - mean_F / max_F : penetración en espacio F (adimensional)
      - mean_m / max_m : penetración aproximada en metros (SDF ≈ |1-F|/||∇F||)
      - frac_viol      : fracción de la malla de mesa que viola
      - mean_n         : penetración proyectada a lo largo de la normal de la mesa (si la pasas)
    """
    device = theta.device
    if table_pts.numel() == 0:
        return 0.0

    # 1) Selección de puntos que violan
    with torch.no_grad():
        F_all = sq_F(table_pts, theta)              # (M,)
        mask = F_all < (1.0 - tol)
        if not mask.any():
            return 0.0
    
    # 2) Volvemos a calcular F sobre los que violan y pedimos gradiente wrt puntos
    pts = table_pts[mask].detach().clone().requires_grad_(True)  # sólo los que violan
    F = sq_F(pts, theta.detach())                                # no metemos θ en el grafo
    # ||∇F|| en esos puntos
    g = torch.autograd.grad(F.sum(), pts, create_graph=False, retain_graph=False)[0]
    gradnorm = g.norm(dim=1).clamp_min(1e-6)

    pen_F = (1.0 - F)                   # cuánto “entra” en espacio-F
    pen_m = pen_F / gradnorm            # metros aprox (SDF)


    mean_m = float(pen_m.mean())
    
    print("mean_m:", mean_m)

    return mean_m