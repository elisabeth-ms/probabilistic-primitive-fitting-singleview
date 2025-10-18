from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import open3d as o3d
from mayavi import mlab
import cv2
from sklearn.cluster import DBSCAN
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import re, datetime as dt
import yaml
from pathlib import Path
from typing import Dict, Any, List, Union
import re, yaml, numpy as np
import torch

def _next_test_num(scene_dir: Path) -> int:
    mx = 0
    pat = re.compile(r"^test(\d+)$")
    for p in scene_dir.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m: mx = max(mx, int(m.group(1)))
    return mx + 1
  
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

# --- Intrinsics (yours) ---
FX = 554.254691191187
FY = 554.254691191187
CX = 320.5
CY = 240.5

def label_points_from_seg(points_np, labels, fx=FX, fy=FY, cx=CX, cy=CY):
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

def dtheta(theta, arclength, threshold, scale, epsilon):
    # calculation the sampling step size
    if theta < threshold:
        dt = np.abs(np.power(arclength / scale[1] +np.power(theta, epsilon), \
             (1 / epsilon)) - theta)
    else:
        dt = arclength / epsilon * ((np.cos(theta) ** 2 * np.sin(theta) ** 2) /
             (scale[0] ** 2 * np.cos(theta) ** (2 * epsilon) * np.sin(theta) ** 4 +
             scale[1] ** 2 * np.sin(theta) ** (2 * epsilon) * np.cos(theta) ** 4)) ** (1 / 2)
    
    return dt
  
def angle2points(theta, scale, epsilon):

    point = np.zeros((2, np.shape(theta)[0]))
    point[0] = scale[0] * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** epsilon
    point[1] = scale[1] * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** epsilon

    return point

def uniformSampledSuperellipse(epsilon, scale, threshold = 1e-2, num_limit = 10000, arclength = 0.02):

    # initialize array storing sampled theta
    theta = np.zeros(num_limit)
    theta[0] = 0

    for i in range(num_limit):
        dt = dtheta(theta[i], arclength, threshold, scale, epsilon)
        theta_temp = theta[i] + dt

        if theta_temp > np.pi / 4:
            theta[i + 1] = np.pi / 4
            break
        else:
            if i + 1 < num_limit:
                theta[i + 1] = theta_temp
            else:
                raise Exception(
                'Number of the sampled points exceed the preset limit', \
                num_limit,
                'Please decrease the sampling arclength.'
                )
    critical = i + 1

    for j in range(critical + 1, num_limit):
        dt = dtheta(theta[j], arclength, threshold, np.flip(scale), epsilon)
        theta_temp = theta[j] + dt
        
        if theta_temp > np.pi / 4:
            break
        else:
            if j + 1 < num_limit:
                theta[j + 1] = theta_temp
            else:
                raise Exception(
                'Number of the sampled points exceed the preset limit', \
                num_limit,
                'Please decrease the sampling arclength.'
                )
    num_pt = j
    theta = theta[0 : num_pt + 1]

    point_fw = angle2points(theta[0 : critical + 1], scale, epsilon)
    point_bw = np.flip(angle2points(theta[critical + 1: num_pt + 1], np.flip(scale), epsilon), (0, 1))
    point = np.concatenate((point_fw, point_bw), 1)
    point = np.concatenate((point, np.flip(point[:, 0 : num_pt], 1) * np.array([[-1], [1]]), 
                           point[:, 1 : num_pt + 1] * np.array([[-1], [-1]]),
                           np.flip(point[:, 0 : num_pt], 1) * np.array([[1], [-1]])), 1)

    return point

def build_rotation_matrix_numpy(theta_np):
    """
    theta_np: numpy array of shape (11,)
    returns: 3x3 rotation matrix
    """
    rz, ry, rx = theta_np[5:8]
    rot = R.from_euler('ZYX', [rz, ry, rx])  # careful with order!!
    return rot.as_matrix()

def bend_points_numpy(P, b, alpha):
    """
    P: (..., 3) points in the local (unposed) frame
    b: bending parameter (>0)
    alpha: bending direction in xy (radians)
    """
    # guardrails
    b = max(float(b), 1e-6)

    x, y, z = P[..., 0], P[..., 1], P[..., 2]

    rho = np.sqrt(x*x + y*y)                     # radial distance in xy
    ang = np.arctan2(y, x)                       # angle of (x,y)
    r   = np.cos(alpha - ang) * rho              # projected radius along alpha
    gamma = b * z

    invb = 1.0 / b
    R = invb - (invb - r) * np.cos(gamma)

    dx = (R - r) * np.cos(alpha)
    dy = (R - r) * np.sin(alpha)
    dz = (invb - r) * np.sin(gamma)

    out = np.empty_like(P)
    out[..., 0] = x + dx
    out[..., 1] = y + dy
    out[..., 2] = dz
    return out

def get_translation_numpy(theta_np):
    """
    theta_np: numpy array of shape (11,)
    returns: (3,) translation vector
    """
    return theta_np[8:11]
  
def showSuperquadrics(x,b=0, alpha=0, threshold = 1e-2, num_limit = 10000, arclength = 0.02):
    print("inside showsuperquadrics b:", b, " alpha: ", alpha)
    # avoid numerical instability in sampling
    if x[0] < 0.007:
        x[0] = 0.007
    if x[1] < 0.007:
        x[1] = 0.007
    # sampling points in superellipse    
    point_eta = uniformSampledSuperellipse(x[0], [1, x[4]], threshold, num_limit, arclength)
    point_omega = uniformSampledSuperellipse(x[1], [x[2], x[3]], threshold, num_limit, arclength)
    
    # preallocate meshgrid
    x_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
    y_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
    z_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
    RotM = build_rotation_matrix_numpy(x)
    print("RotM: ", RotM)
    for m in range(np.shape(point_omega)[1]):
        for n in range(np.shape(point_eta)[1]):
            point_temp = np.zeros(3)
            point_temp[0 : 2] = point_omega[:, m] * point_eta[0, n]
            point_temp[2] = point_eta[1, n]
            
            # # >>> NEW: bend in local frame <<<
            if b > 0.0:
                point_temp = bend_points_numpy(point_temp[None, :], b, alpha)[0]

            # then pose
            point_temp = point_temp@RotM.T + get_translation_numpy(x)

            x_mesh[m, n] = point_temp[0]
            y_mesh[m, n] = point_temp[1]
            z_mesh[m, n] = point_temp[2]
    
    # mlab.view(azimuth=0.0, elevation=0.0, distance=2)
    mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0.894, 0.447, 0.0), opacity=0.8)

def showSupertoroid(x, threshold=1e-2, num_limit=10000, arclength=0.02, color=(0.894, 0.447, 0.0)):
    """
    x layout (mirrors your superellipsoid param vector):
      x[0]=eps_eta (tube exponent), x[1]=eps_omega (ring exponent),
      x[2]=a_r (tube radius in r), x[3]=R (major ring radius), x[4]=a_z (tube radius in z),
      x[5:8]=Euler, x[8:11]=translation.
    """
    # clamps for stability
    e_eta   = max(float(x[0]), 0.05)
    e_omega = max(float(x[1]), 0.05)
    a_r     = max(float(x[3]), 1e-4)
    Rmaj       = max(float(x[2]), a_r + 1e-4)  # keep hole open / avoid self-intersection
    a_z     = max(float(x[4]), 1e-4)

    # superellipse samples:
    # - cross-section (ρ,z) in the rz-plane:  [ a_r*cos^e_eta(η),  a_z*sin^e_eta(η) ]
    se_eta   = uniformSampledSuperellipse(e_eta,   [a_r, a_z], threshold, num_limit, arclength)
    # - ring direction, unit superellipse on xy-plane: [ cos^e_omega(ω), sin^e_omega(ω) ]
    se_omega = uniformSampledSuperellipse(e_omega, [1.0, 1.0], threshold, num_limit, arclength)

    M, N = se_omega.shape[1], se_eta.shape[1]
    x_mesh = np.empty((M, N), dtype=float)
    y_mesh = np.empty((M, N), dtype=float)
    z_mesh = np.empty((M, N), dtype=float)

    RotM = build_rotation_matrix_numpy(x)      # your function
    t    = get_translation_numpy(x)            # your function

    for m in range(M):
        c2 = se_omega[0, m]   # cos^{e_omega}(ω)
        s2 = se_omega[1, m]   # sin^{e_omega}(ω)
        for n in range(N):
            rho = se_eta[0, n]  # a_r * cos^{e_eta}(η)
            z   = se_eta[1, n]  # a_z * sin^{e_eta}(η)

            # canonical -> world (same as your superellipsoid):
            pt_local  = np.array([(Rmaj + rho) * c2, (Rmaj + rho) * s2, z], dtype=float)
            pt_world  = pt_local @ RotM.T + t

            x_mesh[m, n], y_mesh[m, n], z_mesh[m, n] = pt_world

    mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=0.8)

def showTaperedSuperparaboloidWithBase(x, r_offset=0.01, threshold=1e-2, num_limit=10000, arclength=0.02):
    import numpy as np
    from mayavi import mlab

    x = np.asarray(x).flatten()

    # Avoid numerical issues
    x[0] = max(x[0], 0.007)
    x[1] = max(x[1], 0.007)

    e1, e2 = x[0], x[1]
    a1, a2, a3 = x[2], x[3], x[4]
    rot = build_rotation_matrix_numpy(x)
    trans = get_translation_numpy(x)

    # Create grid
    num_z = 100
    num_omega = 100
    z_vals = np.linspace(0, a3, num_z)
    omega_vals = np.linspace(0, 2 * np.pi, num_omega)

    x_mesh = np.zeros((num_omega, num_z))
    y_mesh = np.zeros((num_omega, num_z))
    z_mesh = np.zeros((num_omega, num_z))

    for j, z in enumerate(z_vals):
        r = r_offset + (z / a3) ** (1 / e1)

        for i, omega in enumerate(omega_vals):
            cos_e = np.sign(np.cos(omega)) * np.abs(np.cos(omega)) ** e2
            sin_e = np.sign(np.sin(omega)) * np.abs(np.sin(omega)) ** e2

            px = a1 * r * cos_e
            py = a2 * r * sin_e
            pz = z

            point = np.array([px, py, pz]) @ rot.T + trans
            x_mesh[i, j] = point[0]
            y_mesh[i, j] = point[1]
            z_mesh[i, j] = point[2]

    mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0.894, 0.447, 0.0), opacity=0.8)

def showPoints(points, scale_factor=0.1, color =(1, 0, 0), figure=None):
    if figure is None:
        figure = mlab.gcf()
    return mlab.points3d(points[:,0], points[:,1], points[:,2],
                         scale_factor=scale_factor, color=color, figure=figure)

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
    which: Union[int, str] = "latest",
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
  
scene_ = "scene_39"


base_path = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/results/check"

scene_dir = Path(base_path) / scene_


point_cloud = read_with_open3d("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/pcds/"+scene_+"/cloud.pcd")
point_cloud = remove_close_points(point_cloud, 0.003)
filtered_points, plane_points, plane_model = remove_largest_plane(point_cloud, distance_threshold=0.003)
filtered_points, plane_points1, plane_model1 = remove_largest_plane(filtered_points, distance_threshold=0.003)
point_cloud = filter_by_z(point_cloud, -np.inf, 1.5)

labels, id2color = load_seg_as_labels("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/segmasks/"+scene_+"/gtseg_ord-nearest_first_step-0.png", inflate_px=10)
point_labels = label_points_from_seg(filtered_points, labels)
clusters = split_points_by_label(filtered_points, point_labels)

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
showPoints(filtered_points, scale_factor=0.0025)
mlab.show()
all_params_modeled = {}
idx = 0

tau_10 = 0.010
tau_5 = 0.005

for lid, cluster in clusters.items():
    all_dists = []
    c = read_cluster_yaml(base_path, scene_, cluster_id=lid, which="latest")
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
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    mlab.view(azimuth=108.51, elevation=168.97, distance=0.5805, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.864720847838999), roll=-177.93)
    showPoints(point_cloud, scale_factor=0.0025, color=(0.702, 0.702, 0.702), figure=fig)
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
            showSuperquadrics(theta,b, alpha)
            points = torch.tensor(cluster, dtype=torch.float32, device='cuda')
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device='cuda')
            distances = sq_distances(points, theta_tensor).unsqueeze(1)
            all_dists.append(distances)
        elif shape_type == "supertoroid":
            showSupertoroid(theta)
            points = torch.tensor(cluster, dtype=torch.float32, device='cuda')
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device='cuda')
            distances = st_distances(points, theta_tensor).unsqueeze(1)
            all_dists.append(distances)
        else:
            showTaperedSuperparaboloidWithBase(theta,k)
            points = torch.tensor(cluster, dtype=torch.float32, device='cuda')
            theta_tensor = torch.tensor(theta, dtype=torch.float32, device='cuda')
            k_tensor = torch.tensor(k, dtype=torch.float32, device='cuda')
            distances = spb_distances_autograd(points, theta_tensor, k_tensor).unsqueeze(1)
            all_dists.append(distances)
            

        # print(f"    indices_good (n={indices_good.size}): {indices_good.tolist()}")
        # print(f"    indices_bad (n={indices_bad.size}): {indices_bad.tolist()}")
        # print(f"    indices_remaining (n={indices_remaining.size}): {indices_remaining.tolist()}")
        showPoints(cluster, scale_factor=0.0025, color=(1, 0, 0.0), figure=fig)
        print(len(cluster))
        
        # Stack and take min
        d_all = torch.cat(all_dists, dim=1)  # (N, M)
        d_min, _ = torch.min(d_all, dim=1)   # (N,)
        
        coverage_5 =  (d_min < tau_5).float().mean()
        coverage_10 =  (d_min < tau_10).float().mean()

        rmse = torch.sqrt(torch.mean(d_min**2)).item()
        mae = torch.mean(torch.abs(d_min)).item()
        
        print("rmse: ", rmse)
        print("mae: ", mae)
        print("coverage5mm: ", coverage_5)
        print("coverage5+10mm: ", coverage_10)

    mlab.show()