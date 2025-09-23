import torch
import plyfile
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from mayavi import mlab
from scipy.spatial import cKDTree
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import cv2
# from pytorch3d.structures import Pointclouds
# from pytorch3d.ops import estimate_pointcloud_normals

import collections
import matplotlib.pyplot as plt
from collections import deque

def _auto_eps(points, k=8, scale=2.0):
    """
    Heuristic eps from median k-NN distance.
    """
    if len(points) < k+1:
        return 0.02  # small default
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)  # includes distance to self at [:,0]==0
    knn = dists[:, -1]                    # kth neighbor
    return float(np.median(knn) * scale)

def split_by_distance(points, eps=None, min_samples=20):
    """
    Returns a list of (Mi,3) arrays (one per connected blob).
    """
    if len(points) == 0:
        return []
    if eps is None:
        eps = _auto_eps(points)  # auto-tune per residual
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    uniq = [l for l in np.unique(labels) if l != -1]
    if not uniq:
        return [points]  # nothing separable
    return [points[labels == l] for l in uniq]

def make_hist():
    return collections.defaultdict(list)

@torch.no_grad()
def snap_theta(hist, theta, total_loss=None, fit_loss=None, free=None, sigma2=None):
    # theta: (11,) con [e1,e2,a1,a2,a3, rz,ry,rx, tx,ty,tz]
    hist['e1'].append(theta[0].item())
    hist['e2'].append(theta[1].item())
    hist['a1'].append(theta[2].item())
    hist['a2'].append(theta[3].item())
    hist['a3'].append(theta[4].item())
    if total_loss is not None:   hist['total_loss'].append(total_loss.item() if torch.is_tensor(total_loss) else float(total_loss))
    if free is not None:   hist['free'].append(free.item() if torch.is_tensor(free) else float(free))
    if fit_loss is not None:   hist['fit_loss'].append(fit_loss.item() if torch.is_tensor(fit_loss) else float(fit_loss))
    if sigma2 is not None: hist['sigma2'].append(float(sigma2))



def showPoints(point, scale_factor=0.1, color =(1, 0, 0)):
    
    mlab.view(azimuth=0.0, elevation=0.0, distance=2)
    mlab.points3d(point[:, 0], point[:, 1], point[:, 2], scale_factor=scale_factor, color=color)

import numpy as np
from scipy.spatial.transform import Rotation as R

def build_rotation_matrix_numpy(theta_np):
    """
    theta_np: numpy array of shape (11,)
    returns: 3x3 rotation matrix
    """
    rz, ry, rx = theta_np[5:8]
    rot = R.from_euler('ZYX', [rz, ry, rx])  # careful with order!!
    return rot.as_matrix()

def get_translation_numpy(theta_np):
    """
    theta_np: numpy array of shape (11,)
    returns: (3,) translation vector
    """
    return theta_np[8:11]

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

    mlab.mesh(x_mesh, y_mesh, z_mesh, color=(1.0, 0.2, 0.2), opacity=0.8)
    mlab.view(azimuth=0.0, elevation=0.0, distance=2)


def showTaperedSuperparaboloid(x, kx=0.0, ky=0.0, threshold=1e-2, num_limit=10000, arclength=0.02):
    import numpy as np
    from mayavi import mlab

    x = np.asarray(x).flatten()

    if x[0] < 0.007:
        x[0] = 0.007
    if x[1] < 0.007:
        x[1] = 0.007

    e1, e2 = x[0], x[1]
    a1, a2, a3 = x[2], x[3], x[4]
    rot = build_rotation_matrix_numpy(x)
    trans = get_translation_numpy(x)

    # Create parameter grids
    num_z = 100
    num_omega = 100
    z_vals = np.linspace(0, a3, num_z)
    omega_vals = np.linspace(0, 2 * np.pi, num_omega)

    x_mesh = np.zeros((num_omega, num_z))
    y_mesh = np.zeros((num_omega, num_z))
    z_mesh = np.zeros((num_omega, num_z))

    for j, z in enumerate(z_vals):
        r = (z / a3) ** (1 / e1)

        taper_x = 1.0 + kx * z
        taper_y = 1.0 + ky * z
        taper_x = max(taper_x, 1e-3)  # ensure positive
        taper_y = max(taper_y, 1e-3)

        for i, omega in enumerate(omega_vals):
            cos_e = np.sign(np.cos(omega)) * (np.abs(np.cos(omega)) ** e2)
            sin_e = np.sign(np.sin(omega)) * (np.abs(np.sin(omega)) ** e2)

            px = a1 * r * cos_e * taper_x
            py = a2 * r * sin_e * taper_y
            pz = z

            point = np.array([px, py, pz]) @ rot.T + trans
            x_mesh[i, j] = point[0]
            y_mesh[i, j] = point[1]
            z_mesh[i, j] = point[2]

    mlab.mesh(x_mesh, y_mesh, z_mesh, color=(1.0, 0.0, 0.0), opacity=0.8)
    mlab.view(azimuth=0.0, elevation=0.0, distance=2)


def showSuperparaboloid(x, threshold=1e-2, num_limit=10000, arclength=0.02):
    import numpy as np
    from mayavi import mlab

    x = np.asarray(x).flatten()

    if x[0] < 0.007:
        x[0] = 0.007
    if x[1] < 0.007:
        x[1] = 0.007

    e1, e2 = x[0], x[1]
    a1, a2, a3 = x[2], x[3], x[4]
    rot = build_rotation_matrix_numpy(x)
    trans = get_translation_numpy(x)

    # Create parameter grids
    num_z = 100
    num_omega = 100
    z_vals = np.linspace(0, a3, num_z)
    omega_vals = np.linspace(0, 2 * np.pi, num_omega)

    # Preallocate mesh arrays
    x_mesh = np.zeros((num_omega, num_z))
    y_mesh = np.zeros((num_omega, num_z))
    z_mesh = np.zeros((num_omega, num_z))

    # Build the mesh
    for j, z in enumerate(z_vals):
        r = (z / a3) ** (1 / e1)  # radial component based on height
        # r = (abs(z)/a3)**(1/e1)
        for i, omega in enumerate(omega_vals):
            cos_e = np.sign(np.cos(omega)) * (np.abs(np.cos(omega)) ** e2)
            sin_e = np.sign(np.sin(omega)) * (np.abs(np.sin(omega)) ** e2)

            px = a1 * r * cos_e
            py = a2 * r * sin_e
            pz = z

            point = np.array([px, py, pz]) @ rot.T + trans

            x_mesh[i, j] = point[0]
            y_mesh[i, j] = point[1]
            z_mesh[i, j] = point[2]

    # Plot
    mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0.2, 0.4, 1.0), opacity=0.8)
    mlab.view(azimuth=0.0, elevation=0.0, distance=2)
    
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
            
            # >>> NEW: bend in local frame <<<
            if b > 0.0:
                point_temp = bend_points_numpy(point_temp[None, :], b, alpha)[0]

            # then pose
            point_temp = point_temp@RotM.T + get_translation_numpy(x)

            x_mesh[m, n] = point_temp[0]
            y_mesh[m, n] = point_temp[1]
            z_mesh[m, n] = point_temp[2]
    
    mlab.view(azimuth=0.0, elevation=0.0, distance=2)
    mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0, 0, 1), opacity=0.8)



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


def read_ply(path_to_file):
    # read points from a .ply file and store in an nparray
    plydata = plyfile.PlyData.read(path_to_file)
    pc = plydata['vertex'].data
    return np.array([[x, y, z] for x, y, z in pc])


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

def read_with_open3d(path: str) -> np.ndarray:
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts

  
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

# def build_rotation_matrix(euler_angles):
#     """
#     euler_angles: tensor of shape (3,) [rx, ry, rz] (rotation around x, y, z)
#     returns: tensor of shape (3,3) representing the rotation matrix
#     """
#     rz, ry, rx = euler_angles

#     # Compute individual rotation matrices
#     cosx = torch.cos(rx)
#     sinx = torch.sin(rx)
#     cosy = torch.cos(ry)
#     siny = torch.sin(ry)
#     cosz = torch.cos(rz)
#     sinz = torch.sin(rz)

#     # Rotation around x-axis
#     Rx = torch.tensor([[1, 0, 0],
#                        [0, cosx, -sinx],
#                        [0, sinx, cosx]], device=euler_angles.device)

#     # Rotation around y-axis
#     Ry = torch.tensor([[cosy, 0, siny],
#                        [0, 1, 0],
#                        [-siny, 0, cosy]], device=euler_angles.device)

#     # Rotation around z-axis
#     Rz = torch.tensor([[cosz, -sinz, 0],
#                        [sinz, cosz, 0],
#                        [0, 0, 1]], device=euler_angles.device)

#     # Combined rotation: R = Rz @ Ry @ Rx
#     R = Rz @ Ry @ Rx

#     return R

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

# def rotation_matrix_to_euler(R):
#     """
#     R: tensor of shape (3,3) rotation matrix
#     returns: tensor of shape (3,) representing euler angles (rx, ry, rz)
#     """
#     sy = torch.sqrt(R[0, 0]**2 + R[1, 0]**2)

#     singular = sy < 1e-6

#     if not singular:
#         rx = torch.atan2(R[2, 1], R[2, 2])
#         ry = torch.atan2(-R[2, 0], sy)
#         rz = torch.atan2(R[1, 0], R[0, 0])
#     else:
#         rx = torch.atan2(-R[1, 2], R[1, 1])
#         ry = torch.atan2(-R[2, 0], sy)
#         rz = torch.tensor(0.0, device=R.device)

#     return torch.stack([rx, ry, rz])
def rotation_matrix_to_euler(R):
    """
    Converts a rotation matrix to Euler angles in 'ZYX' order (yaw, pitch, roll).
    
    Args:
        R: torch.Tensor of shape (3, 3) - rotation matrix
    Returns:
        torch.Tensor of shape (3,) - Euler angles (rz, ry, rx)
    """
    # Compute sy = sqrt(R00^2 + R10^2)
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        rz = torch.atan2(R[1, 0], R[0, 0])  # yaw
        ry = torch.atan2(-R[2, 0], sy)       # pitch
        rx = torch.atan2(R[2, 1], R[2, 2])   # roll
    else:
        # Gimbal lock case
        rz = torch.atan2(-R[1, 2], R[1, 1])
        ry = torch.atan2(-R[2, 0], sy)
        rx = torch.tensor(0.0, device=R.device)

    return torch.stack([rz, ry, rx])

def BoundVolume(points):
    V = (torch.max(points[:, 0]) - torch.min(points[:, 0])) * \
        (torch.max(points[:, 1]) - torch.min(points[:, 1])) * \
        (torch.max(points[:, 2]) - torch.min(points[:, 2]))
    return V

def initialize_theta_superparaboloids_pytorch(points, table_normal, rescale=False, init_theta=None):
    """
    points: Tensor of shape (N, 3), already on CUDA
    returns: initialized theta (11,) as nn.Parameter
    """
    device = points.device
    table_normal = F.normalize(table_normal, dim=0)  # ensure unit vector
    
    
    center = points.mean(dim=0)

    # Project points onto the normal direction
    projections = (points-center) @ table_normal  # (N,)
    min_proj = projections.max()

    # Translation needed to shift the lowest point to zero
    offset = min_proj * table_normal

    t0 = center + offset  
    points_centered = points - t0
    
    print("t0: ", t0)

    # 2. Rescale
    if rescale:
        max_length = torch.max(points_centered.abs())
        scale = max_length / 10.0
        points_centered = points_centered / scale
    else:
        scale = 1.0


    # 2. Use table normal as Z axis
    z_axis = -table_normal  # open *away* from the table
    arbitrary = torch.tensor([1.0, 0.0, 0.0], device=device)
    if torch.allclose(z_axis, arbitrary):
        arbitrary = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    x_axis = F.normalize(torch.cross(arbitrary, z_axis), dim=0)
    y_axis = torch.cross(z_axis, x_axis)

    R0 = torch.stack([x_axis, y_axis, z_axis], dim=1)
    import math
    angle = math.radians(8)  # 30° → radians

    device = R0.device
    dtype = R0.dtype

    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle),  math.cos(angle)]
    ], device=device, dtype=dtype)
    
    Ry90 = torch.tensor([
    [0.0, 0.0,  1.0],
    [0.0, 1.0,  0.0],
    [-1.0, 0.0, 0.0]
    ], device=R0.device, dtype=R0.dtype)

    # Apply rotation
    R0_rotated = Rx@R0   # or Rx90 @ R0 depending on convention

    print("R0: ", R0_rotated)
    # print("R0: ", R0)
    # 5. Rotate points
    points_rot0 = points_centered @ R0_rotated

    V = BoundVolume(points_rot0)
    
    print("BoundVolume: ", V)
    
    p0= 1/V
    
    sigma2 = V ** (1 / 3) / 10.0

    # # 6. Estimate semi-axes
    # s0 = torch.median(points_rot0.abs(), dim=0).values

    # # 7. Initial parameters
    e1 = torch.tensor(1.5, device=device)
    e2 = torch.tensor(1.5, device=device)
    # a1, a2, a3 = s0[0], s0[1], s0[2]
    
    a3 = a3 = points_rot0[:, 2].max()*3/4
    radial = torch.norm(points_centered[:, :2], dim=1)
    a1a2 = torch.median(radial).repeat(2)
    a1, a2 = a1a2[0]/2, a1a2[1]/2


    print(a1, a2, a3)
    # 8. Get initial rotation as Euler angles
    
    if init_theta is not None:
      euler_angles = torch.stack([
          torch.as_tensor(init_theta[5], dtype=torch.float32, device=device),
          torch.as_tensor(init_theta[6], dtype=torch.float32, device=device),
          torch.as_tensor(init_theta[7], dtype=torch.float32, device=device)
      ])
      translation = torch.zeros(3, device=device)

    else:
        euler_angles = rotation_matrix_to_euler(R0_rotated)
        translation = torch.zeros(3, device=device)


    print("euler angles: ", euler_angles)
    # 9. Initial translation (zero because points centered)

    # 10. Stack into a single tensor
    theta_init = torch.cat([
        e1.unsqueeze(0),
        e2.unsqueeze(0),
        a1.unsqueeze(0),
        a2.unsqueeze(0),
        a3.unsqueeze(0),
        euler_angles,
        translation
    ])


    # Make it a learnable parameter
    theta = torch.nn.Parameter(theta_init)

    return points_centered,theta, scale, p0, sigma2, t0

def initialize_theta_pytorch(points, rescale=True):
    """
    points: Tensor of shape (N, 3), already on CUDA
    returns: initialized theta (11,) as nn.Parameter
    """
    device = points.device

    # 1. Center points
    t0 = points.mean(dim=0)
    # print("t0: ", t0)
    points_centered = points - t0

    # 2. Rescale
    if rescale:
        max_length = torch.max(points_centered.abs())
        scale = max_length / 10.0
        points_centered = points_centered / scale
    else:
        scale = 1.0

    # 3. PCA (Eigen decomposition)
    centered = points_centered
    cov = centered.T @ centered / centered.shape[0]
    print("cov: ", cov)
    eigvals, eigvecs = torch.linalg.eig(cov)  # Eigh is symmetric, fast
    print("eigVals: ", eigvals)
    print("eigVecs: ", eigvecs)
    idx = torch.argsort(eigvals.real, descending=True)
    eigvecs = eigvecs[:, idx]  # Sorted eigenvectors
    print(idx)
    print("eigvecs: ", eigvecs)

    # 4. Build initial rotation matrix
    # Same as article: [-EigVec[:, 0], -EigVec[:, 2], cross(EigVec[:,0],EigVec[:,2])]
    x_axis = -eigvecs.real[:, 0]
    z_axis = -eigvecs.real[:, 2]
    y_axis = torch.cross(eigvecs.real[:, 0], eigvecs.real[:, 2])

    R0 = torch.stack([x_axis, z_axis, y_axis], dim=1)  # 3x3 matrix
    print("R0: ", R0)
    # print("R0: ", R0)
    # 5. Rotate points
    points_rot0 = points_centered @ R0

    V = BoundVolume(points_rot0)
    
    print("BoundVolume: ", V)
    
    p0= 1/V
    
    sigma2 = V ** (1 / 3) / 10.0

    # 6. Estimate semi-axes
    s0 = torch.median(points_rot0.abs(), dim=0).values

    # 7. Initial parameters
    e1 = torch.tensor(1.0, device=device)
    e2 = torch.tensor(1.0, device=device)
    a1, a2, a3 = s0[0]/2.0, s0[1]/2.0, s0[2]/2.0

    print(a1, a2, a3)
    # 8. Get initial rotation as Euler angles
    euler_angles = rotation_matrix_to_euler(R0)
    print("euler angles: ", euler_angles)
    # 9. Initial translation (zero because points centered)
    translation = torch.zeros(3, device=device)

    # 10. Stack into a single tensor
    theta_init = torch.cat([
        e1.unsqueeze(0),
        e2.unsqueeze(0),
        a1.unsqueeze(0),
        a2.unsqueeze(0),
        a3.unsqueeze(0),
        euler_angles,
        translation
    ])

    # dz_local = torch.min(points_centered[:,2])

    # # Make it a learnable parameter
    # dz_local = torch.nn.Parameter(dz_local)
    
    theta = torch.nn.Parameter(theta_init)

    return points_centered,theta, scale, p0, sigma2, t0

def compute_point_volume(points, probs, theta):
    # Only keep points with high probability
    mask = probs > 0.7
    if mask.sum() < 3:  # Avoid degenerate case
        return torch.tensor(0.0, device=points.device)

    selected = points[mask]
    
    cov = selected.T @ selected / selected.shape[0]
    # eigvals, eigvecs = torch.linalg.eig(cov)  # Eigh is symmetric, fast
    eigvals, eigvecs = torch.linalg.eigh(cov)  # better for symmetric matrices
    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]  # Sorted eigenvectors

    # 4. Build initial rotation matrix
    # Same as article: [-EigVec[:, 0], -EigVec[:, 2], cross(EigVec[:,0],EigVec[:,2])]
    x_axis = -eigvecs[:, 0]
    z_axis = -eigvecs[:, 2]
    y_axis = torch.cross(eigvecs[:, 0], eigvecs[:, 2])

    R0 = torch.stack([x_axis, z_axis, y_axis], dim=1)  # 3x3 matrix
    
    Rot = build_rotation_matrix(theta[5:8])

    # print("R0: ", R0)
    # 5. Rotate points
    points_rot0 = selected @ Rot

    point_volume = BoundVolume(points_rot0)
    return point_volume

def compute_superquadric_volume(theta):
    a1, a2, a3 = theta[2], theta[3], theta[4]
    superquadric_volume = 4/3*torch.pi*a1 * a2 * a3
    return superquadric_volume

def superquadric_normals(points_local, theta):
    e1, e2 = theta[0], theta[1]
    a1, a2, a3 = theta[2], theta[3], theta[4]
    
    x = points_local[:, 0]
    y = points_local[:, 1]
    z = points_local[:, 2]

    eps = 1e-8  # for numerical stability
    
    # Terms
    x_abs = torch.abs(x / a1) + eps
    y_abs = torch.abs(y / a2) + eps
    z_abs = torch.abs(z / a3) + eps

    sgn_x = torch.sign(x)
    sgn_y = torch.sign(y)
    sgn_z = torch.sign(z)

    # Intermediate power
    temp = (x_abs**(2/e2) + y_abs**(2/e2))
    temp_pow = temp ** (e2/e1 - 1)

    # Gradient in local space
    dFdx = (2 / e1) * temp_pow * (x_abs ** ((2/e2) - 1)) * (sgn_x / a1) / a1
    dFdy = (2 / e1) * temp_pow * (y_abs ** ((2/e2) - 1)) * (sgn_y / a2) / a2
    dFdz = (2 / e1) * (z_abs ** ((2/e1) - 1)) * (sgn_z / a3) / a3

    normals_local = torch.stack([dFdx, dFdy, dFdz], dim=1)

    # Normalize
    normals_local = F.normalize(normals_local, dim=1)

    return normals_local  # in local frame


def compactness_loss(points, probs, theta, weight=1.0):
    point_vol = compute_point_volume(points, probs, theta)
    print("point vol: ", point_vol)
    sq_vol = compute_superquadric_volume(theta)
    print("sq vol: ", sq_vol)
    
    # Avoid division by zero
    eps = 1e-6
    loss = torch.relu(sq_vol - 1.05*point_vol) / (point_vol + eps)  

    print("loss compact", loss)
    return weight * loss


def superquadric_function(points, theta):
  
    # Assume points: (N, 3)
    e1, e2 = theta[0], theta[1]
    a1, a2, a3 = theta[2], theta[3], theta[4]
    rot = theta[5:8]  # Euler angles
    t = theta[8:11]   # translation vector
    
    # print("theta: ", theta)    
    # Build rotation matrix from Euler angles
    Rot = build_rotation_matrix(rot)

    
    # Transform points
    points_local = points @ Rot - t @ Rot
    # Normalize by semi-axes
    x_ = points_local[:, 0] / a1
    y_ = points_local[:, 1] / a2
    z_ = points_local[:, 2] / a3
    
    radius_factor = 2.0
    rmax = radius_factor * torch.max(torch.stack([a1, a2, a3]))
    mask = (points_local.norm(dim=1) <= rmax)
    
    EPS, UMAX = 1e-12, 1e3

    ux = torch.clamp(torch.abs(x_), min=EPS, max=UMAX)
    uy = torch.clamp(torch.abs(y_), min=EPS, max=UMAX)
    uz = torch.clamp(torch.abs(z_), min=EPS, max=UMAX)

    # Primero las potencias con exponente 2/e2: base > 0 garantizada
    px = torch.pow(ux, 2.0 / e2)
    py = torch.pow(uy, 2.0 / e2)

    s  = torch.clamp(px + py, min=EPS)                # base del ^(e2/e1) > 0
    term1 = torch.exp((e2 / e1) * torch.log(s))       # en lugar de s**(e2/e1)

    print("uz", uz)
    print("min uz:", uz.min().item())
    print("e1: ", e1)
    term2 = torch.pow(uz, 2.0 / e1)                   # base > 0 garantizada

    # term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
    # term2 = (torch.abs(z_)**(2/e1))
    inside_outside = term1 + term2
    
    
    return inside_outside


import torch
from math import pi

# Forward (paper) helper to keep the definitions together
def _bend_forward(P, b, alpha):
    eps = 1e-8
    b = torch.clamp(b, min=1e-6)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho  = torch.sqrt(x * x + y * y + eps)
    ang  = torch.atan2(y, x)
    r    = torch.cos(alpha - ang) * rho
    gamma = b * z
    invb  = 1.0 / b
    R = invb - (invb - r) * torch.cos(gamma)
    dx = (R - r) * torch.cos(alpha)
    dy = (R - r) * torch.sin(alpha)
    dz = (invb - r) * torch.sin(gamma)
    out = torch.empty_like(P)
    out[:, 0] = x + dx
    out[:, 1] = y + dy
    out[:, 2] = dz
    return out

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


def fitting_loss(points, theta, p0, sigma2, k):
    # Assume points: (N, 3)
    e1, e2 = theta[0], theta[1]
    a1, a2, a3 = theta[2], theta[3], theta[4]
    rot = theta[5:8]  # Euler angles
    t = theta[8:11]   # translation vector
    
    # print("theta: ", theta)    
    # Build rotation matrix from Euler angles
    Rot = build_rotation_matrix(rot)

    
    # Transform points
    points_local = points @ Rot - t @ Rot
    # Normalize by semi-axes
    x_ = points_local[:, 0] / a1
    y_ = points_local[:, 1] / a2
    z_ = points_local[:, 2] / a3
    
    # denom_x = 1 + k * z_
    # denom_x = torch.clamp(denom_x, min=1e-3)  # avoid division by zero or flip

    # x_tapered = points_local[:, 0] / denom_x
    # y_tapered = points_local[:, 1] / (denom_x)
    # z_ = points_local[:, 2]

    # x_ = x_tapered / a1
    # y_ = y_tapered / a2
    # z_ = z_ / a3

    # Option 1: mask or ignore invalid points
    # valid_mask = (points_local[:, 2] >= 0)
    # points_local = points_local[valid_mask]
    # x_ = points_local[:, 0] / a1
    # y_ = points_local[:, 1] / a2
    # z_ = points_local[:, 2] / a3

    # term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
    # term2 = (torch.abs(z_)**(2/e1))
    # inside_outside = term1 + term2
        
    
    # term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
    # term2 = z_
    # F_val = term1 - term2
    
    # Add a learned or fixed base width offset
    r_offset = k  # e.g., 0.01
    
    # Compute r_offset shape
    r_profile = r_offset + ((points_local[:, 2] / a3).clamp(min=0.0)) ** (1 / e1)  # ensure positivity

    # Compute normalized coords
    x_norm = points_local[:, 0] / (a1 * r_profile)
    y_norm = points_local[:, 1] / (a2 * r_profile)

    # Classic superellipse expression
    term1 = (torch.abs(x_norm) ** (2 / e2) + torch.abs(y_norm) ** (2 / e2)) ** (e2 / e1)

    F_val = term1 - 1

    

    # r = r_offset + ((z_ / a3) ** (1 / e1))
    # term1 = (torch.abs(x_ / r) ** (2/e2) + torch.abs(y_ / r) ** (2/e2)) ** (e2 / e1)
    # F_val = term1 - 1
    
    # print("negative: ", torch.sum(z_<0))
    # # print("para: ", F_val)
    
    r_norm = torch.norm(points_local, dim=1)

    distances =  r_norm*torch.abs(F_val)

    
    
    

    

    # r_norm = torch.norm(points_local, dim=1)

    # distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
    # distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
    # print("Distances: ", distances)

    c = (2 * torch.pi * sigma2) ** (- 3 / 2)
    w=0.05
    const = (w * p0) / (c * (1 - w))
    
    dist_term = torch.exp(-1 / (2 * sigma2) * distances ** 2)
    


    # Final inlier probability with normal guidance
    p = dist_term / (const + dist_term)
    p = torch.clamp(p, min=1e-10)
    
    p = torch.clamp(p, min=1e-10)
    # print("prob: ", p)
    
    scaled_distances = distances
    
    cost = p.detach()*scaled_distances**2
    
    loss = cost.sum()

    z_penalty = torch.relu(-z_) ** 2 * 100.0  # weight can be tuned    

    weights = torch.sigmoid((p - 0.8) * 20)  # sigmoid approx. of a threshold at 0.5

    z_vals = points_local[:, 2]
    z_mean = (weights * z_vals).sum() / weights.sum()
    z_var = ((weights * (z_vals - z_mean) ** 2).sum()) / weights.sum()
    z_extent = 2 * torch.sqrt(z_var)*a3  # ≈ 95% of the spread
    
    penalty_weight = 0.4
    # Apply penalty if a3 exceeds z_extent
    overshoot = torch.relu(a3 - z_extent)
    extent_penalty = overshoot ** 2 * penalty_weight
    # p = weights
    
    allowed_margin = 0.01  # small tolerance below point cloud
    shape_base_z = t[2]
    min_z_points = points_local[:, 2].min()

    drop_penalty = torch.relu(min_z_points - shape_base_z - allowed_margin) ** 2 * 20.0
    
    z_max = (z_vals).max()
    # print("z_max: ", z_max)
    # print("a3: ", a3)
    extent_penalty = (a3-z_max) ** 2 * 10.0

    # print("extent_penalty: ", extent_penalty)

    return loss+z_penalty.sum() + drop_penalty+ extent_penalty, p, distances

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
      pl = unbend_points_torch(pl, b, alpha, iters=4)
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
def _active_idx(points, theta, factor=50.0, pad=0.3):
    # máscara dinámica por iteración, usando θ detached (la selección no mete gradiente)
    th = theta.detach()
    R = build_rotation_matrix(th[5:8])
    t = th[8:11]
    pts_local = points @ R - t @ R

    a1 = th[2].abs().clamp_min(1e-6)
    a2 = th[3].abs().clamp_min(1e-6)
    a3 = th[4].abs().clamp_min(1e-6)

    # métrica elipsoidal p=2 (barata): (x/a1)^2+(y/a2)^2+(z/a3)^2 <= (factor+pad)^2
    q = (pts_local[:,0]/a1)**2 + (pts_local[:,1]/a2)**2 + (pts_local[:,2]/a3)**2
    thr = (factor + pad)**2
    return torch.nonzero(q <= thr, as_tuple=False).squeeze(-1)

def _spow(u, p, eps=1e-8, umax=1e3, max_log=100000.0):
    u = torch.clamp(u.abs(), min=eps, max=umax)
    return torch.exp(torch.clamp(p * torch.log(u), min=-max_log, max=max_log))

# --- reusable: differentiable bending in XY plane ----
def bend_points_torch(P, b, alpha):
    """
    P: (N,3) points in LOCAL frame (torch)
    b: scalar tensor > 0  (can be learnable)
    alpha: scalar tensor (radians)
    """
    eps = 1e-8
    b_safe = torch.clamp(b, min=1e-6)

    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    rho  = torch.sqrt(x * x + y * y + eps)
    ang  = torch.atan2(y, x)
    r    = torch.cos(alpha - ang) * rho
    gamma = b_safe * z

    invb = 1.0 / b_safe
    R = invb - (invb - r) * torch.cos(gamma)

    dx = (R - r) * torch.cos(alpha)
    dy = (R - r) * torch.sin(alpha)
    dz = (invb - r) * torch.sin(gamma)

    out = torch.empty_like(P)
    out[:, 0] = x + dx
    out[:, 1] = y + dy
    out[:, 2] = dz
    return out

def superquadric_total_loss(points, theta, b, alpha, p0, weight_compactness, sigma2, number_of_rays, number_samples_per_ray, ray_samples_flat):
    
    # # Assume points: (N, 3)
    # e1, e2 = theta[0], theta[1]
    # a1, a2, a3 = theta[2], theta[3], theta[4]
    # rot = theta[5:8]  # Euler angles
    # t = theta[8:11]   # translation vector
    
    # e1_safe = e1.clamp(0.01, 1.99)
    # e2_safe = e2.clamp(0.01, 1.99)

    # a1_safe = a1.abs().clamp_min(1e-6)   # evita denominadores ~0, mantiene magnitud
    # a2_safe = a2.abs().clamp_min(1e-6)
    # a3_safe = a3.abs().clamp_min(1e-6)

    # # print("theta: ", theta)    
    # # Build rotation matrix from Euler angles
    # Rot = build_rotation_matrix(rot)

    # print("a1: ", a1, " a2: ", a2, " a3: ", a3, " e1: ", e1, " e2: ", e2)
    # # Transform points
    # points_local = points @ Rot - t @ Rot
    # # Normalize by semi-axes
    # x_ = points_local[:, 0] / a1_safe
    # y_ = points_local[:, 1] / a2_safe
    # z_ = points_local[:, 2] / a3_safe
    
    # # term1 = (torch.abs(x_)**(2/e2_safe) + torch.abs(y_)**(2/e2_safe))**(e2_safe/e1_safe)
    # # term2 = (torch.abs(z_)**(2/e1_safe))
    
    # EPS, UMAX = 1e-8, 1e3

    # ux = torch.clamp(torch.abs(x_), min=EPS, max=UMAX)
    # uy = torch.clamp(torch.abs(y_), min=EPS, max=UMAX)
    # uz = torch.clamp(torch.abs(z_), min=EPS, max=UMAX)

    # # Primero las potencias con exponente 2/e2: base > 0 garantizada
    # px = torch.pow(ux, 2.0 / e2)
    # py = torch.pow(uy, 2.0 / e2)

    # s  = torch.clamp(px + py, min=EPS)                # base del ^(e2/e1) > 0
    # term1 = torch.exp((e2 / e1) * torch.log(s))       # en lugar de s**(e2/e1)

    # term2 = torch.pow(uz, 2.0 / e1)                   # base > 0 garantizada
    # # print("term1: ", term1)
    # # print("term2: ", term2)
    
    # inside_outside = term1 + term2

    # # inside_outside = (
    # # (torch.abs(x_).pow(2/e2) + torch.abs(y_).pow(2/e2)).pow(e2/e1)
    # # + torch.abs(z_).pow(2/e1))

    
    # r_norm = torch.norm(points_local, dim=1)

    # distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
    
    # c = (2 * torch.pi * sigma2) ** (- 3 / 2)
    # w=0.1
    # const = (w * p0) / (c * (1 - w))
    
    # dist_term = torch.exp(-1 / (2 * sigma2) * distances ** 2)
    

    # # Final inlier probability with normal guidance
    # p = dist_term / (const + dist_term)
    # p = torch.clamp(p, min=1e-10)
    
    # # print("prob: ", p)
    
    # scaled_distances = distances
    
    # fit_cost = p*scaled_distances**2
    
    # fit_loss = fit_cost.sum()
    
     # --- FIT: usa solo puntos cercanos ---
    idx_fit = _active_idx(points, theta)        # dinámico por iteración
    
    # print("idx_fit: ", idx_fit)
    if idx_fit.numel() == 0:
        fit_loss = torch.tensor(0., device=points.device, requires_grad=True)
        p = torch.zeros(points.shape[0], device=points.device)
        distances = torch.zeros_like(p)
    else:
        # evalúa distancias solo en idx_fit (tu fórmula estable)
        R = build_rotation_matrix(theta[5:8]); t = theta[8:11]
        points_local = points @ R - t @ R

        ################################## BENDING ####################################################
        if b is not None:
            # b/alpha can be Python floats; make them tensors on the same device
            if not isinstance(b, torch.Tensor):     b = torch.tensor(float(b), device=points.device)
            if not isinstance(alpha, torch.Tensor): alpha = torch.tensor(float(alpha), device=points.device)
            points_local = unbend_points_torch(points_local, b, alpha, iters=4)
        ################################### BENDING ####################################################  
        
        a1 = theta[2].abs().clamp_min(1e-6)
        a2 = theta[3].abs().clamp_min(1e-6)
        a3 = theta[4].abs().clamp_min(1e-6)
        e1, e2 = theta[0], theta[1]
        
        x_ = points_local[:, 0] / a1
        y_ = points_local[:, 1] / a2
        z_ = points_local[:, 2] / a3

        
        term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
        term2 = (torch.abs(z_)**(2/e1))
        inside_outside = term1 + term2
        
        r_norm = torch.norm(points_local, dim=1)

        distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
        
        c = (2 * torch.pi * sigma2) ** (- 3 / 2)
        w=0.6
        const = (w * p0) / (c * (1 - w))
        
        dist_term = torch.exp(-1 / (2 * sigma2) * distances ** 2)
        

        # Final inlier probability with normal guidance
        p = dist_term / (const + dist_term)
        p = torch.clamp(p, min=1e-10)
        
        # print("prob: ", p)
        
        scaled_distances = distances
        
        fit_cost = p.detach()*scaled_distances**2
        
        fit_loss = fit_cost.sum()
            
        # pl = points[idx_fit] @ R - t @ R
        # x_ = pl[:,0] / a1; y_ = pl[:,1] / a2; z_ = pl[:,2] / a3

        # px = _spow(x_, 2.0/theta[1]); py = _spow(y_, 2.0/theta[1])
        # s  = torch.clamp(px + py, min=1e-8)
        # term1 = torch.exp(torch.clamp((theta[1]/theta[0]) * torch.log(s), -10000000.0, 10000000.0))
        # term2 = _spow(z_, 2.0/theta[0])
        # F = term1 + term2



        # r_norm = pl.norm(dim=1)
        # dist_idx = r_norm * torch.abs(torch.exp(torch.clamp((-theta[0]/2)*torch.log(torch.clamp(F,1e-8)), -10000000.0, 10000000.0)) - 1.0)

        # # probas EMS solo en idx
        # c = (2 * torch.pi * sigma2) ** (-3/2)
        # w = 0.1
        # const = (w * (1/BoundVolume(pl))) / (c * (1 - w))
        
        # # print("const: ", const)
        # dist_term = torch.exp(-0.5 * (dist_idx**2) / sigma2)
        # p_idx = dist_term / (const + dist_term)
        # p_idx = torch.clamp(p_idx, min=1e-10)

        # # ensamblar vectores completos
        # p = torch.zeros(points.shape[0], device=points.device)
        # p[idx_fit] = p_idx
        # distances = torch.zeros_like(p)
        # distances[idx_fit] = dist_idx

        # fit_loss = torch.sum(p_idx * (dist_idx**2))
        
    print("fit: ", fit_loss)


    # print("ray samples: ",ray_samples_flat)
    # inside_score = superquadric_function(ray_samples_flat, theta)
    # soft_inside = torch.sigmoid(-(inside_score - 1) * 10)  # sharpness ≈ 10–100
    # # penalty = soft_inside.sum()    
    # free_space_penalty = soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()
    inside_score = sq_inside_near_only(ray_samples_flat, theta, b, alpha)  # máscara cambia cada step
    soft_inside  = torch.sigmoid(-(inside_score - 1.0) * 20.0)
    free_space_penalty = soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()
    print("free space penalty: ", free_space_penalty)


    

    return 1.0*fit_loss + 80.0*free_space_penalty, fit_loss, 15.0*free_space_penalty, p, distances
  



def total_loss(points, theta, p0, weight_compactness, sigma2, number_of_rays, number_samples_per_ray, ray_samples_flat, k):
    fit, p, distances = fitting_loss(points, theta, p0, sigma2, k)
    
    # print("fit: ", fit)

    
    
    # values = superquadric_function(ray_samples_flat, theta)
    # print("values: ")
    # inside = values < 1.0
    # penalty = inside.float().sum() / len(ray_samples_flat)
    
    # print("samples: ", len(ray_samples_flat))
    # print("inside: ", inside.float().sum())
    # print("penaly1:", penalty)

    # inside_score = superquadric_function(ray_samples_flat, theta)
    # soft_inside = torch.sigmoid(-(inside_score - 1) * 10)  # sharpness ≈ 10–100
    # # penalty = soft_inside.sum()    
    # penalty = soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()
    # print("penalty: ", penalty)
    # Reshape back to (N, S) and check if any point along ray is inside
    # inside_any = inside.view(N, samples_per_ray).any(dim=1)  # (N,)
    
    # compact = compactness_loss(points, p, theta, weight_compactness)
    # print("penalty: ", penalty)
    
    # loss += lambda_entropy * entropy_penalty

    # f_vals = superquadric_function(ray_samples_flat, theta)  # shape (N * S,)
    
    # print("f_vals: ", f_vals)
    
    # print(torch.sum(f_vals < 1.0))
    # print("N*S", number_of_rays*number_samples_per_ray)
    
    # # Detach to avoid autograd
    # ray_samples_flat = ray_samples_flat.detach()

    # # Evaluate superquadric function: returns (N*S,) values
    # f_vals = superquadric_function(ray_samples_flat, theta)

    # # Reshape back: (N, S)
    # f_vals_per_ray = f_vals.view(number_of_rays, number_samples_per_ray)

    # # Count number of points inside for each ray (f_val < 1)
    # inside_mask = f_vals_per_ray < 1.0
    # per_ray_inside_count = inside_mask.sum(dim=1)  # shape (N,)

    # # Total or average, if needed
    # total_inside = inside_mask.sum()
    # average_inside_per_ray = per_ray_inside_count.float().mean()

    # print("Total inside samples:", total_inside.item())
    # print("Per-ray counts:", per_ray_inside_count)
    # print("Average inside per ray:", average_inside_per_ray.item())
    


    

    return fit, p, distances
  
  
  
  

# point_cloud = read_ply("data/objects7.ply")
# point_cloud = remove_close_points(point_cloud, 0.005)

# point_cloud = filter_by_z(point_cloud, -np.inf, 1.94)

# all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor



# fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
# showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))

point_cloud = read_with_open3d("data/sceneReplica/final_scenes/pcds/scene_25/"+"cloud.pcd")
point_cloud = remove_close_points(point_cloud, 0.0025)
filtered_points, plane_points, plane_model = remove_largest_plane(point_cloud, distance_threshold=0.0025)
point_cloud = filter_by_z(point_cloud, -np.inf, 1.32)
all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
showPoints(filtered_points, scale_factor=0.0025, color=(0,0.5,0.5))
showPoints(np.array([[0,0,0]]))
mlab.show()




print("plane model: ", plane_model)

table_normal = torch.tensor(plane_model[:3], dtype=torch.float32, device='cuda')




# --- Intrinsics (yours) ---
FX = 554.254691191187
FY = 554.254691191187
CX = 320.5
CY = 240.5

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

# ---------------- Example ----------------
# points_np: (N,3) from your /head_camera/depth_registered/points (same camera!)
# seg_png: color segmentation aligned with that camera (same resolution)
labels, id2color = load_seg_as_labels("data/sceneReplica/final_scenes/segmasks/scene_25/gtseg_ord-nearest_first_step-0.png", inflate_px=10)
point_labels = label_points_from_seg(filtered_points, labels)
clusters = split_points_by_label(filtered_points, point_labels)

print("clusters", clusters)
# # clusters[k] is Nx3 for each object; id2color[k] gives its RGB color.

# p= None
# kmeans = KMeans(n_clusters=4).fit(filtered_points)
# clustering = DBSCAN(eps=0.03, min_samples=6).fit(filtered_points)
# n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
# print(f"Number of clusters: {n_clusters}")

camera_origin = torch.zeros_like(all_points)  # shape (N, 3), all (0,0,0)
directions = all_points - camera_origin  # or just points if origin is (0,0,0)

# Now sample along these rays
number_samples_per_ray = 120
number_of_rays = all_points.shape[0]

t_vals = torch.linspace(0.03, 1.1, number_samples_per_ray, device=all_points.device)  # go slightly past the surface
ray_points = camera_origin[:, None, :] + t_vals[None, :, None] * directions[:, None, :]
ray_samples_flat = ray_points.reshape(-1, 3)

k = torch.tensor(0.6)
k = torch.nn.Parameter(k)


all_params_modeled = {}


def fit_multiple_shape_to_clusters(clusters_points_np, label="root", depth=0, max_depth=3):
  n_clusters = len(clusters_points_np)
  n_steps = 300

  # Store everything in lists
  thetas = []
  optimizers = []
  sigma2s = []
  p0s = []
  losses_per_step = [[] for _ in range(n_clusters)]
  points_centered_list = []
  t0s = []
  losses = []

  # Initialize all
  for cluster_np in clusters_points_np:
      points = torch.tensor(cluster_np, dtype=torch.float32, device='cuda')
      points_centered, theta, _, p0, sigma2, t0 = initialize_theta_pytorch(points, False)
      
      ray_samples = ray_samples_flat - t0
      
      thetas.append(theta)
      optimizers.append(torch.optim.Adam([theta], lr=1e-3))
      sigma2s.append(sigma2)
      p0s.append(p0)
      points_centered_list.append(points_centered)
      t0s.append(t0)
      losses.append(None)  # Placeholder for losses per cluster

  # Now interleave the optimization:
  for step in range(n_steps):
      # 1. Zero all grads first
      for opt in optimizers:
          opt.zero_grad()

      # 2. Compute all losses and backward passes
      for i in range(n_clusters):
          theta = thetas[i]
          points_centered = points_centered_list[i]
          p0 = p0s[i]
          sigma2 = sigma2s[i]
          ray_samples = ray_samples_flat - t0s[i]

          loss, p, distances = superquadric_total_loss(
              points_centered, theta, p0, 0.0, sigma2,
              number_of_rays, number_samples_per_ray, ray_samples
          )
          losses[i] = (loss, p, distances)
          loss.backward(retain_graph=True)
          losses_per_step[i].append(loss.item())

      # 3. Step all optimizers
      for opt in optimizers:
          opt.step()

      # 4. Update sigma and clamp theta
      for i in range(n_clusters):
          theta = thetas[i]
          loss, p, distances = losses[i]

          with torch.no_grad():
              if step % 10 == 0:
                  sigma2_new = 2 * torch.sum(p * distances ** 2) / (3 * torch.sum(p) + 1e-8)
                  sigma2s[i] = 0.8 * sigma2s[i] + 0.2 * sigma2_new

              theta[0].clamp_(0.01, 2.0)
              theta[1].clamp_(0.01, 2.0)
              theta[2:5].clamp_(0.001, 1.5)
              theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi

      if step % 50 == 0:
          print(f"Global step {step}: {[l[0].item() for l in losses]}")
  translated_thetas = []
  for i, theta in enumerate(thetas):
      theta = theta.clone()  # (optional if you're not sure)
      theta[8:11] = theta[8:11] + t0s[i]
      translated_thetas.append(theta)
  return translated_thetas

def sq_distances(points, theta, b=None, alpha=None):
    """
    Compute distances from points to the superquadric surface.
    This version is differentiable if called with grad enabled.
    """


    # Rotation + translation
    R = build_rotation_matrix(theta[5:8])
    t = theta[8:11]
    points_local = points @ R - t @ R

    # optional bending
    if b is not None:
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(float(b), device=points.device)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), device=points.device)
        points_local = unbend_points_torch(points_local, b, alpha, iters=4)

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

def compute_p_from_dist(distances, sigma2, p0, w=0.6):
    """
    Compute point responsibilities (E-step).
    distances: (N,) tensor
    sigma2: variance
    p0: uniform outlier prob ~ 1/Volume
    w: mixing weight
    """

    c = (2 * torch.pi * sigma2) ** (- 3 / 2)
    const = (w * p0) / (c * (1 - w))

    dist_term = torch.exp(-1 / (2 * sigma2) * distances ** 2)
    p = dist_term / (const + dist_term)
    return torch.clamp(p, min=1e-10)

def free_space_loss(ray_samples_flat, theta, b, alpha, number_of_rays):
    inside_score = sq_inside_near_only(ray_samples_flat, theta, b, alpha)
    soft_inside = torch.sigmoid(-(inside_score - 1.0) * 20.0)
    return soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()


def fit_shape_to_cluster(cluster_points_np, superquadric = True, init_theta=None):
  

    # loss_per_iteration = []
    
    points = torch.tensor(cluster_points_np, dtype=torch.float32, device='cuda')

    points_centered,theta,_, p0, sigma2, t0 = initialize_theta_pytorch(points, False)

    
    #################################### BENDING #####################################
    
    alpha = torch.tensor(1.57079632679/4.0)
    alpha = torch.nn.Parameter(alpha)
    H= theta[4]
    r_max = torch.max(theta[2], theta[3])   # conservative

    b_phase_max = 1.5 / (H + 1e-6)
    b_geom_max  = 0.8 / (r_max + 1e-6)
    b_max = torch.minimum(b_phase_max, b_geom_max)

    # init (if you're creating b here)
    b = torch.tensor(0.05, device=theta.device) * b_max.detach()
    b = torch.nn.Parameter(b)

    #################################### BENDING #####################################

    
    
    camera_position = torch.tensor([0.0, 0.0, 0.0], device=all_points.device)  # shape (3,)
    cluster_vecs = points - camera_position  # shape (N, 3)
    cluster_vecs = torch.nn.functional.normalize(cluster_vecs, dim=1)
    
    center_dir = torch.mean(cluster_vecs, dim=0)
    center_dir = center_dir / torch.norm(center_dir)
    
    cos_angles = (cluster_vecs @ center_dir)
    max_angle = torch.acos(torch.clamp(cos_angles.min(), -1.0, 1.0))  # in radians
    
    margin = 15 * torch.pi / 180  # radians
    final_cone_angle = max_angle + margin
    cos_thresh = torch.cos(final_cone_angle)

    camera_origin = torch.zeros_like(all_points)  # shape (N, 3), all (0,0,0)

    all_vecs = all_points - camera_position
    all_vecs = torch.nn.functional.normalize(all_vecs, dim=1)

    mask = (all_vecs @ center_dir) > cos_thresh
    points_in_cone = all_points[mask]
    # igual es el points_in_cone lo que esta dando problemas
    # points_in_cone = all_points
    
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    points_np = points_in_cone.detach().cpu().numpy()   # -> Nx3 numpy array
    showPoints(points_np, scale_factor=0.01, color=(0,1,0))
    mlab.show()
    

    camera_origin = torch.zeros_like(points_in_cone)  # shape (N, 3), all (0,0,0)
    directions = points_in_cone - camera_origin  # or just points if origin is (0,0,0)

    # Now sample along these rays
    number_samples_per_ray = 300
    number_of_rays = points_in_cone.shape[0]


    t_vals = torch.linspace(0.5, 0.995, number_samples_per_ray, device=points_in_cone.device)  # go slightly past the surface
    ray_points = camera_origin[:, None, :] + t_vals[None, :, None] * directions[:, None, :]
    
    print("len ray: ", ray_samples_flat.shape)
    
    current_ray_samples_flat = ray_points.reshape(-1, 3) - t0
    
    print("len ray: ", current_ray_samples_flat.shape)

    # current_ray_samples_flat = ray_samples_flat - t0
    print("Initial theta: ", theta)
    
    # sigma2 = torch.nn.Parameter(sigma2)
    iter_sigma = 0

    if superquadric:
        optimizer = torch.optim.Adam([theta, b, alpha], lr=1e-3, weight_decay=0.01)

        tolerance = 1e-4  # or something like 1e-4 depending on your scale
        patience = 80     # number of steps with small change before stopping
        no_improve_steps = 0

        hist = make_hist() 

        T = 3000
        K=5
        freeze_every = 150
        sigma_momentum = 0.7    # EMA for sigma2
        sigma_every = 5         # update cadence
        lambda_free = 80.0
        
        # We are doing what is called "soft-em" we care for a faster optimization
        prev_loss = None
        for outer in range(T):
            if outer % freeze_every == 0:
                # --- E-step: calcula p una vez ---
                d = sq_distances(points, theta, b, alpha)

                with torch.no_grad():
                    p_fixed = compute_p_from_dist(d, sigma2, p0)

                sigma2_new = 2 * torch.sum(p_fixed * d**2) / (3 * torch.sum(p_fixed) + 1e-8)
                sigma2 = sigma_momentum * sigma2 + (1.0 - sigma_momentum) * sigma2_new

                # --- bloque de M-steps con p fijo ---
                for m in range(K):
                    optimizer.zero_grad()
                    d = sq_distances(points, theta, b, alpha)
                    fit = torch.sum(p_fixed.detach() * d**2)
                    free = free_space_loss(current_ray_samples_flat, theta, b, alpha, number_of_rays)
                    loss = fit + lambda_free * free
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        theta[0].clamp_(0.0001, 1.99)
                        theta[1].clamp_(0.0001, 1.99)
                        theta[2:5].clamp_(0.0001, 1.99)
                        theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi

            else:
                # --- soft-EM normal (un step) ---
                optimizer.zero_grad()
                d = sq_distances(points_centered, theta, b, alpha)  # shape (N,)

                # responsibilities from current θ, but no grad through p
                with torch.no_grad():
                    p = compute_p_from_dist(d, sigma2, p0).clamp_(1e-6, 1-1e-6)  # same as your old behavior
                fit = torch.sum(p * d**2)                        # soft responsibilities
                free = free_space_loss(current_ray_samples_flat, theta, b, alpha, number_of_rays)
                loss = fit + lambda_free * free
                loss.backward()
                optimizer.step()
                if (outer % sigma_every) == 0:
                    with torch.no_grad():
                        sigma2_new = 2 * torch.sum(p * d**2) / (3 * torch.sum(p) + 1e-8)
                        sigma2 = sigma_momentum * sigma2 + (1.0 - sigma_momentum) * sigma2_new       
                with torch.no_grad():
                    theta[0].clamp_(0.0001, 1.99)
                    theta[1].clamp_(0.0001, 1.99)
                    theta[2:5].clamp_(0.0001, 1.99)
                    theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi
 
        # for step in range(T):
        #     optimizer.zero_grad()

        #     # distances depend on current θ (keep grad!)
        #     d = sq_distances(points_centered, theta, b, alpha)  # shape (N,)

        #     # responsibilities from current θ, but no grad through p
        #     with torch.no_grad():
        #         p = compute_p_from_dist(d, sigma2, p0).clamp_(1e-6, 1-1e-6)  # same as your old behavior

        #     # fit term uses p frozen, d with grad
        #     fit = torch.sum(p * (d**2))

        #     # free space term uses current θ (has grad)
        #     free = free_space_loss(current_ray_samples_flat, theta, b, alpha, number_of_rays)

        #     loss = fit + lambda_free * free
        #     loss.backward()
        #     optimizer.step()

        #     # --- sigma2 update (like you did before) ---
        #     if (step % sigma_every) == 0:
        #         with torch.no_grad():
        #             sigma2_new = 2 * torch.sum(p * d**2) / (3 * torch.sum(p) + 1e-8)
        #             sigma2 = sigma_momentum * sigma2 + (1.0 - sigma_momentum) * sigma2_new

        #     # clamps (same as before)
        #     with torch.no_grad():
        #         theta[0].clamp_(0.0001, 1.99)
        #         theta[1].clamp_(0.0001, 1.99)
        #         theta[2:5].clamp_(0.0001, 1.99)
        #         theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi

            # # ---- σ² update (optional, infrequent) ----
            # if outer % 3 == 0:
            #     with torch.no_grad():
            #         sigma2_new = 2 * torch.sum(p_fixed * d**2) / (3 * torch.sum(p_fixed) + 1e-8)
            #         sigma2 = 0.8 * sigma2 + 0.2 * sigma2_new
        #         snap_theta(hist, theta, total_loss=loss, fit_loss=fit_loss, free=free_space_loss, sigma2=sigma2)
        #         import pandas as pd
        #         df = pd.DataFrame(hist)
        #         df.to_csv(f"results/evol_{step}.csv", index=False)


        # for step in range(700):  # or until convergence
        #     optimizer.zero_grad()

        #     loss, fit_loss, free_space_loss, p, distances = superquadric_total_loss(points_centered, theta, b, alpha, p0, 0.0, sigma2, number_of_rays, number_samples_per_ray, current_ray_samples_flat)
        #     print("loss: ", loss)
        #     loss_per_iteration.append(loss.item())  # Save it for plotting later
        #     loss.backward()
        #     optimizer.step()
        #     torch.autograd.set_detect_anomaly(True)
        #     # Clamp theta values to stay valid
        #     with torch.no_grad():
                
        #         theta1 = theta.clone()  # (optional if you're not sure)
        #         theta1[8:11] = theta1[8:11] + t0
        #         # print("theta", theta1)
        #         # print("p", p)
        #         iter_sigma+=1
                

        #         if iter_sigma == 5:
        #             iter_sigma = 0
        #             sigma2_new = 2 * torch.sum(p * distances**2) / (3 * torch.sum(p) + 1e-8)
        #             sigma2 = sigma2_new*0.3+0.7*sigma2
                    

                    
        #         # Clamp e1 and e2 between [0.1, 2.0]
        #         theta[0].clamp_(0.0001, 1.99)  # e1
        #         theta[1].clamp_(0.0001, 1.99)  # e2
                
        #         # Clamp semi-axes a1, a2, a3 to be positive
        #         theta[2:5].clamp_(0.0001,1.99)  # a1, a2, a3 positive
                
        #         # Optionally clamp rotation angles between [-pi, pi]
        #         theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi
                
        #         # --- actualización sigma + clamps (como ya haces) ---

        #         theta[0].clamp_(0.0001, 1.99)
        #         theta[1].clamp_(0.0001, 1.99)
        #         theta[2:5].clamp_(0.0001, 1.99)
        #         theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi

        #         # <<<--- SNAPSHOT por paso
        #         snap_theta(hist, theta, total_loss=loss, fit_loss=fit_loss, free=free_space_loss, sigma2=sigma2)
        #         import pandas as pd
        #         df = pd.DataFrame(hist)
        #         df.to_csv(f"results/evol_{step}.csv", index=False)

        #     if step>1:
        #         loss_change = abs(loss_per_iteration[-1] - loss_per_iteration[-2])
        #         print("loss_change: ",loss_change)
        #         if loss_change < tolerance:
        #             no_improve_steps += 1
        #         else:
        #             no_improve_steps = 0
        #         print("no improve steps: ", no_improve_steps)
        #         if no_improve_steps >= patience:
        #             print(f"Early stopping at step {step} (Δloss < {tolerance})")
        #             break
                
                
        #     if step % 100 == 0:
        #         print(f"Step {step}: Loss = {loss.item()}")
        theta_np = theta.detach().cpu().numpy()
        theta = theta.clone()  # (optional if you're not sure)
        theta[8:11] = theta[8:11] + t0
        theta_np = theta.detach().cpu().numpy()
        print(theta_np)
        b_np = b.detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()
        

        
        
        points_centered_np = points_centered.detach().cpu().numpy()

        indices = torch.nonzero(p > 0.9, as_tuple=False).squeeze()
        
        all_indices = np.arange(cluster_points_np.shape[0])
        
        selected_indices_good = indices.cpu().numpy()
        
        remaining_indices = np.setdiff1d(all_indices, selected_indices_good)
        
        indices_bad = torch.nonzero(p <= 0.3, as_tuple=False).view(-1)
        print("indices  bad: ", indices_bad)
        selected_indices_bad = indices_bad.cpu().numpy()
        print("indices bad: ", selected_indices_bad)
        remaining_indices1 = np.setdiff1d(remaining_indices, selected_indices_bad)
        

        
        camera_origin1 = torch.zeros_like(points[selected_indices_good])  # shape (N, 3), all (0,0,0)
        directions1 = points[selected_indices_good] - camera_origin1  # or just points if origin is (0,0,0)

        k_np = 0
        free_space_penalty1 = 0
        if selected_indices_good.size>0:
            # Now sample along these rays
            number_samples_per_ray1 = 200
            number_of_rays1 = points[selected_indices_good].shape[0]

            t_vals1 = torch.linspace(0, 0.96, number_samples_per_ray1, device=points.device)  # go slightly past the surface
            ray_points1 = camera_origin1[:, None, :] + t_vals1[None, :, None] * directions1[:, None, :]
            ray_samples_flat1 = ray_points1.reshape(-1, 3)

            inside_score1 = superquadric_function(ray_samples_flat1, theta)
            soft_inside1 = torch.sigmoid(-(inside_score1 - 1) * 10)  # sharpness ≈ 10–100
            # penalty = soft_inside.sum()    
            free_space_penalty1 = soft_inside1.view(number_of_rays1, -1).max(dim=1).values.mean()
            print("After optimization checking rays: ", free_space_penalty1)
    else:
        points_centered,theta,_, p0, sigma2, t0 = initialize_theta_superparaboloids_pytorch(points, table_normal, False)
        k = torch.tensor(0.8)
        k = torch.nn.Parameter(k)
        optimizer_superparaboloid = torch.optim.Adam([theta,k], lr=1e-3, weight_decay=0.01)


        for step in range(3500):  # or until convergence
          optimizer_superparaboloid.zero_grad()

          loss, p, distances = total_loss(points_centered, theta, p0, 0.0, sigma2, number_of_rays, number_samples_per_ray, current_ray_samples_flat, k)

          loss_per_iteration.append(loss.item())  # Save it for plotting later
          loss.backward()
          optimizer_superparaboloid.step()

          # Clamp theta values to stay valid
          with torch.no_grad():
              
              iter_sigma+=1
              

              if iter_sigma == 10:
                  iter_sigma = 0
                  fitting_error = torch.sum(p*(distances)**2)
                  
                  sigma2_new = 2 * torch.sum(p * distances**2) / (3 * torch.sum(p) + 1e-8)
                  sigma2 = 0.7 *sigma2+0.3*sigma2_new
                  
                  print("sigma2: ", sigma2)

                  
              # Clamp e1 and e2 between [0.1, 2.0]
              theta[0].clamp_(0.1, 2.0)  # e1
              theta[1].clamp_(0.1, 2.0)  # e2
              
              # Clamp semi-axes a1, a2, a3 to be positive
              theta[2:5].clamp_(0.001,1.5)  # a1, a2, a3 positive
              
              # Optionally clamp rotation angles between [-pi, pi]
              theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi
              
              
              
              
          if step % 100 == 0:
              print(f"Step {step}: Loss = {loss.item()}")
        # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        theta_np = theta.detach().cpu().numpy()
        theta = theta.clone()  # (optional if you're not sure)
        theta[8:11] = theta[8:11] + t0
        theta_np = theta.detach().cpu().numpy()
        k_np = k.detach().cpu().numpy()
        b_np = 0
        alpha_np = 0
        print(theta_np)



        p = torch.clamp(p, max=1.0)
        
        indices = torch.nonzero(p > 0.9, as_tuple=False).view(-1)        
        
        all_indices = np.arange(cluster_points_np.shape[0])
        
        selected_indices_good = indices.cpu().numpy()
        
        remaining_indices = np.setdiff1d(all_indices, selected_indices_good)
        
        indices_bad = torch.nonzero(p <= 0.2, as_tuple=False).view(-1)
        print("indices  bad: ", indices_bad)
        selected_indices_bad = indices_bad.cpu().numpy()
        print("indices bad: ", selected_indices_bad)
        remaining_indices1 = np.setdiff1d(remaining_indices, selected_indices_bad)
        free_space_penalty1 = 0



    return theta_np,k_np,b_np, alpha_np, selected_indices_good,remaining_indices1, selected_indices_bad, free_space_penalty1


    # import matplotlib.pyplot as plt

    # plt.plot(loss_per_iteration)
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("Loss vs Iteration")
    # plt.grid(True)
    # plt.show()
all_params_modeled = {}
idx = 0
fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
showPoints(filtered_points, scale_factor=0.005, color=(1,0,0))

# showSuperquadrics([ 0.9990,  0.9990,  0.0370,  0.0054,  0.0165, -2.4880, -0.2190, -2.3814,
#          0.1396,  0.0891,  0.6829])


mlab.show()

# for lid, cluster in clusters.items():
#     print(lid, cluster.shape)                    # each pts is Nx3
#     loss_per_iteration = []
#     # cluster = clusters[i]
#     current_cluster = cluster
#     fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
#     showPoints(current_cluster, scale_factor=0.01, color=(0,1,0))
#     showPoints(filtered_points, scale_factor=0.005, color=(1,0,0))
#     mlab.show()
    
# for i in range(0, len(clusters)-1):
for lid, cluster in clusters.items():
    print(lid, cluster.shape)                    # each pts is Nx3
    loss_per_iteration = []
    if lid == 1:
      continue
    # cluster = clusters[i]
    current_cluster = cluster
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    showPoints(current_cluster, scale_factor=0.01, color=(0,1,0))
    showPoints(filtered_points, scale_factor=0.005, color=(1,0,0))
    mlab.show()
    
    theta_np_sq = None
    queue = deque([current_cluster])
    
    while queue:
        current_cluster = queue.popleft()
        if current_cluster.shape[0]<=40:
          continue
        theta_np, k_np, b_np, alpha_np, selected_indices_good, remaining_indices, selected_indices_bad, free_space_penalty = fit_shape_to_cluster(current_cluster, True)
        print("theta_np: ", theta_np)
        print("b_np: ", b_np)
        print("alpha_np: ", alpha_np)
        
        all_params_modeled[idx] = {"cluster": lid,"type": "superquadric", "theta": theta_np, "k": 0, "b": b_np, "alpha": alpha_np, "free_space_penalty":free_space_penalty, "indices_good": selected_indices_good}
        idx +=1
        
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        showPoints(current_cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
        if selected_indices_bad.size >0:
            showPoints(current_cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
        showPoints(current_cluster[remaining_indices], scale_factor=0.01, color=(0,0,1))
        showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
        showSuperquadrics(theta_np, b_np, alpha_np)

        mlab.show()
        theta_np_sq = theta_np

        # else:
        #     theta_np, k_np, selected_indices_good, remaining_indices, selected_indices_bad, free_space_penalty = fit_shape_to_cluster(current_cluster, False)
        #     current_cluster = current_cluster[selected_indices_bad]
        #     all_params_modeled[idx] = {"type": "superparaboloid", "theta": theta_np, "k": k_np}
        #     idx+=1
            # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
            # showPoints(cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
            # if selected_indices_bad.size >0:
            #     showPoints(cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
            # showPoints(cluster[remaining_indices], scale_factor=0.01, color=(0,0,1))
            # showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
            # showTaperedSuperparaboloidWithBase(theta_np,k_np)
            # mlab.show()
        print("all_params", all_params_modeled)
        for id,params in all_params_modeled.items():
            if params["free_space_penalty"]>0.03:
              print("good: ",params["indices_good"])
              theta_np, k_np, b_np, alpha_np, selected_indices_good, remaining_indices, selected_indices_bad, free_space_penalty = fit_shape_to_cluster(current_cluster, False)
              params["type"] = "superparabolid"
              params["theta"] = theta_np
              params["k"] = k_np
              params["free_space_penalty"]=0
              fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
              showPoints(current_cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
              if selected_indices_bad.size >0:
                  showPoints(current_cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
              showPoints(current_cluster[remaining_indices], scale_factor=0.01, color=(0,0,1))
              showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
              showTaperedSuperparaboloidWithBase(theta_np, k_np)
              mlab.show()

                    
        next_indices = np.union1d(remaining_indices, selected_indices_bad)
        residual = current_cluster[next_indices]
        
        # ---- NEW: subcluster residual and enqueue ----
        subclusters = split_by_distance(residual, eps=None, min_samples=20)
        for sub in subclusters:
            if sub.shape[0] > 40:
                queue.append(sub)

#       showTaperedSuperparaboloidWithBase(params['theta'], params['k'])
# showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))
# mlab.show()



print(all_params_modeled)

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
for idx,params in all_params_modeled.items():
    if params["type"] == "superquadric":
      showSuperquadrics(params['theta'], params["b"], params["alpha"])
    else:
      showTaperedSuperparaboloidWithBase(params['theta'], params['k'])
showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))
mlab.show()


# if free_space_penalty1>0.1: #Probably is a open shape, lets model it using a superparaboloid
#         print("lets try to model it using a superparaboloid")
        # points_centered,theta,_, p0, sigma2, t0 = initialize_theta_superparaboloids_pytorch(points, table_normal, False)
        # optimizer_superparaboloid = torch.optim.Adam([theta], lr=1e-3, weight_decay=0.001)
        # k = torch.tensor(0.6)
        # k = torch.nn.Parameter(k)

        # for step in range(300):  # or until convergence
        #   optimizer_superparaboloid.zero_grad()

        #   loss, p, distances = total_loss(points_centered, theta, p0, 0.0, sigma2, number_of_rays, number_samples_per_ray, current_ray_samples_flat, k)

        #   loss_per_iteration.append(loss.item())  # Save it for plotting later
        #   loss.backward()
        #   optimizer_superparaboloid.step()

        #   # Clamp theta values to stay valid
        #   with torch.no_grad():
              
        #       iter_sigma+=1
              

        #       if iter_sigma == 10:
        #           iter_sigma = 0
        #           fitting_error = torch.sum(p*(distances)**2)
                  
        #           sigma2_new = 2 * torch.sum(p * distances**2) / (3 * torch.sum(p) + 1e-8)
        #           sigma2 = 0.8 *sigma2+0.2*sigma2_new
                  
        #           print("sigma2: ", sigma2)

                  
        #       # Clamp e1 and e2 between [0.1, 2.0]
        #       theta[0].clamp_(0.01, 2.0)  # e1
        #       theta[1].clamp_(0.01, 2.0)  # e2
              
        #       # Clamp semi-axes a1, a2, a3 to be positive
        #       theta[2:5].clamp_(0.001,1.5)  # a1, a2, a3 positive
              
        #       # Optionally clamp rotation angles between [-pi, pi]
        #       theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi
              
              
              
              
        #   if step % 100 == 0:
        #       print(f"Step {step}: Loss = {loss.item()}")
        # print(theta)
        # # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        # theta_np = theta.detach().cpu().numpy()
        # theta = theta.clone()  # (optional if you're not sure)
        # theta[8:11] = theta[8:11] + t0
        # theta_np = theta.detach().cpu().numpy()
        # k_np = k.detach().cpu().numpy()

      
        # thetas = torch.tensor(theta_np, dtype=torch.float32, device='cuda')  # or 'cpu' if no GPU
        # # Roti = build_rotation_matrix(thetas[5:8])
        # # print("Roti: ", Roti)
        
        # points_centered_np = points_centered.detach().cpu().numpy()
        # # print(p)
        # p = torch.clamp(p, max=1.0)
        
        # indices = torch.nonzero(p > 0.9, as_tuple=False).view(-1)        
        
        # all_indices = np.arange(cluster.shape[0])
        
        # selected_indices_good = indices.cpu().numpy()
        
        # remaining_indices = np.setdiff1d(all_indices, selected_indices_good)
        
        # indices_bad = torch.nonzero(p <= 0.2, as_tuple=False).view(-1)
        # print("indices  bad: ", indices_bad)
        # selected_indices_bad = indices_bad.cpu().numpy()
        # print("indices bad: ", selected_indices_bad)
        # remaining_indices1 = np.setdiff1d(remaining_indices, selected_indices_bad)
    
        # all_params_modeled[i] = {"type": "superparaboloid", "theta": theta_np, "k": k_np}
        # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        # showPoints(cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
        # if selected_indices_bad.size >0:
        #     showPoints(cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
        # showPoints(cluster[remaining_indices1], scale_factor=0.01, color=(0,0,1))
        # showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
        # showTaperedSuperparaboloidWithBase(theta_np, k_np)
        # mlab.show()
