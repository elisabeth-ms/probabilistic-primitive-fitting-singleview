import torch
import plyfile
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from mayavi import mlab
from scipy.spatial import cKDTree
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
# from pytorch3d.structures import Pointclouds
# from pytorch3d.ops import estimate_pointcloud_normals
from matplotlib import cm

from tvtk.api import tvtk
from mayavi.sources.vtk_data_source import VTKDataSource


def interpolate_color(val):
    """
    Map a value in [0,1] to a color along a custom dark gradient (no white).
    Dark purple → blue → green → yellow → red
    """
    val = np.clip(val, 0, 1)
    
    if val < 0.25:
        # Purple (0, 0, 0.5) → Blue (0, 0, 1)
        t = val / 0.25
        return (0, 0, 0.5 + 0.5 * t)
    elif val < 0.5:
        # Blue (0, 0, 1) → Green (0, 1, 0)
        t = (val - 0.25) / 0.25
        return (0, t, 1 - t)
    elif val < 0.75:
        # Green (0, 1, 0) → Yellow (1, 1, 0)
        t = (val - 0.5) / 0.25
        return (t, 1, 0)
    else:
        # Yellow (1, 1, 0) → Red (1, 0, 0)
        t = (val - 0.75) / 0.25
        return (1, 1 - t, 0)

def show_points_colored(points, probs, scale_factor=0.02):
    for i in range(len(points)):
        color = interpolate_color(probs[i])
        
        # Generate values between 0 and 1

        # Get corresponding colors from the Viridis colormap
        color = cm.viridis(probs[i])[:3]
        mlab.points3d(points[i, 0], points[i, 1], points[i, 2],
                      scale_factor=scale_factor,
                      color=color,
                      mode='sphere')

def show_points_manual_rgb(points, p, scale_factor=0.02):
    """
    Show 3D points with fully custom RGB colors (no LUTs).
    p: array of values in [0, 1] controlling the color.
    """
    # Normalize and define color mapping: from purple (0) to yellow (1)
    p = np.clip(p, 0.0, 1.0)
    r = p
    g = p
    b = 1.0 - p
    colors = np.vstack((r, g, b)).T

    for i in range(len(points)):
        val = p[i]
        # Define your own RGB mapping here, e.g., purple (low) to yellow (high)
        r = val
        g = val
        b = 1.0 - val
        mlab.points3d(points[i, 0], points[i, 1], points[i, 2],
                      scale_factor=scale_factor,
                      color=(r, g, b),
                      mode='sphere')  # Optional: '2dcircle', 'cube', etc.





def show_points_with_custom_color(points, p, scale_factor=0.02):
    """
    Visualize 3D points colored by inlier probability p ∈ [0, 1],
    where p = 0 is dark red and p = 1 is bright yellow.
    """
    # Ensure p is in [0, 1]
    p = np.clip(p, 0.0, 1.0)

    # Custom RGB mapping: e.g., from dark red → yellow
    # You can adjust this mapping to suit your contrast needs
    r = 1.0 * np.ones_like(p)
    g = p  # increases with p
    b = np.zeros_like(p)

    # Stack into RGB array
    colors = np.vstack((r, g, b)).T

    mlab.figure(bgcolor=(1, 1, 1))  # white background

    pts = mlab.points3d(
        points[:, 0], points[:, 1], points[:, 2],
        scale_factor=scale_factor,
        color=(1, 1, 1),  # dummy color, overridden below
        mode='sphere'
    )

    # Manually set per-point RGB colors
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.module_manager.scalar_lut_manager.lut_mode = 'gray'  # disabled color map
    pts.mlab_source.dataset.point_data.scalars = None
    pts.mlab_source.dataset.point_data.vectors = colors
    pts.mlab_source.dataset.modified()



def showPoints(point, scale_factor=0.1, color =(1, 0, 0)):
    
    mlab.points3d(point[:, 0], point[:, 1], point[:, 2], scale_factor=scale_factor, color=color)

def custom_colormap_no_white(n=256):
    # Use a perceptually uniform colormap like viridis or magma
    base_cmap = cm.get_cmap('magma', n)  # Or 'viridis', 'plasma', etc.
    return (base_cmap(np.linspace(0, 1, n)) * 255).astype(np.uint8)

def lightgray_to_red_colormap(n=256):
    reds = cm.get_cmap('Reds', n)(np.linspace(0, 1, n))
    # Override low end to be light gray instead of white
    reds[:10, :3] = [0.7, 0.7, 0.7]  # light gray
    return (reds * 255).astype(np.uint8)

def showBasedOnProbabilityPoints(points, p, scale_factor=0.1):
    """
    Show 3D points with a color gradient based on Z values.
    """
    mlab.view(azimuth=0.0, elevation=0.0, distance=2)


    for pi in p:
      if pi<0.05:
        pi=0.05
    
    print("Min p:", p.min(), "Max p:", p.max())
    print("Any NaNs?", np.isnan(p).any())
    pts = mlab.points3d(
        points[:, 0], points[:, 1], points[:, 2],
        p,
        scale_factor=scale_factor,
        mode='sphere',
        colormap='viridis', 
    )
    # Force LUT to cover the full scalar range manually
    lut_manager = pts.module_manager.scalar_lut_manager
    lut_manager.use_default_range = False
    lut_manager.data_range = (0.0, 1.0)

    # Optional: ensure opacity is 1 for all scalars
    lut = lut_manager.lut.table.to_array()
    lut[:, -1] = 255  # Set alpha to 255 (fully opaque)
    lut_manager.lut.table = lut

    # # Force the scalar lookup table to cover full range
    # pts.module_manager.scalar_lut_manager.use_default_range = False
    # pts.module_manager.scalar_lut_manager.data_range = (0.0, 1.0)
    
    # pts.module_manager.scalar_lut_manager.lut.table = custom_colormap_no_white()
    mlab.colorbar(title='Inlier probability $p$', orientation='vertical')


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

def showTaperedSuperparaboloidWithBase(x, r_offset=0.01, threshold=1e-2, num_limit=10000, arclength=0.02, color=(1.0, 0.5, 0.0)):
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

    mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=0.8)


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


def showSuperquadrics(x, threshold = 1e-2, num_limit = 10000, arclength = 0.02, color=(1, 0.5, 0)):
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
            point_temp = point_temp@RotM.T + get_translation_numpy(x)

            x_mesh[m, n] = point_temp[0]
            y_mesh[m, n] = point_temp[1]
            z_mesh[m, n] = point_temp[2]
    
    mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=1.0)



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

def initialize_theta_superparaboloids_pytorch(points, table_normal, rescale=False):
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
    
    

    print("R0: ", R0)
    # print("R0: ", R0)
    # 5. Rotate points
    points_rot0 = points_centered @ R0

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
    e1 = torch.tensor(2.0, device=device)
    e2 = torch.tensor(1.0, device=device)
    a1, a2, a3 = s0[0], s0[1], s0[2]

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
    

    term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
    term2 = (torch.abs(z_)**(2/e1))
    inside_outside = term1 + term2
    
    
    return inside_outside




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
    points_local = (points - t)@ Rot
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
    w=0.1
    const = (w * p0) / (c * (1 - w))
    
    dist_term = torch.exp(-1 / (2 * sigma2) * distances ** 2)
    


    # Final inlier probability with normal guidance
    p = dist_term / (const + dist_term)
    p = torch.clamp(p, min=1e-10)
    
    p = torch.clamp(p, min=1e-10)
    # print("prob: ", p)
    
    scaled_distances = distances
    
    cost = p*scaled_distances**2
    
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
    
    allowed_margin = 0.02  # small tolerance below point cloud
    shape_base_z = t[2]
    min_z_points = points_local[:, 2].min()

    drop_penalty = torch.relu(min_z_points - shape_base_z - allowed_margin) ** 2 * 10.0
    
    z_max = (z_vals).max()
    # print("z_max: ", z_max)
    # print("a3: ", a3)
    extent_penalty = torch.relu(a3-z_max) ** 2 * 10.0

    # print("extent_penalty: ", extent_penalty)

    return loss+1.0*z_penalty.sum() + drop_penalty+ extent_penalty, p, distances





def superquadric_total_loss(points, theta, p0, weight_compactness, sigma2, number_of_rays, number_samples_per_ray, ray_samples_flat):
    
    # Assume points: (N, 3)
    e1, e2 = theta[0], theta[1]
    a1, a2, a3 = theta[2], theta[3], theta[4]
    rot = theta[5:8]  # Euler angles
    t = theta[8:11]   # translation vector
    print("theta: ", theta)
    # print("theta: ", theta)    
    # Build rotation matrix from Euler angles
    Rot = build_rotation_matrix(rot)

    
    # Transform points
    points_local = points@Rot - t @ Rot
    # Normalize by semi-axes
    x_ = points_local[:, 0] / a1
    y_ = points_local[:, 1] / a2
    z_ = points_local[:, 2] / a3
    
    term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
    term2 = (torch.abs(z_)**(2/e1))
    inside_outside = term1 + term2
    
    print("inside_outside: ", inside_outside)
    # inside_outside = (
    # (torch.abs(x_).pow(2/e2) + torch.abs(y_).pow(2/e2)).pow(e2/e1)
    # + torch.abs(z_).pow(2/e1))

    
    r_norm = torch.norm(points_local, dim=1)

    distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
    
    c = (2 * torch.pi * sigma2) ** (- 3 / 2)
    w=0.05
    const = (w * p0) / (c * (1 - w))
    
    dist_term = torch.exp(-1 / (2 * sigma2) * distances ** 2)
    

    # Final inlier probability with normal guidance
    p = dist_term / (const + dist_term)
    p = torch.clamp(p, min=1e-10)
    
    
    print("prob: ", p)
    
    scaled_distances = distances
    
    fit_cost = p*scaled_distances**2
    
    fit_loss = fit_cost.sum()
    
        
    # print("fit: ", fit_loss)


    inside_score = superquadric_function(ray_samples_flat, theta)
    soft_inside = torch.sigmoid(-(inside_score - 1) * 10)  # sharpness ≈ 10–100
    # penalty = soft_inside.sum()    
    free_space_penalty = soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()
    # print("free space penalty: ", free_space_penalty)


    

    return fit_loss + 1.0*free_space_penalty, p, distances
  



def total_loss(points, theta, p0, weight_compactness, sigma2, number_of_rays, number_samples_per_ray, ray_samples_flat, k):
    fit, p, distances = fitting_loss(points, theta, p0, sigma2, k)

    

    return fit, p, distances
  


# Plots for the shapes on the article

# fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
# theta_box = [0.2, 0.2, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# showSuperquadrics(theta_box)
# theta_cylinder = [0.2, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]
# showSuperquadrics(theta_cylinder)
# theta_ellipsoid = [1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0]
# showSuperquadrics(theta_ellipsoid)
# theta_octahedron = [2.0, 0.2, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0]
# showSuperquadrics(theta_octahedron)
# mlab.show()

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
theta_1 = [2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
r0_1 = 0.1
showTaperedSuperparaboloidWithBase(theta_1, r0_1)

# # theta_4 = [1.0, 1.8, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# # r0_4 = 0.1
# # showTaperedSuperparaboloidWithBase(theta_4, r0_4)

# theta_2 = [1.5, 1.0, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# r0_2 = 0.8
# showTaperedSuperparaboloidWithBase(theta_2, r0_2)

# theta_3 = [0.5, 1.2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# r0_3 = 0.5
# showTaperedSuperparaboloidWithBase(theta_3, r0_3)



mlab.show()

point_cloud = read_ply("data/objects7.ply")
point_cloud = remove_close_points(point_cloud, 0.006)

point_cloud = filter_by_z(point_cloud, -np.inf, 1.94)

all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor


# def print_camera_view(scene):
#     azimuth, elevation, distance, focalpoint = mlab.view()
#     roll = mlab.roll()
    
#     print("Azimuth:", azimuth)
#     print("Elevation:", elevation)
#     print("Distance:", distance)
#     print("Focal Point:", focalpoint)
#     print("Roll:", roll)
#     print("-" * 50)

# @mlab.animate
# def live_view_monitor():
#     while True:
#         print_camera_view(mlab.gcf())
#         yield



fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
mlab.view(azimuth=-127, elevation=171, distance=1.26, focalpoint=(-0.04977768,0.06968273,0.79997664), roll=-178.84)
showPoints(point_cloud, scale_factor=0.0025, color=(0.6,0.6,0.6))
# live_view_monitor()
# Lets plot the fitting 
theta1 = [0.43085322,  0.3334181 ,  0.1755724 ,  0.04760716,  0.10751509,
        1.5580609 , -0.51621795, -1.2906431 ,  0.20583807, -0.04588583,
        0.7744524]
showSuperquadrics(theta1, color=(0.518, 0.780, 0.667))

theta2 = [ 0.34291595,  0.46497834,  0.13244838,  0.03630781,  0.03517469,
        1.5672905 , -0.5343182 , -1.9973316 , -0.16091624,  0.07729545,
        0.67674136]
showSuperquadrics(theta2,  color=(0.161, 0.431, 0.529))

theta3 = [0.6378871 ,  0.6546559 ,  0.0380326 ,  0.01433728,  0.0123732 ,
        -1.5435802 ,  0.53311396, -1.134063  , -0.16173   , -0.07120007,
        0.59157586]
showSuperquadrics(theta3,  color=(0.161, 0.431, 0.529))

theta4 = [1.8046732 ,  1.1858393 ,  0.06681335,  0.07090971,  0.12584554,
        1.5481961 ,  1.0237458 , -3.1111438 , -0.09766323,  0.062757  ,
        0.92298627]
k4=0.8
showTaperedSuperparaboloidWithBase(theta4, k4, color=(1.0, 1.0, 0.169))

theta5 = [0.39294407,  0.6082235 ,  0.01773949,  0.01672928,  0.04021339,
        1.5518134 , -0.57437634, -1.8318247 ,  0.22142781,  0.05443245,
        0.6215799]
showSuperquadrics(theta5, color=(0.294, 0.071, 0.412))

theta6 = [0.5046233 ,  0.8154403 ,  0.03351911,  0.02931844,  0.03654086,
        1.0883882 , -0.36557555, -1.4172481 ,  0.21653208,  0.16831969,
        0.6861477]
showSuperquadrics(theta6,  color=(0.294, 0.071, 0.412))

theta7 = [0.39621717,  0.41882688,  0.03269098,  0.00909346,  0.01354168,
        -1.835317  ,  0.5494602 , -1.3893318 ,  0.22361767,  0.09255629,
        0.6463623]
showSuperquadrics(theta7,  color=(0.294, 0.071, 0.412))

theta8= [0.43654877,  0.3062408 ,  0.04732962,  0.03263379,  0.03109895,
        1.5629718 , -0.52601504,  1.3117464 , -0.06861866,  0.1902022 ,
        0.568279]
showSuperquadrics(theta8, color=(0.580, 0.847, 0.251))

theta9 = [1.8730378 ,  1.0435834 ,  0.02369763,  0.0238652 ,  0.14368284,
        1.5949976 ,  1.0735142 , -3.118101  ,  0.05201951,  0.20876324,
        0.6709036]
k9=0.8
showTaperedSuperparaboloidWithBase(theta9, k9)

mlab.show()

blocks = [
    [0.10146605, 0.08979341, 0.13970061, 0.03661494, 0.11091885, -1.56827101, 0.5124012, -1.84647234, 0.20674956, -0.07030025, 0.74962691],
    [0.10156348, 0.40713203, 0.15034812, 0.03152425, 0.03357434, 1.57018739, -0.520114, 1.13127339, -0.16040713, 0.07441799, 0.67079592],
    [0.92511971, 1.29361505, 0.07981475, 0.02545856, 0.02616961, 1.55060284, -0.5646786, -0.64815516, -0.16460688, -0.04343885, 0.61577923],
    [0.98330942, 0.98392758, 0.10951403, 0.10619891, 0.10841419, -1.93637997, -0.79104754, -1.8518818, -0.10256101, -0.03034356, 0.86605207],
    [0.03770605, 0.8767673, 0.07587676, 0.04367812, 0.04903468, 1.75918479, 0.09641413, -1.28667489, 0.18986037, 0.09976493, 0.64044212],
    [0.16963001, 0.34912939, 0.01359951, 0.03792285, 0.07608895, 1.47430028, 1.01702226, -0.2084636, 0.21957674, 0.10073388, 0.65537968],
    [0.99736174, 0.866563, 0.10277332, 0.0397981, 0.04076992, 1.5726279, -0.50088703, -1.33331975, 0.05181471, 0.09999471, 0.61220761],
    [0.72178447, 0.09768885, 0.04682529, 0.04540695, 0.03527874, 1.56807535, -0.5197711, 1.40234277, -0.06979633, 0.18399139, 0.57851643]
]

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
mlab.view(azimuth=-127, elevation=171, distance=1.26, focalpoint=(-0.04977768,0.06968273,0.79997664), roll=-178.84)
showPoints(point_cloud, scale_factor=0.0025, color=(0.6,0.6,0.6))
# live_view_monitor()
# Lets plot the fitting 

showSuperquadrics(blocks[0], color=(0.518, 0.780, 0.667))


showSuperquadrics(blocks[1],  color=(0.161, 0.431, 0.529))


showSuperquadrics(blocks[2],  color=(0.161, 0.431, 0.529))


showSuperquadrics(blocks[3], color=(1.0, 1.0, 0.169))


showSuperquadrics(blocks[4], color=(0.294, 0.071, 0.412))


showSuperquadrics(blocks[5],  color=(0.294, 0.071, 0.412))


showSuperquadrics(blocks[6], color=(0.580, 0.847, 0.251))


showSuperquadrics(blocks[7])

mlab.show()

filtered_points, plane_points, plane_model = remove_largest_plane(point_cloud, distance_threshold=0.08)
print("plane model: ", plane_model)

table_normal = torch.tensor(plane_model[:3], dtype=torch.float32, device='cuda')




p= None
kmeans = KMeans(n_clusters=4).fit(filtered_points)
clustering = DBSCAN(eps=0.03, min_samples=3).fit(filtered_points)
n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(f"Number of clusters: {n_clusters}")

camera_origin = torch.zeros_like(all_points)  # shape (N, 3), all (0,0,0)
directions = all_points - camera_origin  # or just points if origin is (0,0,0)

# Now sample along these rays
number_samples_per_ray = 100
number_of_rays = all_points.shape[0]

t_vals = torch.linspace(0.0, 0.99, number_samples_per_ray, device=all_points.device)  # go slightly past the surface
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


def fit_shape_to_cluster(cluster_points_np, superquadric = True):
  

    # loss_per_iteration = []
    
    points = torch.tensor(cluster_points_np, dtype=torch.float32, device='cuda')

    points_centered,theta,_, p0, sigma2, t0 = initialize_theta_pytorch(points, False)

    camera_position = torch.tensor([0.0, 0.0, 0.0], device=all_points.device)  # shape (3,)
    cluster_vecs = points - camera_position  # shape (N, 3)
    cluster_vecs = torch.nn.functional.normalize(cluster_vecs, dim=1)
    
    center_dir = torch.mean(cluster_vecs, dim=0)
    center_dir = center_dir / torch.norm(center_dir)
    
    cos_angles = (cluster_vecs @ center_dir)
    max_angle = torch.acos(torch.clamp(cos_angles.min(), -1.0, 1.0))  # in radians
    
    margin = 10 * torch.pi / 180  # radians
    final_cone_angle = max_angle + margin
    cos_thresh = torch.cos(final_cone_angle)

    camera_origin = torch.zeros_like(all_points)  # shape (N, 3), all (0,0,0)

    all_vecs = all_points - camera_position
    all_vecs = torch.nn.functional.normalize(all_vecs, dim=1)

    mask = (all_vecs @ center_dir) > cos_thresh
    points_in_cone = all_points[mask]
    
    camera_origin = torch.zeros_like(points_in_cone)  # shape (N, 3), all (0,0,0)
    directions = points_in_cone - camera_origin  # or just points if origin is (0,0,0)

    # Now sample along these rays
    number_samples_per_ray = 200
    number_of_rays = points_in_cone.shape[0]

    t_vals = torch.linspace(0.2, 0.95, number_samples_per_ray, device=points_in_cone.device)  # go slightly past the surface
    ray_points = camera_origin[:, None, :] + t_vals[None, :, None] * directions[:, None, :]
    
    print("len ray: ", ray_samples_flat.shape)
    
    current_ray_samples_flat = ray_points.reshape(-1, 3) - t0
    
    current_ray_samples_flat_np = current_ray_samples_flat.detach().cpu().numpy()

    # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    # showPoints(current_ray_samples_flat_np, scale_factor=0.001, color=(0,1,0))
    # mlab.show()


    
    print("len ray: ", current_ray_samples_flat.shape)

    # current_ray_samples_flat = ray_samples_flat - t0
    print("Initial theta: ", theta)
    
    # sigma2 = torch.nn.Parameter(sigma2)
    iter_sigma = 0

    if superquadric:
        optimizer = torch.optim.Adam([theta], lr=1e-4, weight_decay=0.001)

        tolerance = 1e-6  # or something like 1e-4 depending on your scale
        patience = 100    # number of steps with small change before stopping
        no_improve_steps = 0



        for step in range(1500):  # or until convergence
            optimizer.zero_grad()

            loss, p, distances = superquadric_total_loss(points_centered, theta, p0, 0.0, sigma2, number_of_rays, number_samples_per_ray, current_ray_samples_flat)

            loss_per_iteration.append(loss.item())  # Save it for plotting later
            if not torch.isfinite(loss):
                print(f"Loss became NaN at step {step}, stopping.")
                break
            loss.backward()
            optimizer.step()

            # Clamp theta values to stay valid
            with torch.no_grad():

                
                iter_sigma+=1
                

                if iter_sigma == 15:
                    iter_sigma = 0
                    sigma2_new = 2 * torch.sum(p * distances**2) / (3 * torch.sum(p) + 1e-8)
                    sigma2 = 0.9 *sigma2+0.1*sigma2_new
                    sigma2 = torch.clamp(sigma2, min=1e-8, max=1e2)                    

                    
                # Clamp e1 and e2 between [0.1, 2.0]
                theta[0].clamp_(0.01, 2.0)  # e1
                theta[1].clamp_(0.01, 2.0)  # e2
                
                # Clamp semi-axes a1, a2, a3 to be positive
                theta[2:5].clamp_(0.001,1.5)  # a1, a2, a3 positive
                
                # Optionally clamp rotation angles between [-pi, pi]
                theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi
            if step>1:
                loss_change = abs(loss_per_iteration[-1] - loss_per_iteration[-2])
                if loss_change < tolerance:        # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        # showPoints(current_cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
        # if selected_indices_bad.size >0:
        #     showPoints(current_cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
        # showPoints(current_cluster[remaining_indices], scale_factor=0.01, color=(0,0,1))
        # showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
        # showSuperquadrics(theta_np)
        # mlab.show()
        
                    no_improve_steps += 1
                else:
                    no_improve_steps = 0

                if no_improve_steps >= patience:
                    print(f"Early stopping at step {step} (Δloss < {tolerance})")
                    break
                
                
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item()}")
                
                
        print(theta)
        theta_np = theta.detach().cpu().numpy()
        theta = theta.clone()  # (optional if you're not sure)
        theta[8:11] = theta[8:11] + t0
        theta_np = theta.detach().cpu().numpy()

        
        
        points_centered_np = points_centered.detach().cpu().numpy()

        indices = torch.nonzero(p > 0.85, as_tuple=False).squeeze()
        
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

            t_vals1 = torch.linspace(0, 0.95, number_samples_per_ray1, device=points.device)  # go slightly past the surface
            ray_points1 = camera_origin1[:, None, :] + t_vals1[None, :, None] * directions1[:, None, :]
            ray_samples_flat1 = ray_points1.reshape(-1, 3)

            inside_score1 = superquadric_function(ray_samples_flat1, theta)
            soft_inside1 = torch.sigmoid(-(inside_score1 - 1) * 10)  # sharpness ≈ 10–100
            # penalty = soft_inside.sum()    
            free_space_penalty1 = soft_inside1.view(number_of_rays1, -1).max(dim=1).values.mean()
            print("After optimization checking rays: ", free_space_penalty1)
    else:
        points_centered,theta,_, p0, sigma2, t0 = initialize_theta_superparaboloids_pytorch(points, table_normal, False)
        optimizer_superparaboloid = torch.optim.Adam([theta], lr=1e-3, weight_decay=0.001)
        k = torch.tensor(0.2)
        k = torch.nn.Parameter(k)

        for step in range(1200):  # or until convergence
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
              theta[0].clamp_(0.01, 2.0)  # e1
              theta[1].clamp_(0.01, 2.0)  # e2
              
              # Clamp semi-axes a1, a2, a3 to be positive
              theta[2:5].clamp_(0.001,1.5)  # a1, a2, a3 positive
              
              # Optionally clamp rotation angles between [-pi, pi]
              theta[5:8] = (theta[5:8] + torch.pi) % (2 * torch.pi) - torch.pi
              
              
              
              
          if step % 100 == 0:
              print(f"Step {step}: Loss = {loss.item()}")
        print(theta)
        # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        theta_np = theta.detach().cpu().numpy()
        theta = theta.clone()  # (optional if you're not sure)
        theta[8:11] = theta[8:11] + t0
        theta_np = theta.detach().cpu().numpy()
        k_np = k.detach().cpu().numpy()



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



    return theta_np,k_np, selected_indices_good,remaining_indices1, selected_indices_bad, free_space_penalty1, p


    # import matplotlib.pyplot as plt

    # plt.plot(loss_per_iteration)
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title("Loss vs Iteration")
    # plt.grid(True)
    # plt.show()
all_params_modeled = {}
idx = 0
for i in range(n_clusters):
    loss_per_iteration = []
    
    cluster = filtered_points[clustering.labels_ == i]
    current_cluster = cluster
    while current_cluster.shape[0]>40:
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        mlab.view(azimuth=-127, elevation=171, distance=1.26, focalpoint=(-0.04977768,0.06968273,0.79997664), roll=-178.84)
        showPoints(current_cluster, scale_factor=0.0025, color=(0.6,0.6,0.6))
        mlab.show()
        
        theta_np, k_np, selected_indices_good, remaining_indices, selected_indices_bad, free_space_penalty,p = fit_shape_to_cluster(current_cluster, True)
        all_params_modeled[idx] = {"cluster": i,"type": "superquadric", "theta": theta_np, "k": 0, "free_space_penalty":free_space_penalty, "indices_good": selected_indices_good}
        idx +=1
        
        p_np = p.detach().cpu().numpy()

        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        showPoints(current_cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
        if selected_indices_bad.size >0:
            showPoints(current_cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
        showPoints(current_cluster[remaining_indices], scale_factor=0.01, color=(0,0,1))
        showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
        showSuperquadrics(theta_np)
        mlab.show()
        
        # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        # mlab.view(azimuth=-127, elevation=171, distance=1.26, focalpoint=(-0.04977768,0.06968273,0.79997664), roll=-178.84)
        # show_points_colored(current_cluster,p_np, scale_factor=0.005)
        # showPoints(point_cloud, scale_factor=0.0025, color=(0.6,0.6,0.6))
        # showSuperquadrics(theta_np)
        # mlab.show()
        

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
              theta_np, k_np, selected_indices_good, remaining_indices, selected_indices_bad1, free_space_penalty,p = fit_shape_to_cluster(current_cluster[params["indices_good"]], False)
              params["type"] = "superparabolid"
              params["theta"] = theta_np
              params["k"] = k_np
              params["free_space_penalty"]=0
              # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
              # mlab.view(azimuth=-127, elevation=171, distance=1.26, focalpoint=(-0.04977768,0.06968273,0.79997664), roll=-178.84)
              # show_points_colored(current_cluster,p_np, scale_factor=0.005)
              # showPoints(point_cloud, scale_factor=0.0025, color=(0.6,0.6,0.6))
              # showTaperedSuperparaboloidWithBase(theta_np,k_np)
              # mlab.show()
        current_cluster = current_cluster[selected_indices_bad]
        
    print(all_params_modeled)
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    mlab.view(azimuth=-127, elevation=171, distance=1.26, focalpoint=(-0.04977768,0.06968273,0.79997664), roll=-178.84)

    for idx,params in all_params_modeled.items():
        if params["type"] == "superquadric":
          showSuperquadrics(params['theta'])
        else:
          showTaperedSuperparaboloidWithBase(params['theta'], params['k'])
    showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))
    mlab.show()


#       showTaperedSuperparaboloidWithBase(params['theta'], params['k'])
# showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))
# mlab.show()



print(all_params_modeled)

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
mlab.view(azimuth=-127, elevation=171, distance=1.26, focalpoint=(-0.04977768,0.06968273,0.79997664), roll=-178.84)
for idx,params in all_params_modeled.items():
    if params["type"] == "superquadric":
      showSuperquadrics(params['theta'])
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
