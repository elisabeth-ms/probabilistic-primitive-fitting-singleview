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


def showSuperquadrics(x, threshold = 1e-2, num_limit = 10000, arclength = 0.02):
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
    e2 = torch.tensor(1.0, device=device)
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
    e1 = torch.tensor(1.0, device=device)
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

    dz_local = torch.min(points_centered[:,2])

    # Make it a learnable parameter
    theta = torch.nn.Parameter(theta_init)
    dz_local = torch.nn.Parameter(dz_local)
    
    return points_centered,theta, scale, p0, sigma2, t0, dz_local

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


def fitting_loss(points, theta, p0, sigma2, normals_est, k):
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
    
    print("negative: ", torch.sum(z_<0))
    # # print("para: ", F_val)
    
    r_norm = torch.norm(points_local, dim=1)

    distances =  r_norm*torch.abs(F_val)

    
    
    

    

    # r_norm = torch.norm(points_local, dim=1)

    # distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
    # distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
    print("Distances: ", distances)

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

    # p = weights
    return loss+z_penalty.sum(), p, distances


def total_loss(points, theta, p0, weight_compactness, sigma2, normals_est, number_of_rays, number_samples_per_ray, ray_samples_flat, k):
    fit, p, distances = fitting_loss(points, theta, p0, sigma2, normals_est, k)
    
    print("fit: ", fit)

    
    
    # values = superquadric_function(ray_samples_flat, theta)
    # print("values: ")
    # inside = values < 1.0
    # penalty = inside.float().sum() / len(ray_samples_flat)
    
    # print("samples: ", len(ray_samples_flat))
    # print("inside: ", inside.float().sum())
    # print("penaly1:", penalty)

    inside_score = superquadric_function(ray_samples_flat, theta)
    soft_inside = torch.sigmoid(-(inside_score - 1) * 10)  # sharpness ≈ 10–100
    # penalty = soft_inside.sum()    
    penalty = soft_inside.view(number_of_rays, -1).max(dim=1).values.mean()
    print("penalty: ", penalty)
    # Reshape back to (N, S) and check if any point along ray is inside
    # inside_any = inside.view(N, samples_per_ray).any(dim=1)  # (N,)
    
    compact = compactness_loss(points, p, theta, weight_compactness)
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
    

    return fit+0.0*penalty, p, distances
  
  
  
  

point_cloud = read_ply("data/glass.ply")
point_cloud = remove_close_points(point_cloud, 0.01)


all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor


# camera_origin = torch.zeros((1, 3), device=points.device)   # (1, 3)
# dirs = points - camera_origin                               # (N, 3)
# dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)         # Normalize each ray


# num_samples = 50
# t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=points.device)  # (50,)
# t_vals = t_vals.view(1, num_samples, 1)                                     # (1, 50, 1)

# ray_samples = camera_origin.view(1, 1, 3) + t_vals * dirs.view(-1, 1, 3)    # (N, 50, 3)
# ray_samples_flat = ray_samples.view(-1, 3)                                  # (N*50, 3)

# number_samples_per_ray = 50
# # Generate sample ratios [0, 1]
# t = torch.linspace(0.0, 1.0, number_samples_per_ray, device=all_points.device)  # shape (S,)
    
# ray_points = all_points[:, None, :] * t[None, :, None]  # (N, S, 3)

# print("points ", point_cloud[:5,:])
# print("ray points: ", ray_points[:5, :])

# # Flatten to (N*S, 3) to batch evaluate superquadric implicit function
# ray_samples_flat = ray_points.reshape(-1, 3)



point_cloud = filter_by_z(point_cloud, -np.inf, 0.94)



filtered_points, plane_points, plane_model = remove_largest_plane(point_cloud, distance_threshold=0.016)
print("plane model: ", plane_model)

table_normal = torch.tensor(plane_model[:3], dtype=torch.float32, device='cuda')




fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
showTaperedSuperparaboloidWithBase(
    x=[1.5, 1.0, 0.05, 0.05, 0.1, 1.5809, -0.21128, -0.98208, 0.00227, -0.00818, -0.00075],
    r_offset=0.5  # Try 0.02–0.08 for visible widening
)




# showPoints(filtered_points, scale_factor=0.01)
# showPoints(np.array([[0,0,0]]), 0.1, (1,0,0))

mlab.show()
p= None
kmeans = KMeans(n_clusters=4).fit(filtered_points)
clustering = DBSCAN(eps=0.03, min_samples=4).fit(filtered_points)
n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
print(f"Number of clusters: {n_clusters}")

camera_origin = torch.zeros_like(all_points)  # shape (N, 3), all (0,0,0)
directions = all_points - camera_origin  # or just points if origin is (0,0,0)

# Now sample along these rays
number_samples_per_ray = 200
number_of_rays = all_points.shape[0]

t_vals = torch.linspace(0, 0.99, number_samples_per_ray, device=all_points.device)  # go slightly past the surface
ray_points = camera_origin[:, None, :] + t_vals[None, :, None] * directions[:, None, :]
ray_samples_flat = ray_points.reshape(-1, 3)

k = torch.tensor(0.6)
k = torch.nn.Parameter(k)


for i in range(n_clusters):
    cluster = filtered_points[clustering.labels_ == i]
    loss_per_iteration = []
    



    points = torch.tensor(cluster, dtype=torch.float32, device='cuda')  # or 'cpu' if no GPU
    
    # camera_origin = torch.zeros_like(points)  # shape (N, 3), all (0,0,0)
    # directions = points - camera_origin  # or just points if origin is (0,0,0)

    # # Now sample along these rays
    # number_samples_per_ray = 50
    # number_of_rays = cluster.shape[0]

    # t_vals = torch.linspace(0, 1.0, number_samples_per_ray, device=points.device)  # go slightly past the surface
    # ray_points = camera_origin[:, None, :] + t_vals[None, :, None] * directions[:, None, :]
    # ray_samples_flat = ray_points.reshape(-1, 3)
    
    # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    # showPoints(filtered_points, scale_factor=0.01)
    # ray_samples_flat_np = ray_samples_flat.detach().cpu().numpy()
    # showPoints(ray_samples_flat_np, scale_factor=0.005, color=(0,1,0))
    # mlab.show()
    
    # Initialize superquadric parameters randomly or based on prior knowledge
    points_centered,theta,_, p0, sigma2, t0 = initialize_theta_superparaboloids_pytorch(points, table_normal, False)
    
    current_ray_samples_flat = ray_samples_flat - t0
    print("Initial theta: ", theta)
    
    # sigma2 = torch.nn.Parameter(sigma2)
    
    optimizer = torch.optim.Adam([theta,k], lr=1e-3, weight_decay=0.001)

    iter_sigma = 0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster)

    # Estimate normals using a k-nearest neighbors search
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

    # Optionally orient normals consistently (e.g., towards camera)
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # Convert back to numpy array if needed
    normals = np.asarray(pcd.normals)

    normals = torch.tensor(normals, dtype=torch.float32, device='cuda')  # or 'cpu' if no GPU

    
    # # Wrap into a batched structure
    # pc = Pointclouds(points=[points])
    
    
    # normals = estimate_pointcloud_normals(pc, neighborhood_size=20, disambiguate_directions=True)

    # # normals is (1, N, 3) tensor
    # normals = normals[0]  # remove batch dim → (N, 3)

    

    for step in range(1000):  # or until convergence
        optimizer.zero_grad()

        # if sigma2>1e-4:
        loss, p, distances = total_loss(points_centered, theta, p0, 0.0, sigma2, normals, number_of_rays, number_samples_per_ray, current_ray_samples_flat, k)
        # else:
        #     loss, p, distances = total_loss(points_centered, theta, p0, 0.0, sigma2, normals_est=normals, ray_samples_flat=ray_samples_flat)
        
        loss_per_iteration.append(loss.item())  # Save it for plotting later
        loss.backward()
        print(theta.grad)
        optimizer.step()

        # Clamp theta values to stay valid
        with torch.no_grad():
            # print("iter: ", step)
            # print("sigma2: ", sigma2)
            # print("distances: ", distances)
            # print("loss", loss)
            # print("theta grad: ", theta.grad)
            # print("theta: ", theta)
            # print("prob: ", p)    # camera_origin = torch.zeros_like(points)  # shape (N, 3), all (0,0,0)
            
            iter_sigma+=1
            
            # Update sigma2 every N steps (or every step)
            # if i % 100 == 0:
            #     sigma2 = torch.mean(distances.detach()**2) / 3

            if iter_sigma == 10:
                iter_sigma = 0
                fitting_error = torch.sum(p*(distances)**2)
                # print("fittt: ", fitting_error)
                # sigma2 = 2*fitting_error / (3 * torch.sum(p))
                
                sigma2_new = 2 * torch.sum(p * distances**2) / (3 * torch.sum(p) + 1e-8)
                sigma2 = 0.8 *sigma2+0.2*sigma2_new
                
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
    print("k: ", k)
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    theta_np = theta.detach().cpu().numpy()
    theta = theta.clone()  # (optional if you're not sure)
    theta[8:11] = theta[8:11] + t0
    theta_np = theta.detach().cpu().numpy()
    k_np = k.detach().cpu().numpy()
    # theta_np[0] = 0.40621552
    # theta_np[1] = 1.77195719
    # theta_np[2] = 2.00324044
    # theta_np[3] = 1.94088239
    # theta_np[4] = 0.68314028
    # theta_np[5] = -2.31931324
    # theta_np[6] = -0.35531119
    # theta_np[7] = 1.00312802
    # theta_np[8] = -0.06891287
    # theta_np[9] = 0.17462756
    # theta_np[10] = -0.32216445
    
    thetas = torch.tensor(theta_np, dtype=torch.float32, device='cuda')  # or 'cpu' if no GPU
    # Roti = build_rotation_matrix(thetas[5:8])
    # print("Roti: ", Roti)
    
    points_centered_np = points_centered.detach().cpu().numpy()
    print(p)
    p = torch.clamp(p, max=1.0)
    indices = torch.nonzero(p > 0.9, as_tuple=False).squeeze()
    
    all_indices = np.arange(cluster.shape[0])
    
    selected_indices_good = indices.cpu().numpy()
    
    remaining_indices = np.setdiff1d(all_indices, selected_indices_good)
    
    indices_bad = torch.nonzero(p <= 0.2, as_tuple=False).view(-1)
    print("indices  bad: ", indices_bad)
    selected_indices_bad = indices_bad.cpu().numpy()
    print("indices bad: ", selected_indices_bad)
    remaining_indices1 = np.setdiff1d(remaining_indices, selected_indices_bad)
    



    
    # print("indices: ", indices)
    
    
    showPoints(cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
    if selected_indices_bad.size >0:
        showPoints(cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
    showPoints(cluster[remaining_indices1], scale_factor=0.01, color=(0,0,1))
    showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
    # showSuperquadrics(theta_np)
    # showSuperparaboloid(theta_np)
    showTaperedSuperparaboloidWithBase(theta_np, k_np)

    mlab.show()
    
    import matplotlib.pyplot as plt

    plt.plot(loss_per_iteration)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Iteration")
    plt.grid(True)
    plt.show()
