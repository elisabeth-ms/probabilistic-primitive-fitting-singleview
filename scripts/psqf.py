import torch
import plyfile
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from mayavi import mlab
from scipy.spatial import cKDTree
import torch.nn.functional as F

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
    # Convert to Open3D PointCloud
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

    return remaining_points, plane_points

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
    
    sigma2 = V ** (1 / 3) / 10

    # 6. Estimate semi-axes
    s0 = torch.median(points_rot0.abs(), dim=0).values

    # translate the points to the center of mass
    # points_np = points.detach().cpu().numpy()
    # point_ = np.array(points_np, dtype=float)
    # t0_ = np.mean(point_, 0)
    # point_ = point_ - t0_
    
    # print("t0: ",t0_)

    # CovM = point_.T @ point_ / point_.shape[0]
    
    # print("CovM:", CovM)
    # EVal, EVec = np.linalg.eig(CovM)
    # idx = np.flip(np.argsort(EVal))
    # EigVec = EVec[:, idx]
    
    # print("EigVec: ", EigVec)
    
    # # eigen analysis for rotation initialization
    # RotM_ = np.array([-EigVec[:, 0], -EigVec[:, 2],
    #                    np.cross(EigVec[:, 0], EigVec[:, 2])]).T
    
    # RM_=R.from_matrix(RotM_)
    # print("RotM_: ", RM_)
    
    # print("euler: ", RM_.as_euler('ZYX'))

    # scale initialization
    # point_rot0_ = point_ @ RotM_
    # s0_ = np.median(np.abs(point_rot0_), 0)

    # print("s0: ", s0_)

    # 7. Initial parameters
    e1 = torch.tensor(1.0, device=device)
    e2 = torch.tensor(1.0, device=device)
    a1, a2, a3 = s0[0], s0[1], s0[2]

    print(a1, a2, a3)
    # 8. Get initial rotation as Euler angles
    euler_angles = rotation_matrix_to_euler(R0)
    # euler_angles[0] = 0.5
    # euler_angles[1] = 0.3
    # euler_angles[2] = 0.2
    # euler_angles[0] += torch.rand((), device=euler_angles.device) * torch.pi / 10
    # euler_angles[1] +=torch.rand((), device=euler_angles.device) * torch.pi / 10
    # euler_angles[2] +=torch.rand((), device=euler_angles.device) * torch.pi / 10

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

    #     # initialize configuration
    # x0 = np.array([1.0, 1.0, s0_[0], s0_[1], s0_[2],
    #               RM_.as_euler('ZYX')[0], RM_.as_euler('ZYX')[1], RM_.as_euler('ZYX')[2], 0, 0, 0])

    # print("x0: ",x0)
    # Make it a learnable parameter
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


def fitting_loss(points, theta, p0, sigma2):
    # Assume points: (N, 3)
    e1, e2 = theta[0], theta[1]
    a1, a2, a3 = theta[2], theta[3], theta[4]
    rot = theta[5:8]  # Euler angles
    t = theta[8:11]   # translation vector
    
    # print("theta: ", theta)    
    # Build rotation matrix from Euler angles
    Rot = build_rotation_matrix(rot)

    
    # Transform points
    points_local = (points - t) @ Rot
    points_local = points @ Rot - t @ Rot
    # Normalize by semi-axes
    x_ = points_local[:, 0] / a1
    y_ = points_local[:, 1] / a2
    z_ = points_local[:, 2] / a3
    
    # eps = 1e-6
    # e1 = F.softplus(theta[0]) + eps
    # e2 = F.softplus(theta[1]) + eps

    # x_ = points_local[:, 0] / (a1)
    # y_ = points_local[:, 1] / (a2)
    # z_ = points_local[:, 2] / (a3)


    term1 = (torch.abs(x_)**(2/e2) + torch.abs(y_)**(2/e2))**(e2/e1)
    term2 = (torch.abs(z_)**(2/e1))
    inside_outside = term1 + term2

    

    r_norm = torch.norm(points_local, dim=1)

    distances =  r_norm*torch.abs(inside_outside**(-e1/2)-1)
    
    # print("Distances: ", distances)

    c = (2 * torch.pi * sigma2) ** (- 3 / 2)
    w=0.1
    const = (w * p0) / (c * (1 - w))
    p = torch.exp(-1 / (2 * sigma2) * distances ** 2)
    p = p / (const + p)
    
    p = torch.clamp(p, min=1e-10)
    # print("prob: ", p)
    
    scaled_distances = distances
    
    cost = p*scaled_distances**2
    
    loss = cost.sum()
    
    # log_prob = -scaled_distances / (2 * sigma2) - 0.5 * torch.log(2*torch.pi * sigma2)
    # loss = -torch.mean(log_prob)
    
    # weights = torch.softmax(-distances / sigma2, dim=0)
    # # loss = torch.sum(weights * distances)  # weighted average distance
    
                    


    # p = weights
    return loss, p, distances

def total_loss(points, theta, p0, weight_compactness, sigma2):
    fit, p, distances = fitting_loss(points, theta, p0, sigma2=sigma2)
    compact = compactness_loss(points, p, theta, weight_compactness)
    print("compact: ", compact)
    return fit+compact, p, distances
point_cloud = read_ply("data/case_2.ply")
point_cloud = remove_close_points(point_cloud, 0.01)

filtered_points, plane_points = remove_largest_plane(point_cloud, distance_threshold=0.015)

filtered_points = filter_by_z(filtered_points, -np.inf, 1.1)



fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
showPoints(filtered_points, scale_factor=0.1)
mlab.show()
p= None
kmeans = KMeans(n_clusters=4).fit(filtered_points)


for i in range(5):
    cluster = filtered_points[kmeans.labels_ == i]
    loss_per_iteration = []

    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    showPoints(filtered_points, scale_factor=0.01)
    mlab.show()
    points = torch.tensor(cluster, dtype=torch.float32, device='cuda')  # or 'cpu' if no GPU
    print("p ",points[0])
    # Initialize superquadric parameters randomly or based on prior knowledge
    points_centered,theta,_, p0, sigma2, t0 = initialize_theta_pytorch(points, False)
    print("Initial theta: ", theta)
    
    # sigma2 = torch.nn.Parameter(sigma2)
    optimizer = torch.optim.Adam([theta], lr=1e-3)

    iter_sigma = 0

    for step in range(1000):  # or until convergence
        optimizer.zero_grad()

        loss, p, distances = total_loss(points_centered, theta, p0, 0.0, sigma2)

        loss_per_iteration.append(loss.item())  # Save it for plotting later
        loss.backward()
        optimizer.step()

        # Clamp theta values to stay valid
        with torch.no_grad():
            print("iter: ", step)
            print("sigma2: ", sigma2)
            print("distances: ", distances)
            print("loss", loss)
            print("theta grad: ", theta.grad)
            print("theta: ", theta)
            print("prob: ", p)
            iter_sigma+=1
            
            # Update sigma2 every N steps (or every step)
            # if i % 100 == 0:
            #     sigma2 = torch.mean(distances.detach()**2) / 3

            if iter_sigma == 2:
                iter_sigma = 0
                fitting_error = torch.sum(p*(distances)**2)
                # print("fittt: ", fitting_error)
                # sigma2 = 2*fitting_error / (3 * torch.sum(p))
                
                sigma2_new = 2 * torch.sum(p * distances**2) / (3 * torch.sum(p) + 1e-8)
                sigma2 = 0.9 *sigma2+0.1*sigma2_new
                sigma2 = torch.clamp(sigma2, min=1e-4)
                print("sigma2: ", sigma2)


            #     sigma2 = torch.sum(p * distances) / (torch.sum(p) * 3)
                
                # theta[5:8] += 0.01
                # theta[8:11] += -0.2
                
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
    fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
    theta_np = theta.detach().cpu().numpy()
    theta = theta.clone()  # (optional if you're not sure)
    theta[8:11] = theta[8:11] + t0
    theta_np = theta.detach().cpu().numpy()
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
    
    indices_bad = torch.nonzero(p <= 0.2, as_tuple=False).squeeze()
    selected_indices_bad = indices_bad.cpu().numpy()

    remaining_indices1 = np.setdiff1d(remaining_indices, selected_indices_bad)
    



    
    # print("indices: ", indices)
    
    
    showPoints(cluster[selected_indices_good], scale_factor=0.01, color=(0,1,0))
    showPoints(cluster[selected_indices_bad], scale_factor=0.01, color=(1,0,0))
    showPoints(cluster[remaining_indices1], scale_factor=0.01, color=(0,0,1))
    showPoints(filtered_points, scale_factor=0.005, color=(0,0.5,0.5))

    showSuperquadrics(theta_np)

    mlab.show()
    
    import matplotlib.pyplot as plt

    plt.plot(loss_per_iteration)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Iteration")
    plt.grid(True)
    plt.show()
