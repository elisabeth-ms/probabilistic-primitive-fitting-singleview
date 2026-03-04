import plot_functions
from mayavi import mlab
import numpy as np
import psqf
import torch


MAX_GRIPPER_WIDTH = 0.19



def downsample_equally_by_arclength(points_2d, n_points):
    """
    points_2d: (2, M)
    returns:   (2, n_points) approximately equally spaced along the curve length
    """
    P = points_2d.T  # (M,2)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])  # cumulative arc-length

    if n_points >= P.shape[0]:
        return points_2d

    targets = np.linspace(0.0, s[-1], n_points)
    out = np.zeros((n_points, 2))
    out[:, 0] = np.interp(targets, s, P[:, 0])
    out[:, 1] = np.interp(targets, s, P[:, 1])
    return out.T  # (2, n_points)

import numpy as np

def downsample_equally_by_index(points_2d, n_points):
    """
    points_2d: (2, M) from uniformSampledSuperellipse
    returns:   (2, n_points) equally spaced by index
    """
    M = points_2d.shape[1]
    if n_points >= M:
        return points_2d
    idx = np.linspace(0, M - 1, n_points, dtype=int)
    return points_2d[:, idx]
  
def sample_sq_section_local_numpy_uniform(theta, plane="y", n_points=200,
                                          threshold=1e-2, num_limit=10000, arclength=0.02,
                                          method="arclength"):
    """
    Sample section using your uniformSampledSuperellipse, then keep exactly n_points.
    method: "index" or "arclength"
    """
    e1 = float(theta[0])
    e2 = float(theta[1])
    a1 = max(abs(float(theta[2])), 1e-6)
    a2 = max(abs(float(theta[3])), 1e-6)
    a3 = max(abs(float(theta[4])), 1e-6)

    if plane == "y":
        pts2 = psqf.uniformSampledSuperellipse(e1, [a1, a3], threshold, num_limit, arclength)  # (2,M)
        if method == "index":
            pts2 = downsample_equally_by_index(pts2, n_points)
        else:
            pts2 = downsample_equally_by_arclength(pts2, n_points)
        x, z = pts2[0], pts2[1]
        y = np.zeros_like(x)

    elif plane == "x":
        pts2 = psqf.uniformSampledSuperellipse(e2, [a2, a3], threshold, num_limit, arclength)
        pts2 = downsample_equally_by_index(pts2, n_points) if method=="index" else downsample_equally_by_arclength(pts2, n_points)
        y, z = pts2[0], pts2[1]
        x = np.zeros_like(y)

    elif plane == "z":
        pts2 = psqf.uniformSampledSuperellipse(e2, [a1, a2], threshold, num_limit, arclength)
        pts2 = downsample_equally_by_index(pts2, n_points) if method=="index" else downsample_equally_by_arclength(pts2, n_points)
        x, y = pts2[0], pts2[1]
        z = np.zeros_like(x)

    else:
        raise ValueError("plane must be 'x', 'y', or 'z'.")

    return np.vstack([x, y, z]).T  # (n_points, 3)
  
def superellipsoid_normals_local(points, theta, eps=1e-12):
    """
    Compute outward unit normals for a superellipsoid/superquadric (NO bend/taper),
    using the gradient of the implicit inside-outside function F.

    points: (N,3) points ON the surface in LOCAL coordinates.
    theta:  numpy array, expects:
      theta[0]=e1, theta[1]=e2, theta[2]=a1, theta[3]=a2, theta[4]=a3

    Returns:
      normals: (N,3) unit normals in LOCAL coordinates
    """
    e1 = float(theta[0])
    e2 = float(theta[1])

    a1 = max(abs(theta[2]), 1e-6)
    a2 = max(abs(theta[3]), 1e-6)
    a3 = max(abs(theta[4]), 1e-6)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # normalized coordinates
    xn = x / a1
    yn = y / a2
    zn = z / a3

    # Helpers: |u|^p with stability
    def abs_pow(u, p):
        return (np.abs(u) + eps) ** p

    # Build A = |x/a1|^(2/e2) + |y/a2|^(2/e2)
    p_xy = 2.0 / e2
    X = abs_pow(xn, p_xy)
    Y = abs_pow(yn, p_xy)
    A = X + Y

    # Factor: A^(e2/e1 - 1)
    k = (e2 / e1) - 1.0
    A_fac = abs_pow(A, k)  # safe even if A ~ 0

    # dF/dx, dF/dy
    # |xn|^(2/e2 - 1)
    p_xy_m1 = (2.0 / e2) - 1.0
    dx_term = abs_pow(xn, p_xy_m1) * np.sign(xn)
    dy_term = abs_pow(yn, p_xy_m1) * np.sign(yn)

    dFdx = (2.0 / (e1 * a1)) * A_fac * dx_term
    dFdy = (2.0 / (e1 * a2)) * A_fac * dy_term

    # dF/dz
    p_z_m1 = (2.0 / e1) - 1.0
    dz_term = abs_pow(zn, p_z_m1) * np.sign(zn)
    dFdz = (2.0 / (e1 * a3)) * dz_term

    grads = np.stack([dFdx, dFdy, dFdz], axis=1)

    # normalize
    norms = np.linalg.norm(grads, axis=1, keepdims=True) + eps
    normals = grads / norms
    return normals

import numpy as np

def gripper_lines_local_3d_independent(
    jaw_top=0.04,
    jaw_bottom=0.04,
    jaw_length=0.12,
    back_length=0.06,
    wrist_length=0.05
):

    L = float(jaw_length)
    B = float(back_length)
    W = float(wrist_length)

    y_top = +jaw_top
    y_bot = -jaw_bottom

    x_left  = -B
    x_right = L

    segs = []

    # wrist
    segs.append((np.array([x_left - W, 0, 0]),
                 np.array([x_left, 0, 0])))

    # back connector
    segs.append((np.array([x_left, y_bot, 0]),
                 np.array([x_left, y_top, 0])))

    # top finger
    segs.append((np.array([x_left, y_top, 0]),
                 np.array([x_right, y_top, 0])))

    # bottom finger
    segs.append((np.array([x_left, y_bot, 0]),
                 np.array([x_right, y_bot, 0])))

    return segs

def transform_lines_3d(segs_local, R, t):
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3,)
    return [(R @ p0 + t, R @ p1 + t) for (p0, p1) in segs_local]

def rotation_from_xy(x_axis, y_axis, eps=1e-9):
    """
    Build a right-handed rotation matrix R whose columns are the world-frame
    axes of the gripper local frame:
      R = [x y z]
    where:
      x = normalized x_axis (approach)
      y = normalized y_axis, orthogonalized to x
      z = x × y   (right-handed completion)

    x_axis, y_axis: shape (3,)
    """
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_axis, dtype=float)

    x = x / (np.linalg.norm(x) + eps)

    # remove any component of y along x (Gram-Schmidt) so it's perpendicular
    y = y - np.dot(y, x) * x
    y_norm = np.linalg.norm(y)
    if y_norm < eps:
        raise ValueError("y_axis is parallel (or too close) to x_axis; cannot define a frame.")
    y = y / y_norm

    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + eps)

    R = np.column_stack([x, y, z])  # local axes expressed in world coords
    return R

def draw_segments_mlab(segs, tube_radius=0.002, color=(0,0,0), figure=None):
    if figure is None:
        figure = mlab.figure(size=(800, 600), bgcolor=(1,1,1))
    for p0, p1 in segs:
        mlab.plot3d([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    tube_radius=tube_radius, color=color, figure=figure)
    return figure
  

def grasp_candidate_positions_from_theta(theta, max_width, n_points=200):
    """
    Return LOCAL grasp candidate positions by sampling the appropriate
    cross-section plane, based on which axis scale is < max_width.

    Rule:
      if only a1 < max_width -> sample x=0
      if only a2 < max_width -> sample y=0
      if only a3 < max_width -> sample z=0

    returns:
      points_local: (N,3)
      info: dict with chosen_axis and chosen_plane
    """
    a1 = abs(float(theta[2]))
    a2 = abs(float(theta[3]))
    a3 = abs(float(theta[4]))

    small = np.array([a1 < max_width, a2 < max_width, a3 < max_width], dtype=bool)
    num_small = int(small.sum())

    if num_small != 1:
        raise ValueError(
            f"Expected exactly one axis to be < max_width={max_width}, "
            f"but got a1={a1:.4f}, a2={a2:.4f}, a3={a3:.4f} (small flags={small})."
        )

    if small[0]:      # a1 small => thickness along X => sample x=0 (YZ ring)
        plane = "x"
        closing_axis = np.array([1, 0, 0])
    elif small[1]:    # a2 small => thickness along Y => sample y=0 (XZ ring)
        plane = "y"
        closing_axis = np.array([0, 1, 0])
    else:             # a3 small => thickness along Z => sample z=0 (XY ring)
        plane = "z"
        closing_axis = np.array([0, 0, 1])

    pts_local = sample_sq_section_local_numpy_uniform(theta, plane=plane, n_points=n_points)
    nrm_out_local = superellipsoid_normals_local(pts_local, theta)
    approach_dir_local = - nrm_out_local
    

    info = {
        "a1": a1, "a2": a2, "a3": a3,
        "max_width": max_width,
        "chosen_axis": closing_axis,
        "chosen_plane": f"{plane}=0",
        "n_points": n_points,
    }
    return pts_local, approach_dir_local, closing_axis, info

def sample_superquadric_rings(theta, n_rings=5, n_points_ring=40):

    e1, e2 = theta[0], theta[1]
    a1, a2, a3 = theta[2], theta[3], theta[4]

    z_vals = np.linspace(-a3, a3, n_rings)

    all_pts = []

    for z in z_vals:

        # scaling of cross section from superquadric equation
        scale = (1 - (abs(z)/a3)**(2/e1))**(e1/2)

        if scale <= 0:
            continue

        pts2d = psqf.uniformSampledSuperellipse(
            e2,
            [a1*scale, a2*scale]
        )

        pts2d = downsample_equally_by_index(pts2d, n_points_ring)

        x = pts2d[0]
        y = pts2d[1]

        z_arr = np.ones_like(x)*z

        pts3d = np.vstack([x,y,z_arr]).T

        all_pts.append(pts3d)

    return np.vstack(all_pts)

def rotation_from_xz(x_axis, z_axis, eps=1e-9):
    """
    Build right-handed R with columns [x y z] (gripper local axes expressed in the frame you pass in).
    x_axis -> gripper +X (approach)
    z_axis -> gripper +Z (your chosen perpendicular-to-closing direction)

    Returns R (3x3).
    """
    x = np.asarray(x_axis, dtype=float)
    z = np.asarray(z_axis, dtype=float)

    x = x / (np.linalg.norm(x) + eps)

    # Make z orthogonal to x (Gram–Schmidt)
    z = z - np.dot(z, x) * x
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        raise ValueError("z_axis is parallel (or too close) to x_axis; cannot define a frame.")
    z = z / z_norm

    # Right-handed completion
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + eps)

    # Recompute z to ensure perfect orthonormality
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + eps)

    R = np.column_stack([x, y, z])
    return R
  
# fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
# # theta_box = [0.2, 0.2, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# # theta_cylinder = [0.2, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]
# theta_ellipsoid = [1.0, 1.0, 0.07, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# pts_y0 = sample_sq_section_local_numpy(theta_ellipsoid, plane="x", n_points=30)
# local_nrm_out = superellipsoid_normals_local(pts_y0, theta_ellipsoid)                      # local normals

# approach_dir = -local_nrm_out

# standoff = -0.1  # 5 cm outside
# p_pre = pts_y0 + standoff * approach_dir

# segs_local = gripper_lines_local_3d_independent(
#     jaw_top=0.08,      # top finger opened more
#     jaw_bottom=0.08,   # bottom finger opened less
#     jaw_length=0.1,
#     back_length=0.0,
#     wrist_length=0.06
# )


# plot_functions.showPoints(pts_y0, scale_factor=0.01)

# print(pts_y0)

# plot_functions.showSuperquadrics(theta_ellipsoid)


# # Probando las flechas
# P = np.array([
#     [0, 0, 0],      # flecha 1 origen
#     [1, 0, 0],      # flecha 2 origen
#     [0, 1, 0],      # flecha 3 origen
#     [0, 0, 1]       # flecha 4 origen
# ])

# Q = np.array([
#     [1, 1, 0],      # flecha 1 destino
#     [2, 0, 0],      # flecha 2 destino
#     [0, 2, 0],      # flecha 3 destino
#     [0, 0, 2]       # flecha 4 destino
# ])

# plot_functions.show_vectors(P, Q, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)

# fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))


# plot_functions.show_vectors(origin, x_axis, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)
# plot_functions.show_vectors(origin, y_axis, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)
# plot_functions.show_vectors(origin, z_axis, mode='arrow', every=1, color=(0,0,1), scale_factor=1.0, figure=fig)

# plot_functions.show_vectors(p_pre, pts_y0, mode='arrow', every=4, color=(0,1,0), scale_factor=1.0, figure=fig)

# # Lets draw the gripper in just one of the grasp poses.
# t = pts_y0[2]
# Rot = rotation_from_xy(approach_dir[2], np.array([1,0,0]))
# segs_gripper_at_pose = transform_lines_3d(segs_local, Rot, t)
# draw_segments_mlab(segs_gripper_at_pose, tube_radius=0.003, figure=fig)
# mlab.show()

# fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
# draw_segments_mlab(segs_local, tube_radius=0.003)
# mlab.show()

print("ONLY a1 < MAX_WIDTH/2")
theta_ellipsoid = [1.0, 1.0, 0.07, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
pts_grasp_local, approach_dir_local, closing_axis, info = grasp_candidate_positions_from_theta(theta_ellipsoid, MAX_GRIPPER_WIDTH/2, 50)
gripper_segs = gripper_lines_local_3d_independent(
    jaw_top=0.08,      # top finger opened more
    jaw_bottom=0.08,   # bottom finger opened less
    jaw_length=0.1,
    back_length=0.0,
    wrist_length=0.06
)


t_grasp_poses = []
rot_grasp_poses = []
for i in range(0, len(pts_grasp_local), 5):
    t_grasp_poses.append(pts_grasp_local[i])
    rot_grasp_poses.append(rotation_from_xy(approach_dir_local[i], closing_axis))



fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
plot_functions.showSuperquadrics(theta_ellipsoid)
plot_functions.showPoints(pts_grasp_local, scale_factor=0.01, figure=fig)
standoff = -0.1  # 5 cm outside
p_pre_local = pts_grasp_local + standoff * approach_dir_local
plot_functions.show_vectors(p_pre_local, pts_grasp_local, mode='arrow', every=4, color=(0,1,0), scale_factor=1.0, figure=fig)
for i in range(0, len(t_grasp_poses)):
    gripper_at_pose = transform_lines_3d(gripper_segs, rot_grasp_poses[i], t_grasp_poses[i])
    draw_segments_mlab(gripper_at_pose, tube_radius=0.003, figure=fig)
origin = np.array([[0, 0, 0]])
x_axis = np.array([[1, 0, 0]])
y_axis = np.array([[0, 1, 0]])
z_axis = np.array([[0, 0, 1]])
plot_functions.show_vectors(origin, x_axis, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, y_axis, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, z_axis, mode='arrow', every=1, color=(0,0,1), scale_factor=1.0, figure=fig)
mlab.show()

print("ONLY a2 < MAX_WIDTH/2")
theta_ellipsoid_A2 = [1.0, 1.0, 0.3, 0.07, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
pts_grasp_local, approach_dir_local, closing_axis, info = grasp_candidate_positions_from_theta(theta_ellipsoid_A2, MAX_GRIPPER_WIDTH/2, 50)
gripper_segs = gripper_lines_local_3d_independent(
    jaw_top=0.08,      # top finger opened more
    jaw_bottom=0.08,   # bottom finger opened less
    jaw_length=0.1,
    back_length=0.0,
    wrist_length=0.06
)


t_grasp_poses = []
rot_grasp_poses = []
for i in range(0, len(pts_grasp_local), 5):
    t_grasp_poses.append(pts_grasp_local[i])
    rot_grasp_poses.append(rotation_from_xy(approach_dir_local[i], closing_axis))



fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
plot_functions.showSuperquadrics(theta_ellipsoid_A2)
plot_functions.showPoints(pts_grasp_local, scale_factor=0.01, figure=fig)
standoff = -0.1  # 5 cm outside
p_pre_local = pts_grasp_local + standoff * approach_dir_local
plot_functions.show_vectors(p_pre_local, pts_grasp_local, mode='arrow', every=4, color=(0,1,0), scale_factor=1.0, figure=fig)
for i in range(0, len(t_grasp_poses)):
    gripper_at_pose = transform_lines_3d(gripper_segs, rot_grasp_poses[i], t_grasp_poses[i])
    draw_segments_mlab(gripper_at_pose, tube_radius=0.003, figure=fig)
origin = np.array([[0, 0, 0]])
x_axis = np.array([[1, 0, 0]])
y_axis = np.array([[0, 1, 0]])
z_axis = np.array([[0, 0, 1]])
plot_functions.show_vectors(origin, x_axis, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, y_axis, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, z_axis, mode='arrow', every=1, color=(0,0,1), scale_factor=1.0, figure=fig)
mlab.show()

print("ONLY a3 < MAX_WIDTH/2")
theta_ellipsoid_A3 = [0.2, 0.2, 0.3, 0.2, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
pts_grasp_local, approach_dir_local, closing_axis, info = grasp_candidate_positions_from_theta(theta_ellipsoid_A3, MAX_GRIPPER_WIDTH/2, 100)
gripper_segs = gripper_lines_local_3d_independent(
    jaw_top=0.08,      # top finger opened more
    jaw_bottom=0.08,   # bottom finger opened less
    jaw_length=0.1,
    back_length=0.0,
    wrist_length=0.06
)


t_grasp_poses = []
rot_grasp_poses = []
for i in range(0, len(pts_grasp_local), 10):
    t_grasp_poses.append(pts_grasp_local[i])
    rot_grasp_poses.append(rotation_from_xy(approach_dir_local[i], closing_axis))



fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
plot_functions.showSuperquadrics(theta_ellipsoid_A3)
plot_functions.showPoints(pts_grasp_local, scale_factor=0.01, figure=fig)
standoff = -0.1  # 5 cm outside
p_pre_local = pts_grasp_local + standoff * approach_dir_local
plot_functions.show_vectors(p_pre_local, pts_grasp_local, mode='arrow', every=4, color=(0,1,0), scale_factor=1.0, figure=fig)
for i in range(0, len(t_grasp_poses)):
    gripper_at_pose = transform_lines_3d(gripper_segs, rot_grasp_poses[i], t_grasp_poses[i])
    draw_segments_mlab(gripper_at_pose, tube_radius=0.003, figure=fig)
origin = np.array([[0, 0, 0]])
x_axis = np.array([[1, 0, 0]])
y_axis = np.array([[0, 1, 0]])
z_axis = np.array([[0, 0, 1]])
plot_functions.show_vectors(origin, x_axis, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, y_axis, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, z_axis, mode='arrow', every=1, color=(0,0,1), scale_factor=1.0, figure=fig)
mlab.show()

print("A1 and A2 < MAX_WIDTH/2")
theta_ellipsoid_A1A2 = [1.0, 1.0, 0.07, 0.07, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
pts_full_surface_local = sample_superquadric_rings(theta_ellipsoid_A1A2, 10, 50)

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
plot_functions.showSuperquadrics(theta_ellipsoid_A1A2)
plot_functions.show_vectors(origin, x_axis, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, y_axis, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, z_axis, mode='arrow', every=1, color=(0,0,1), scale_factor=1.0, figure=fig)
plot_functions.showPoints(pts_full_surface_local, scale_factor=0.01, figure=fig)


local_nrm_out = superellipsoid_normals_local(pts_full_surface_local, theta_ellipsoid_A1A2)                      # local normals

approach_dir = -local_nrm_out

standoff = -0.1  # 5 cm outside
p_pre_local = pts_full_surface_local + standoff * approach_dir
plot_functions.show_vectors(p_pre_local, pts_full_surface_local, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)


mlab.show()
