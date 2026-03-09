import plot_functions
from mayavi import mlab
import numpy as np
import psqf
import torch
import superellipsoid_tools





import numpy as np

def gripper_lines_local_3d_independent(
    jaw_top=0.04,
    jaw_bottom=0.04,
    jaw_length=0.12,
    back_length=0.06,
    wrist_length=0.05,
    pad_width=0.03
):

    L = float(jaw_length)
    B = float(back_length)
    W = float(wrist_length)

    y_top = +jaw_top
    y_bot = -jaw_bottom

    x_left  = -B
    x_right = L

    z_left  = -pad_width/2
    z_right = +pad_width/2

    segs = []
    top_finger = []
    bottom_finger = []

    # wrist
    segs.append((np.array([x_left - W, 0, 0]),
                 np.array([x_left, 0, 0])))

    # back connector
    segs.append((np.array([x_left, y_bot, 0]),
                 np.array([x_left, y_top, 0])))

   # ----- top finger rectangle -----

    s = (np.array([x_right, y_top, z_left]), np.array([x_right, y_top, z_right]))
    segs.append(s); top_finger.append(s)

    s = (np.array([x_left, y_top, z_left]), np.array([x_left, y_top, z_right]))
    segs.append(s); top_finger.append(s)

    s = (np.array([x_left, y_top, z_left]), np.array([x_right, y_top, z_left]))
    segs.append(s); top_finger.append(s)

    s = (np.array([x_left, y_top, z_right]), np.array([x_right, y_top, z_right]))
    segs.append(s); top_finger.append(s)

    # ----- bottom finger rectangle -----

    s = (np.array([x_right, y_bot, z_left]), np.array([x_right, y_bot, z_right]))
    segs.append(s); bottom_finger.append(s)

    s = (np.array([x_left, y_bot, z_left]), np.array([x_left, y_bot, z_right]))
    segs.append(s); bottom_finger.append(s)

    s = (np.array([x_left, y_bot, z_left]), np.array([x_right, y_bot, z_left]))
    segs.append(s); bottom_finger.append(s)

    s = (np.array([x_left, y_bot, z_right]), np.array([x_right, y_bot, z_right]))
    segs.append(s); bottom_finger.append(s)

    return segs, top_finger, bottom_finger

import numpy as np

def sample_pad_pairs_local(
    x_left,
    x_right,
    y_top,
    y_bot,
    z_left,
    z_right,
    n_length=5,
    n_width=3,
    avoid_borders=False
):
    """
    Sample corresponding points on top and bottom pads in GRIPPER LOCAL frame.

    Returns:
        pts_top_local : (N,3)
        pts_bot_local : (N,3)
        dist_x        : (N,) distance from x_left
        x_vals        : (N,)
        z_vals        : (N,)
    """
    if avoid_borders:
        xs = np.linspace(x_left, x_right, n_length + 2)[1:-1]
        zs = np.linspace(z_left, z_right, n_width + 2)[1:-1]
    else:
        xs = np.linspace(x_left, x_right, n_length)
        zs = np.linspace(z_left, z_right, n_width)

    pts_top_local = []
    pts_bot_local = []
    dist_x = []
    x_vals = []
    z_vals = []

    for x in xs:
        for z in zs:
            pts_top_local.append([x, y_top, z])
            pts_bot_local.append([x, y_bot, z])
            dist_x.append(x - x_left)
            x_vals.append(x)
            z_vals.append(z)

    return (
        np.asarray(pts_top_local, dtype=float),
        np.asarray(pts_bot_local, dtype=float),
        np.asarray(dist_x, dtype=float),
        np.asarray(x_vals, dtype=float),
        np.asarray(z_vals, dtype=float),
    )

def transform_pad_pairs_to_sq_frame(pts_top_local, pts_bot_local, R_g, t_g):
    """
    Transform sampled pad points from gripper local frame to SQ local frame.
    """
    R_g = np.asarray(R_g, dtype=float)
    t_g = np.asarray(t_g, dtype=float).reshape(3,)

    pts_top_sq = (R_g @ pts_top_local.T).T + t_g[None, :]
    pts_bot_sq = (R_g @ pts_bot_local.T).T + t_g[None, :]

    return pts_top_sq, pts_bot_sq



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
        figure = mlab.figure(size=(800,600), bgcolor=(1,1,1))

    segs = np.asarray(segs)

    n = len(segs)

    points = np.zeros((2*n,3))
    connections = np.zeros((n,2), dtype=int)

    for i,(p0,p1) in enumerate(segs):
        points[2*i] = p0
        points[2*i+1] = p1
        connections[i] = [2*i,2*i+1]

    src = mlab.pipeline.line_source(points[:,0], points[:,1], points[:,2])

    src.mlab_source.dataset.lines = connections

    lines = mlab.pipeline.tube(src, tube_radius=tube_radius)
    mlab.pipeline.surface(lines, color=color)

    return src
  
def update_segments_mlab(src, segs):

    segs = np.asarray(segs)

    n = len(segs)

    points = np.zeros((2*n,3))

    points[0::2] = segs[:,0]
    points[1::2] = segs[:,1]

    src.mlab_source.set(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2]
    )
    
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

    pts_local = superellipsoid_tools.sample_sq_section_local_numpy_uniform(theta, plane=plane, n_points=n_points)
    nrm_out_local = superellipsoid_tools.superellipsoid_normals_local(pts_local, theta)
    approach_dir_local = - nrm_out_local
    

    info = {
        "a1": a1, "a2": a2, "a3": a3,
        "max_width": max_width,
        "chosen_axis": closing_axis,
        "chosen_plane": f"{plane}=0",
        "n_points": n_points,
    }
    return pts_local, approach_dir_local, closing_axis, info




def fingers_collide_sq_local(finger_segs_local, R_g, t_g, theta, samples_per_seg=15):
    """
    Returns True if any sampled point on any finger/pad segment lies inside the SQ (F <= 1).

    finger_segs_local: list of segments [(p0, p1), (p0, p1), ...] in gripper local frame
                       e.g. all border segments of the top and bottom pads
    R_g, t_g: gripper pose in SQ local coordinates
    """

    R_g = np.asarray(R_g, dtype=float)
    t_g = np.asarray(t_g, dtype=float).reshape(3,)

    if len(finger_segs_local) == 0:
        return False

    ts = np.linspace(0.0, 1.0, samples_per_seg)

    pts = []
    for p0, p1 in finger_segs_local:
        p0 = np.asarray(p0, dtype=float).reshape(3,)
        p1 = np.asarray(p1, dtype=float).reshape(3,)

        seg_pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        pts.append(seg_pts)

    pts_g = np.vstack(pts)                       # (N_segments * samples_per_seg, 3)
    pts_sq = (R_g @ pts_g.T).T + t_g[None, :]   # to SQ local

    F = superellipsoid_tools.sq_F_local(pts_sq, theta)
    return np.any(F <= 1.0)

# def sample_superquadric_rings(theta, n_rings=5, n_points_ring=40):

#     e1, e2 = theta[0], theta[1]
#     a1, a2, a3 = theta[2], theta[3], theta[4]

#     z_vals = np.linspace(-a3, a3, n_rings)

#     all_pts = []

#     for z in z_vals:

#         # scaling of cross section from superquadric equation
#         scale = (1 - (abs(z)/a3)**(2/e1))**(e1/2)

#         if scale <= 0:
#             continue

#         pts2d = psqf.uniformSampledSuperellipse(
#             e2,
#             [a1*scale, a2*scale]
#         )

#         pts2d = downsample_equally_by_index(pts2d, n_points_ring)

#         x = pts2d[0]
#         y = pts2d[1]

#         z_arr = np.ones_like(x)*z

#         pts3d = np.vstack([x,y,z_arr]).T

#         all_pts.append(pts3d)

#     return np.vstack(all_pts)





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



def cap_is_face_like(theta, delta=0.01, min_cap_radius=0.01):
    """
    delta: how close to the cap (fraction of a3). delta=0.01 => z = a3*(1-0.01)
    min_cap_radius: threshold radius in meters for considering the cap 'usable'
    """
    e1 = max(abs(float(theta[0])), 1e-6)
    a1 = abs(float(theta[2]))
    a2 = abs(float(theta[3]))
    a3 = abs(float(theta[4]))

    # z at cap band
    r = 1.0 - float(delta)  # r = |z|/a3
    # scale from your ring formula
    base = 1.0 - (r ** (2.0 / e1))
    if base <= 0.0:
        return False, 0.0

    scale = base ** (e1 / 2.0)

    cap_rx = a1 * scale
    cap_ry = a2 * scale
    cap_r = max(cap_rx, cap_ry)

    return (cap_r >= min_cap_radius), cap_r

def filter_points_near_caps(pts, a3, tol=0.005):
    z = pts[:, 2]

    mask = (np.abs(z - a3) < tol) | (np.abs(z + a3) < tol)

    return pts[mask]

def tangent_closest_to_axis(normals, axis=np.array([0.0, 0.0, 1.0]), eps=1e-9):
    """
    normals: (N,3) unit (or non-unit) normals
    axis: (3,) reference direction (e.g., z axis)
    returns: tangents (N,3) unit tangent directions (in the plane orthogonal to normal)
    """
    n = np.asarray(normals, dtype=float)
    # normalize normals
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + eps)

    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + eps)

    # project axis onto tangent plane: t = a - (a·n)n
    dot = np.sum(n * a[None, :], axis=1, keepdims=True)   # (N,1)
    t = a[None, :] - dot * n                                # (N,3)

    # handle degeneracy when axis || normal (projection ~ 0)
    t_norm = np.linalg.norm(t, axis=1, keepdims=True)
    bad = (t_norm[:, 0] < 1e-6)

    if np.any(bad):
        # fallback axis (pick something not parallel to n)
        fallback = np.array([1.0, 0.0, 0.0])
        dot2 = np.sum(n[bad] * fallback[None, :], axis=1, keepdims=True)
        t2 = fallback[None, :] - dot2 * n[bad]
        t[bad] = t2
        t_norm[bad] = np.linalg.norm(t2, axis=1, keepdims=True)

    t = t / (t_norm + eps)
    return t

def prune_points_voxel(points, rots=None, voxel=0.02):
    """
    Keep at most 1 point per voxel cell of size 'voxel' (meters).
    points: (N,3)
    rots: (N,3,3) or list aligned with points
    """
    P = np.asarray(points, float)
    key = np.floor(P / voxel).astype(np.int64)

    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)

    Pk = P[idx]
    if rots is None:
        return Pk

    R = np.asarray(rots)
    return Pk, R[idx]
  

def prune_indices_voxel(points, voxel=0.02):
    """
    Returns indices of points to keep (one per voxel cell).
    """
    P = np.asarray(points, float)
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return np.sort(idx)
  

def fingers_sample_pairs_local(finger_segs_local, R_g, t_g, samples_per_seg=15):
    """
    Returns:
        pts_top : (N,3) sampled points on top finger
        pts_bot : (N,3) sampled points on bottom finger
        dist    : (N,) distance along the finger segment from p0
    """

    R_g = np.asarray(R_g, float)
    t_g = np.asarray(t_g, float).reshape(3,)

    (p0_top, p1_top), (p0_bot, p1_bot) = finger_segs_local

    p0_top = np.asarray(p0_top, float)
    p1_top = np.asarray(p1_top, float)

    p0_bot = np.asarray(p0_bot, float)
    p1_bot = np.asarray(p1_bot, float)

    ts = np.linspace(0.0, 1.0, samples_per_seg)

    pts_top = []
    pts_bot = []

    # segment length (same for both fingers)
    L = np.linalg.norm(p1_top - p0_top)

    for t in ts:

        pt_top = p0_top + t * (p1_top - p0_top)
        pt_bot = p0_bot + t * (p1_bot - p0_bot)

        pts_top.append(pt_top)
        pts_bot.append(pt_bot)

    pts_top = np.array(pts_top)
    pts_bot = np.array(pts_bot)

    # transform to SQ frame
    pts_top = (R_g @ pts_top.T).T + t_g
    pts_bot = (R_g @ pts_bot.T).T + t_g

    dist = ts * L

    return pts_top, pts_bot, dist

def intersect_segment_superquadric_all(p0, p1, theta, samples=400, refine_steps=40):
    """
    Return all intersection points between the segment p0->p1 and the
    superquadric surface in local coordinates.

    Returns:
        hits        : (K,3) array, typically K = 0, 1, or 2
        t_hits      : (K,) parameter values in [0,1]
        n_hits      : int
        dist_top    : distance from p0 to the closest hit (None if no hit)
        dist_bottom : distance from p1 to the closest hit (None if no hit)
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    e1, e2 = float(theta[0]), float(theta[1])
    a1 = max(abs(float(theta[2])), 1e-6)
    a2 = max(abs(float(theta[3])), 1e-6)
    a3 = max(abs(float(theta[4])), 1e-6)

    def F(p):
        x, y, z = p
        term1 = (abs(x / a1) ** (2.0 / e2) + abs(y / a2) ** (2.0 / e2)) ** (e2 / e1)
        term2 = abs(z / a3) ** (2.0 / e1)
        return term1 + term2 - 1.0

    ts = np.linspace(0.0, 1.0, samples)
    vals = np.array([F(p0 + t * (p1 - p0)) for t in ts])

    candidate_intervals = []
    for i in range(len(ts) - 1):
        v0, v1 = vals[i], vals[i + 1]

        if abs(v0) < 1e-10:
            candidate_intervals.append((ts[i], ts[i]))
        elif v0 * v1 < 0 or abs(v1) < 1e-10:
            candidate_intervals.append((ts[i], ts[i + 1]))

    t_hits = []

    for a, b in candidate_intervals:
        if a == b:
            t_hit = a
        else:
            fa = F(p0 + a * (p1 - p0))

            for _ in range(refine_steps):
                m = 0.5 * (a + b)
                fm = F(p0 + m * (p1 - p0))

                if abs(fm) < 1e-12:
                    a = b = m
                    break

                if fa * fm <= 0:
                    b = m
                else:
                    a = m
                    fa = fm

            t_hit = 0.5 * (a + b)

        if not any(abs(t_hit - t_old) < 1e-6 for t_old in t_hits):
            t_hits.append(t_hit)

    t_hits = np.array(sorted(t_hits), dtype=float)
    hits = np.array([p0 + t * (p1 - p0) for t in t_hits], dtype=float)

    n_hits = len(t_hits)
    seg_len = np.linalg.norm(p1 - p0)

    if n_hits == 0:
        dist_top = None
        dist_bottom = None
    else:
        dist_top = float(np.min(t_hits) * seg_len)
        dist_bottom = float((1.0 - np.max(t_hits)) * seg_len)

    return hits, t_hits, n_hits, dist_top, dist_bottom

def rotation_matrix_about_axis(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c

    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ]
    ], dtype=float)

    return R

def intersect_segments_superquadric_batch(p0s, p1s, theta, samples=400, refine_steps=40):

    p0s = np.asarray(p0s, dtype=float)
    p1s = np.asarray(p1s, dtype=float)

    N = p0s.shape[0]

    dirs = p1s - p0s
    seg_lens = np.linalg.norm(dirs, axis=1)

    e1, e2 = float(theta[0]), float(theta[1])
    a1 = max(abs(float(theta[2])), 1e-6)
    a2 = max(abs(float(theta[3])), 1e-6)
    a3 = max(abs(float(theta[4])), 1e-6)

    def F_batch(points):

        x = points[...,0]
        y = points[...,1]
        z = points[...,2]

        term1 = (np.abs(x/a1)**(2/e2) + np.abs(y/a2)**(2/e2))**(e2/e1)
        term2 = np.abs(z/a3)**(2/e1)

        return term1 + term2 - 1

    def F_single(p):

        x,y,z = p

        term1 = (abs(x/a1)**(2/e2) + abs(y/a2)**(2/e2))**(e2/e1)
        term2 = abs(z/a3)**(2/e1)

        return term1 + term2 - 1

    ts = np.linspace(0,1,samples)

    pts = p0s[:,None,:] + ts[None,:,None]*(dirs[:,None,:])

    vals = F_batch(pts)

    hits = np.zeros((N,2,3))
    t_hits = np.zeros((N,2))
    n_hits = np.zeros(N,dtype=int)

    dist_top = np.zeros(N)
    dist_bottom = np.zeros(N)

    tol_zero = 1e-10

    v0 = vals[:, :-1]
    v1 = vals[:, 1:]

    mask_exact_v0 = np.abs(v0) < tol_zero
    mask_candidate = mask_exact_v0 | (v0 * v1 < 0) | (np.abs(v1) < tol_zero)

    ray_idx, sample_idx = np.where(mask_candidate)

    t_start = ts[sample_idx]
    t_end = ts[sample_idx + 1]

    # for exact v0 roots, collapse interval to a point
    is_exact = mask_exact_v0[ray_idx, sample_idx]
    t_end[is_exact] = t_start[is_exact]

    for r in range(N):

        candidate_intervals = []

        for i in range(samples-1):

            v0 = vals[r,i]
            v1 = vals[r,i+1]

            if abs(v0) < 1e-10:
                candidate_intervals.append((ts[i],ts[i]))

            elif v0*v1 < 0 or abs(v1) < 1e-10:
                candidate_intervals.append((ts[i],ts[i+1]))

        t_local = []

        for a,b in candidate_intervals:

            if a==b:
                t_hit = a

            else:

                p0 = p0s[r]
                d = dirs[r]

                fa = F_single(p0 + a*d)

                for _ in range(refine_steps):

                    m = 0.5*(a+b)
                    fm = F_single(p0 + m*d)

                    if abs(fm) < 1e-12:
                        a=b=m
                        break

                    if fa*fm <= 0:
                        b=m
                    else:
                        a=m
                        fa=fm

                t_hit = 0.5*(a+b)

            if not any(abs(t_hit-t_old)<1e-6 for t_old in t_local):
                t_local.append(t_hit)

        t_local = sorted(t_local)
        k = min(len(t_local),2)

        n_hits[r] = len(t_local)

        for j in range(k):

            t_hits[r,j] = t_local[j]
            hits[r,j] = p0s[r] + t_local[j]*dirs[r]

        if len(t_local) > 0:

            dist_top[r] = t_local[0]*seg_lens[r]
            dist_bottom[r] = (1-t_local[-1])*seg_lens[r]

    return hits, t_hits, n_hits, dist_top, dist_bottom


def detect_first_last_intervals(vals, ts, tol_zero=1e-10):
    """
    vals: (N,S) values of implicit function along each ray
    ts:   (S,) sampled parameters in [0,1]

    Returns:
        has_hit      : (N,) bool
        first_idx    : (N,) interval index or -1
        last_idx     : (N,) interval index or -1
        first_exact  : (N,) bool  -> interval collapses to a point at ts[first_idx]
        last_exact   : (N,) bool
        a_first,b_first,a_last,b_last : (N,) arrays with interval bounds
    """
    v0 = vals[:, :-1]   # (N,S-1)
    v1 = vals[:, 1:]    # (N,S-1)

    mask_exact_v0 = np.abs(v0) < tol_zero
    mask_cross = (v0 * v1) < 0
    mask_exact_v1 = np.abs(v1) < tol_zero

    mask_candidate = mask_exact_v0 | mask_cross | mask_exact_v1   # (N,S-1)

    has_hit = np.any(mask_candidate, axis=1)

    # first candidate interval per ray
    first_idx = np.argmax(mask_candidate, axis=1)
    first_idx = np.where(has_hit, first_idx, -1)

    # last candidate interval per ray
    rev_idx = np.argmax(mask_candidate[:, ::-1], axis=1)
    last_idx = mask_candidate.shape[1] - 1 - rev_idx
    last_idx = np.where(has_hit, last_idx, -1)

    N = vals.shape[0]
    row = np.arange(N)

    first_exact = np.zeros(N, dtype=bool)
    last_exact = np.zeros(N, dtype=bool)

    valid_first = first_idx >= 0
    valid_last = last_idx >= 0

    first_exact[valid_first] = mask_exact_v0[row[valid_first], first_idx[valid_first]]
    last_exact[valid_last] = mask_exact_v0[row[valid_last], last_idx[valid_last]]

    a_first = np.zeros(N, dtype=float)
    b_first = np.zeros(N, dtype=float)
    a_last = np.zeros(N, dtype=float)
    b_last = np.zeros(N, dtype=float)

    a_first[valid_first] = ts[first_idx[valid_first]]
    b_first[valid_first] = ts[np.clip(first_idx[valid_first] + 1, 0, len(ts) - 1)]

    a_last[valid_last] = ts[last_idx[valid_last]]
    b_last[valid_last] = ts[np.clip(last_idx[valid_last] + 1, 0, len(ts) - 1)]

    # collapse exact-root intervals
    b_first[first_exact] = a_first[first_exact]
    b_last[last_exact] = a_last[last_exact]

    return has_hit, first_idx, last_idx, first_exact, last_exact, a_first, b_first, a_last, b_last
  
def batched_bisection_segment_roots(p0s, p1s, theta, a, b, active_mask, refine_steps=40, tol=1e-12):
    """
    Refine one root interval per ray simultaneously.

    p0s, p1s : (N,3)
    a, b     : (N,) interval bounds
    active_mask : (N,) bool, rays that actually need refinement

    Returns:
        t_root : (N,) root parameter, 0 for inactive rays
    """
    p0s = np.asarray(p0s, dtype=float)
    p1s = np.asarray(p1s, dtype=float)

    e1, e2 = float(theta[0]), float(theta[1])
    a1 = max(abs(float(theta[2])), 1e-6)
    a2 = max(abs(float(theta[3])), 1e-6)
    a3 = max(abs(float(theta[4])), 1e-6)

    dirs = p1s - p0s

    def F_batch(points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        term1 = (np.abs(x / a1) ** (2.0 / e2) + np.abs(y / a2) ** (2.0 / e2)) ** (e2 / e1)
        term2 = np.abs(z / a3) ** (2.0 / e1)
        return term1 + term2 - 1.0

    t_root = np.zeros(len(a), dtype=float)

    # rays with exact root already have a==b
    exact_mask = active_mask & (np.abs(a - b) < 1e-15)
    t_root[exact_mask] = a[exact_mask]

    refine_mask = active_mask & ~exact_mask
    if not np.any(refine_mask):
        return t_root

    a_ref = a.copy()
    b_ref = b.copy()

    pts_a = p0s[refine_mask] + a_ref[refine_mask, None] * dirs[refine_mask]
    fa = F_batch(pts_a)

    idx_ref = np.where(refine_mask)[0]

    for _ in range(refine_steps):
        m = 0.5 * (a_ref[idx_ref] + b_ref[idx_ref])
        pts_m = p0s[idx_ref] + m[:, None] * dirs[idx_ref]
        fm = F_batch(pts_m)

        done = np.abs(fm) < tol

        # update intervals only for unfinished rays
        left_mask = (fa * fm <= 0) & (~done)
        right_mask = (~left_mask) & (~done)

        b_ref[idx_ref[left_mask]] = m[left_mask]
        a_ref[idx_ref[right_mask]] = m[right_mask]
        fa[right_mask] = fm[right_mask]

        # optional: freeze converged rays
        a_ref[idx_ref[done]] = m[done]
        b_ref[idx_ref[done]] = m[done]

    t_root[refine_mask] = 0.5 * (a_ref[refine_mask] + b_ref[refine_mask])

    return t_root

def intersect_segments_superquadric_first_last_batch(p0s, p1s, theta, samples=400, refine_steps=40, tol_zero=1e-10):
    """
    Compute first and last intersections for all rays at once.

    Returns:
        hits        : (N,2,3)
        t_hits      : (N,2)
        n_hits      : (N,)
        dist_top    : (N,)
        dist_bottom : (N,)
    """
    p0s = np.asarray(p0s, dtype=float)
    p1s = np.asarray(p1s, dtype=float)

    N = p0s.shape[0]
    dirs = p1s - p0s
    seg_lens = np.linalg.norm(dirs, axis=1)

    e1, e2 = float(theta[0]), float(theta[1])
    a1 = max(abs(float(theta[2])), 1e-6)
    a2 = max(abs(float(theta[3])), 1e-6)
    a3 = max(abs(float(theta[4])), 1e-6)

    def F_batch(points):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]

        term1 = (np.abs(x / a1) ** (2.0 / e2) + np.abs(y / a2) ** (2.0 / e2)) ** (e2 / e1)
        term2 = np.abs(z / a3) ** (2.0 / e1)
        return term1 + term2 - 1.0

    ts = np.linspace(0.0, 1.0, samples)
    pts = p0s[:, None, :] + ts[None, :, None] * dirs[:, None, :]
    vals = F_batch(pts)   # (N,S)

    has_hit, first_idx, last_idx, first_exact, last_exact, a_first, b_first, a_last, b_last = \
        detect_first_last_intervals(vals, ts, tol_zero=tol_zero)

    # refine first and last roots
    t_first = batched_bisection_segment_roots(
        p0s, p1s, theta, a_first, b_first, has_hit, refine_steps=refine_steps
    )
    t_last = batched_bisection_segment_roots(
        p0s, p1s, theta, a_last, b_last, has_hit, refine_steps=refine_steps
    )

    # count hits
    n_hits = np.zeros(N, dtype=int)
    n_hits[has_hit] = 1

    two_hits = has_hit & (np.abs(t_last - t_first) > 1e-6)
    n_hits[two_hits] = 2

    # outputs
    t_hits = np.zeros((N, 2), dtype=float)
    hits = np.zeros((N, 2, 3), dtype=float)
    dist_top = np.zeros(N, dtype=float)
    dist_bottom = np.zeros(N, dtype=float)

    # first hit always in slot 0
    t_hits[has_hit, 0] = t_first[has_hit]
    hits[has_hit, 0] = p0s[has_hit] + t_first[has_hit, None] * dirs[has_hit]

    # second hit only if distinct
    t_hits[two_hits, 1] = t_last[two_hits]
    hits[two_hits, 1] = p0s[two_hits] + t_last[two_hits, None] * dirs[two_hits]

    dist_top[has_hit] = t_first[has_hit] * seg_lens[has_hit]
    dist_bottom[has_hit] = (1.0 - t_last[has_hit]) * seg_lens[has_hit]

    return hits, t_hits, n_hits, dist_top, dist_bottom

def transform_pad_pairs_to_sq_frame_all(pts_top_local, pts_bot_local, R_all, t_all):
    """
    pts_top_local, pts_bot_local: (M,3)
    R_all: (G,3,3)
    t_all: (G,3)

    Returns:
        pts_top_sq_all, pts_bot_sq_all: (G,M,3)
    """
    pts_top_local = np.asarray(pts_top_local, dtype=float)
    pts_bot_local = np.asarray(pts_bot_local, dtype=float)
    R_all = np.asarray(R_all, dtype=float)
    t_all = np.asarray(t_all, dtype=float)

    pts_top_sq_all = np.einsum('gij,mj->gmi', R_all, pts_top_local) + t_all[:, None, :]
    pts_bot_sq_all = np.einsum('gij,mj->gmi', R_all, pts_bot_local) + t_all[:, None, :]

    return pts_top_sq_all, pts_bot_sq_all


def batched_grasp_intersections_chunked(
    pts_top_local,
    pts_bot_local,
    R_all,
    t_all,
    theta,
    samples=400,
    refine_steps=40,
    chunk_size=128
):
    """
    Compute intersections for all grasp poses in chunks.

    Inputs:
        pts_top_local, pts_bot_local: (M,3) pad sample pairs in gripper local frame
        R_all: (G,3,3)
        t_all: (G,3)

    Returns:
        hits:        (G,M,2,3)
        t_hits:      (G,M,2)
        n_hits:      (G,M)
        dist_top:    (G,M)
        dist_bottom: (G,M)
    """
    R_all = np.asarray(R_all, dtype=float)
    t_all = np.asarray(t_all, dtype=float)

    G = len(t_all)

    all_hits = []
    all_t_hits = []
    all_n_hits = []
    all_dist_top = []
    all_dist_bottom = []

    for start in range(0, G, chunk_size):
        end = min(start + chunk_size, G)

        pts_top_sq_all, pts_bot_sq_all = transform_pad_pairs_to_sq_frame_all(
            pts_top_local, pts_bot_local, R_all[start:end], t_all[start:end]
        )

        g, m, _ = pts_top_sq_all.shape

        p0s = pts_top_sq_all.reshape(-1, 3)
        p1s = pts_bot_sq_all.reshape(-1, 3)

        hits, t_hits, n_hits, dist_top, dist_bottom = \
            intersect_segments_superquadric_first_last_batch(
                p0s,
                p1s,
                theta,
                samples=samples,
                refine_steps=refine_steps
            )

        all_hits.append(hits.reshape(g, m, 2, 3))
        all_t_hits.append(t_hits.reshape(g, m, 2))
        all_n_hits.append(n_hits.reshape(g, m))
        all_dist_top.append(dist_top.reshape(g, m))
        all_dist_bottom.append(dist_bottom.reshape(g, m))

    return (
        np.concatenate(all_hits, axis=0),
        np.concatenate(all_t_hits, axis=0),
        np.concatenate(all_n_hits, axis=0),
        np.concatenate(all_dist_top, axis=0),
        np.concatenate(all_dist_bottom, axis=0),
    )
    
def is_antipodal(normals_contact_point, tol_dot=-0.99):
    """
    normals_contact_point: (2,3), outward normals at the two contacts
    tol_dot: threshold for antipodality
             -1.0 = perfectly opposite
             e.g. -0.95 means angle > about 162 deg
    """
    n1 = normals_contact_point[0]
    n2 = normals_contact_point[1]

    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    n2 = n2 / (np.linalg.norm(n2) + 1e-12)

    dot = np.dot(n1, n2)
    return dot <= tol_dot, dot

def points_sq_to_gripper_local(points_sq, R_g, t_g):
    points_sq = np.asarray(points_sq, dtype=float)
    R_g = np.asarray(R_g, dtype=float)
    t_g = np.asarray(t_g, dtype=float).reshape(3,)
    return (R_g.T @ (points_sq - t_g[None, :]).T).T

def centroid_in_pad_central_region_gripper(points_gripper,
                                           x_left, x_right,
                                           z_left, z_right,
                                           low_x=0.3, high_x=0.7,
                                           low_z=0.2, high_z=0.8):
    """
    Check whether the centroid of the contact points lies within the allowed
    region of the pad.

    Only x (pad length) and z (pad width) are checked.
    y (closing direction) is ignored.

    Returns:
        ok       : bool
        centroid : (3,)
        ux, uz   : normalized centroid coordinates in x and z
    """

    points_gripper = np.asarray(points_gripper, dtype=float)

    if points_gripper.shape[0] == 0:
        return False, None, None, None

    centroid = np.mean(points_gripper, axis=0)

    ux = (centroid[0] - x_left) / (x_right - x_left)
    uz = (centroid[2] - z_left) / (z_right - z_left)

    ok = (low_x <= ux <= high_x) and (low_z <= uz <= high_z)

    return ok, centroid, ux, uz

def rotation_angle_between(R1, R2):
    R_rel = R1.T @ R2
    val = (np.trace(R_rel) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)

def prune_grasp_poses_greedy(translations, rotations, pos_thresh=0.01, ang_thresh_deg=10.0):
    """
    translations: (N,3)
    rotations:    (N,3,3)

    Returns:
        keep_idx: indices of kept poses
    """
    T = np.asarray(translations, dtype=float)
    R = np.asarray(rotations, dtype=float)

    ang_thresh = np.deg2rad(ang_thresh_deg)

    keep_idx = []

    for i in range(len(T)):
        keep = True
        for j in keep_idx:
            pos_dist = np.linalg.norm(T[i] - T[j])
            if pos_dist > pos_thresh:
                continue

            ang_dist = rotation_angle_between(R[i], R[j])
            if ang_dist <= ang_thresh:
                keep = False
                break

        if keep:
            keep_idx.append(i)

    return np.array(keep_idx, dtype=int)

import numpy as np

def prune_grasp_poses_fast(
    translations,
    rotations,
    pos_res=0.008,
    ang_res=0.10,
    use_closing_axis=True
):
    """
    Fast O(N) pruning using hashing.

    pos_res: position voxel size in meters
    ang_res: quantization step for unit-vector components
    """

    T = np.asarray(translations, dtype=float)
    R = np.asarray(rotations, dtype=float)

    keep_idx = []
    seen = set()

    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-12)

    def canonicalize_axis(v):
        """
        Make sign deterministic so v and -v do not become arbitrary duplicates.
        Flip so the first non-negligible component is positive.
        """
        v = normalize(v).copy()
        for k in range(3):
            if abs(v[k]) > 1e-9:
                if v[k] < 0:
                    v = -v
                break
        return v

    def axis_key(v):
        v = canonicalize_axis(v)
        return tuple(np.round(v / ang_res).astype(np.int64))

    for i in range(len(T)):
        t_key = tuple(np.round(T[i] / pos_res).astype(np.int64))

        a_key = axis_key(R[i][:, 0])

        if use_closing_axis:
            c_key = axis_key(R[i][:, 1])
            key = (t_key, a_key, c_key)
        else:
            key = (t_key, a_key)

        if key not in seen:
            seen.add(key)
            keep_idx.append(i)

    return np.asarray(keep_idx, dtype=int)
  

import numpy as np

def tangent_candidates_from_axes(
    normals,
    axes=None,
    min_norm=1e-6,
    duplicate_angle_deg=5.0,
    eps=1e-9
):
    """
    For each normal, return all distinct tangent directions obtained by
    projecting the candidate axes onto the tangent plane.

    normals: (N,3)
    axes:    (K,3), default = world x,y,z

    Returns:
        tangent_lists: list of length N
            tangent_lists[i] is a list of (3,) tangent vectors
    """
    n = np.asarray(normals, dtype=float)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + eps)

    if axes is None:
        axes = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)
    else:
        axes = np.asarray(axes, dtype=float)

    axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + eps)

    cos_dup = np.cos(np.deg2rad(duplicate_angle_deg))

    tangent_lists = []

    for i in range(len(n)):
        ni = n[i]
        tangents_i = []

        for a in axes:
            t = a - np.dot(a, ni) * ni
            t_norm = np.linalg.norm(t)

            if t_norm < min_norm:
                continue

            t = t / (t_norm + eps)

            # remove duplicates up to sign
            is_duplicate = False
            for tj in tangents_i:
                d = abs(np.dot(t, tj))
                if d > cos_dup:
                    is_duplicate = True
                    break

            if not is_duplicate:
                tangents_i.append(t)

        tangent_lists.append(tangents_i)

    return tangent_lists
  
MAX_GRIPPER_WIDTH = 0.19
jaw_top = 0.1
jaw_bottom = 0.1
jaw_length = 0.1
back_length = 0.0
wrist_length = 0.06
pad_width = 0.03
tol_contact = 0.001   # example: 2 mm
tol_dot = -0.99       # antipodal threshold
theta_ellipsoid_A1A2 = [0.1, 0.1, 0.04, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#theta_ellipsoid_A1A2 = [0.2, 1.0, 0.08, 0.08, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
n_rotations = 4
gripper_segs, top_pad_segs, bottom_pad_segs = gripper_lines_local_3d_independent(
    jaw_top=jaw_top,      # top finger opened more
    jaw_bottom=jaw_bottom,   # bottom finger opened less
    jaw_length=jaw_length,
    back_length=back_length,
    wrist_length=wrist_length,
    pad_width=pad_width
)

x_left = -back_length
x_right = jaw_length
y_top = jaw_top
y_bot = -jaw_bottom
z_left = -pad_width / 2
z_right = pad_width / 2



finger_segs = top_pad_segs + bottom_pad_segs
rot_local = []

# pts_full_surface_local_x = superellipsoid_tools.sample_superquadric_rings(theta_ellipsoid_A1A2, 40, 30, True, 'x')
# local_nrm_out_x = superellipsoid_tools.superellipsoid_normals_local(pts_full_surface_local_x, theta_ellipsoid_A1A2)                      # local normals
# tangent_x = tangent_closest_to_axis(-local_nrm_out_x, axis=np.array([1,0,0]))
# for i in range(len(tangent_x)):
#     aux_rot_local = rotation_from_xz(-local_nrm_out_x[i], tangent_x[i])
#     rot_local.append(aux_rot_local)

# pts_full_surface_local_z = superellipsoid_tools.sample_superquadric_rings(theta_ellipsoid_A1A2, 40, 30, True, 'z')
# local_nrm_out_z = superellipsoid_tools.superellipsoid_normals_local(pts_full_surface_local_z, theta_ellipsoid_A1A2)                      # local normals
# tangent_z = tangent_closest_to_axis(-local_nrm_out_z, axis=np.array([0,0,1]))

# for i in range(len(tangent_z)):
#     aux_rot_local = rotation_from_xz(-local_nrm_out_z[i], tangent_z[i])
#     rot_local.append(aux_rot_local)

# pts_full_surface_local_y = superellipsoid_tools.sample_superquadric_rings(theta_ellipsoid_A1A2, 40, 30, True, 'y')
# local_nrm_out_y = superellipsoid_tools.superellipsoid_normals_local(pts_full_surface_local_y, theta_ellipsoid_A1A2)                      # local normals
# tangent_y = tangent_closest_to_axis(-local_nrm_out_y, axis=np.array([0,1,0]))

# for i in range(len(tangent_y)):
#     aux_rot_local = rotation_from_xz(-local_nrm_out_y[i], tangent_y[i])
#     rot_local.append(aux_rot_local)


# local_nrm_out = np.vstack([local_nrm_out_x])

# pts_full_local = np.vstack([pts_full_surface_local_x])

pts_full_local = superellipsoid_tools.sample_superquadric_projected(
    theta_ellipsoid_A1A2,
    n_u=10,
    n_v=10,
    axes=('x', 'y', 'z')
)
local_nrm_out = superellipsoid_tools.superellipsoid_normals_local(pts_full_local, theta_ellipsoid_A1A2)                      # local normals
tangent_lists = tangent_candidates_from_axes(-local_nrm_out)

pts_expanded = []
normals_expanded = []
rot_local = []

for i in range(len(pts_full_local)):
    n = -local_nrm_out[i]

    for t in tangent_lists[i]:
        R = rotation_from_xz(n, t)

        pts_expanded.append(pts_full_local[i])
        normals_expanded.append(local_nrm_out[i])
        rot_local.append(R)

pts_full_local = np.asarray(pts_expanded)
local_nrm_out = np.asarray(normals_expanded)
rot_local = np.asarray(rot_local)

pts_all_local = pts_full_local.copy()
# print(pts_full_local)
################################### PRUNE OR NOT ############################################3
# idx = prune_indices_voxel(pts_full_local, 0.008)
# local_nrm_out = local_nrm_out[idx]
# pts_full_local = pts_full_local[idx]
# rot_local = np.stack(rot_local)   
# rot_local = rot_local[idx]
# rot_local = rot_local.tolist()



angles = np.linspace(0.0, np.pi/2.0, n_rotations, endpoint=False)

rot_local_all = []
pts_full_local_all = []
local_nrm_out_all = []

print("Number of grasp before adding rotation: ", len(pts_full_local))


for i in range(len(pts_full_local)):

    p = pts_full_local[i]
    n = local_nrm_out[i]
    base_R = rot_local[i]   # already computed from normal + tangent

    for ang in angles:
        R_spin = rotation_matrix_about_axis(-n, ang)
        R_new = R_spin @ base_R

        pts_full_local_all.append(p)
        local_nrm_out_all.append(n)
        rot_local_all.append(R_new)


pts_full_local = np.asarray(pts_full_local_all)
local_nrm_out = np.asarray(local_nrm_out_all)
rot_local = np.asarray(rot_local_all)

print("Number of grasp after adding rotation: ", len(pts_full_local))

keep_idx = prune_grasp_poses_fast(
    pts_full_local,
    rot_local,
    pos_res=0.002,
    ang_res=0.1,
    use_closing_axis=True
)
pts_full_local = pts_full_local[keep_idx]
rot_local = rot_local[keep_idx]
local_nrm_out = local_nrm_out[keep_idx]

print("Number of grasp after prunning rotation: ", len(pts_full_local))
# # First lets filter out poses that collide with the object.
t_grasp_poses = []
rot_grasp_poses = []
for i in range(0, len(pts_full_local), 1):
    if not fingers_collide_sq_local(finger_segs, rot_local[i], pts_full_local[i]+0.2*jaw_length*local_nrm_out[i], theta_ellipsoid_A1A2, 15):
        


        
        t_grasp_poses.append(pts_full_local[i]+0.2*jaw_length*local_nrm_out[i])
        rot_grasp_poses.append(rot_local[i])
        

origin = np.array([[0, 0, 0]])
x_axis = np.array([[1, 0, 0]])
y_axis = np.array([[0, 1, 0]])
z_axis = np.array([[0, 0, 1]])

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
plot_functions.showSuperquadrics(theta_ellipsoid_A1A2)
plot_functions.show_vectors(origin, x_axis, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, y_axis, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)
plot_functions.show_vectors(origin, z_axis, mode='arrow', every=1, color=(0,0,1), scale_factor=1.0, figure=fig)
plot_functions.showPoints(pts_full_local, scale_factor=0.005, figure=fig)
plot_functions.showPoints(np.array(t_grasp_poses), scale_factor=0.01, color=(0,0,1), figure=fig)
for i in range(0, len(t_grasp_poses), 1000):
    gripper_at_pose = transform_lines_3d(gripper_segs, rot_grasp_poses[i], t_grasp_poses[i])
    draw_segments_mlab(gripper_at_pose, tube_radius=0.003, figure=fig)
mlab.show()

pts_top_local, pts_bot_local, dist_x, x_vals, z_vals = sample_pad_pairs_local(
    x_left=x_left,
    x_right=x_right,
    y_top=y_top,
    y_bot=y_bot,
    z_left=z_left,
    z_right=z_right,
    n_length=15,
    n_width=10
)

t_new_grasp_poses = []
rot_new_grasp_poses = []
center_points = []
top_contact_points_close_all_grasps = []
bottom_contact_points_close_all_grasps = []
closing_positions_top = []
closing_positions_bottom = []
# print("Number of grasp that do not collide: ", len(t_grasp_poses))

R_all = np.asarray(rot_grasp_poses)
t_all = np.asarray(t_grasp_poses)

hits_all, t_hits_all, n_hits_all, dist_top_all, dist_bottom_all = \
    batched_grasp_intersections_chunked(
        pts_top_local=pts_top_local,
        pts_bot_local=pts_bot_local,
        R_all=R_all,
        t_all=t_all,
        theta=theta_ellipsoid_A1A2,
        samples=100,
        refine_steps=5,
        chunk_size=4500
    )

t_new_grasp_poses = []
rot_new_grasp_poses = []
center_points = []
top_contact_points_close_all_grasps = []
bottom_contact_points_close_all_grasps = []
closing_positions_top = []
closing_positions_bottom = []

for i in range(1, len(t_grasp_poses)):
    print("i:", i)

    hits = hits_all[i]              # (M,2,3)
    t_hits = t_hits_all[i]          # (M,2)
    n_hits = n_hits_all[i]          # (M,)
    dist_top = dist_top_all[i]      # (M,)
    dist_bottom = dist_bottom_all[i]# (M,)

    valid = (n_hits == 2)

    if not np.any(valid):
        continue

    dist_top_valid = dist_top[valid]
    dist_bottom_valid = dist_bottom[valid]
    hits_valid = hits[valid]

    idx_top = np.argmin(dist_top_valid)
    idx_bottom = np.argmin(dist_bottom_valid)

    d_top_min_contact = dist_top_valid[idx_top]
    d_bottom_min_contact = dist_bottom_valid[idx_bottom]

    pt_top_contact = hits_valid[idx_top, 0]
    pt_bottom_contact = hits_valid[idx_bottom, 1]

    # keep_grasp = abs(d_top_min_contact - d_bottom_min_contact) <= 3*tol_contact
    # # keep_grasp = True
    # if not keep_grasp:
    #     continue

    mask_top_close = np.abs(dist_top_valid - d_top_min_contact) <= tol_contact
    mask_bottom_close = np.abs(dist_bottom_valid - d_bottom_min_contact) <= tol_contact

    top_contact_points_close = hits_valid[mask_top_close, 0]
    bottom_contact_points_close = hits_valid[mask_bottom_close, 1]

    centroid_top_sq = np.mean(top_contact_points_close, axis=0)
    centroid_bottom_sq = np.mean(bottom_contact_points_close, axis=0)

    contact_pts_centroid = np.vstack([centroid_top_sq, centroid_bottom_sq])
    normals_contact = superellipsoid_tools.superellipsoid_normals_local(
        contact_pts_centroid,
        theta_ellipsoid_A1A2
    )

    n_top = normals_contact[0]
    n_bottom = normals_contact[1]
    dot_val = np.dot(n_top, n_bottom)
    ok_antipodal = dot_val <= tol_dot
    if not ok_antipodal:
        continue
    center_point = 0.5 * (centroid_top_sq + centroid_bottom_sq)
    
    approach = rot_grasp_poses[i][:,0]   # gripper x axis

    v = center_point - t_grasp_poses[i]

    alignment = abs(np.dot(v, approach)) / (np.linalg.norm(v) + 1e-12)
    
    if alignment < 0.95:
        continue
    
    closing_position_top = jaw_top - d_top_min_contact
    closing_position_bottom = jaw_bottom - d_bottom_min_contact

    centroid_contact_top_gripper = points_sq_to_gripper_local(
        centroid_top_sq[None, :],
        rot_grasp_poses[i],
        t_grasp_poses[i]
    )

    ok_top_centered, centroid_top, ux_top, uz_top = \
        centroid_in_pad_central_region_gripper(
            centroid_contact_top_gripper,
            x_left, x_right, z_left, z_right,
            low_x=0.2, low_z=0.2, high_x=0.8, high_z=0.8
        )

    if not ok_top_centered:
        continue

    top_contact_points_close_all_grasps.append(top_contact_points_close)
    bottom_contact_points_close_all_grasps.append(bottom_contact_points_close)

    closing_positions_bottom.append(closing_position_bottom)
    closing_positions_top.append(closing_position_top)

    t_new = t_grasp_poses[i]
    R_new = rot_grasp_poses[i]
    center_point = 0.5 * (pt_top_contact + pt_bottom_contact)

    t_new_grasp_poses.append(t_new)
    rot_new_grasp_poses.append(R_new)
    center_points.append(center_point)


print("computed")


# for i in range(0, len(t_new_grasp_poses), 4):
#     fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
#     plot_functions.showSuperquadrics(theta_ellipsoid_A1A2)
#     # plot_functions.show_vectors(origin, x_axis, mode='arrow', every=1, color=(1,0,0), scale_factor=1.0, figure=fig)
#     # plot_functions.show_vectors(origin, y_axis, mode='arrow', every=1, color=(0,1,0), scale_factor=1.0, figure=fig)
#     # plot_functions.show_vectors(origin, z_axis, mode='arrow', every=1, color=(0,0,1), scale_factor=1.0, figure=fig)
#     plot_functions.showPoints(pts_full_local, scale_factor=0.005, figure=fig)
#     plot_functions.showPoints(np.array(t_new_grasp_poses), color=(0,1,0), scale_factor=0.008, figure=fig)
#     gripper_segs_contact, _, _ = gripper_lines_local_3d_independent(
#     jaw_top=closing_positions_top[i],      # top finger opened more
#     jaw_bottom=closing_positions_bottom[i],   # bottom finger opened less
#     jaw_length=jaw_length,
#     back_length=back_length,
#     wrist_length=wrist_length,
#     pad_width=pad_width
#     )
#     gripper_at_pose = transform_lines_3d(gripper_segs_contact, rot_new_grasp_poses[i], t_new_grasp_poses[i])
#     draw_segments_mlab(gripper_at_pose, tube_radius=0.003, figure=fig)
#     plot_functions.showPoints(np.array(top_contact_points_close_all_grasps[i]), color=(0,0,1), scale_factor=0.008, figure=fig)
#     plot_functions.showPoints(np.array(bottom_contact_points_close_all_grasps[i]), color=(0,0,1), scale_factor=0.008, figure=fig)

#     mlab.show()
print("Figure")
fig = mlab.figure(size=(400, 400), bgcolor=(1,1,1))
MAX_CONTACT_POINTS = 100
contact_top_array = np.full((MAX_CONTACT_POINTS, 3), np.nan)
contact_bottom_array = np.full((MAX_CONTACT_POINTS, 3), np.nan)
plot_functions.showSuperquadrics(theta_ellipsoid_A1A2)

# initial gripper
i = 0
gripper_segs_contact, _, _ = gripper_lines_local_3d_independent(
    jaw_top=closing_positions_top[i],
    jaw_bottom=closing_positions_bottom[i],
    jaw_length=jaw_length,
    back_length=back_length,
    wrist_length=wrist_length,
    pad_width=pad_width
)
gripper_at_pose = transform_lines_3d(gripper_segs_contact, rot_new_grasp_poses[i], t_new_grasp_poses[i])
gripper_plot = draw_segments_mlab(gripper_at_pose, tube_radius=0.003, figure=fig)

contact_top = mlab.points3d(
    contact_top_array[:,0],
    contact_top_array[:,1],
    contact_top_array[:,2],
    scale_factor=0.008,
    color=(0,0,1),
    figure=fig
)

contact_bottom = mlab.points3d(
    contact_bottom_array[:,0],
    contact_bottom_array[:,1],
    contact_bottom_array[:,2],
    scale_factor=0.008,
    color=(0,0,1),
    figure=fig
)
def update_contacts(actor, pts, max_pts):

    arr = np.full((max_pts,3), np.nan)

    pts = np.asarray(pts)

    if pts.ndim == 1:
        pts = pts.reshape(1,3)

    n = min(len(pts), max_pts)

    arr[:n] = pts[:n]

    actor.mlab_source.set(
        x=arr[:,0],
        y=arr[:,1],
        z=arr[:,2]
    )
@mlab.animate(delay=100)
def anim():
    for i in range(0,len(t_new_grasp_poses), 1):

        gripper_segs_contact, _, _ = gripper_lines_local_3d_independent(
            jaw_top=closing_positions_top[i],
            jaw_bottom=closing_positions_bottom[i],
            jaw_length=jaw_length,
            back_length=back_length,
            wrist_length=wrist_length,
            pad_width=pad_width
        )

        gripper_at_pose = transform_lines_3d(
            gripper_segs_contact,
            rot_new_grasp_poses[i],
            t_new_grasp_poses[i]
        )

        # update gripper
        update_segments_mlab(gripper_plot, gripper_at_pose)

        # # update contact points
        # contact_top.mlab_source.set(
        #     x=top_contact_points_close_all_grasps[i][:,0],
        #     y=top_contact_points_close_all_grasps[i][:,1],
        #     z=top_contact_points_close_all_grasps[i][:,2]
        # )

        # contact_bottom.mlab_source.set(
        #     x=bottom_contact_points_close_all_grasps[i][:,0],
        #     y=bottom_contact_points_close_all_grasps[i][:,1],
        #     z=bottom_contact_points_close_all_grasps[i][:,2]
        # )

        update_contacts(contact_top,
                top_contact_points_close_all_grasps[i],
                MAX_CONTACT_POINTS)

        update_contacts(contact_bottom,
                        bottom_contact_points_close_all_grasps[i],
                        MAX_CONTACT_POINTS)
        yield

anim()
mlab.show()
# theta = [0.4, 0.1, 0.07, 0.07, 0.3, 0, 0, 0, 0, 0, 0]

# p0 = np.array([-0.2, 0.0, 0.0])
# p1 = np.array([ 0.2, 0.0, 0.0])

# hits, t_hits = intersect_segment_superquadric_all(p0, p1, theta)

# print("t_hits:", t_hits)
# print("hits:\n", hits)

# local_nrm_out_hits = superellipsoid_normals_local(hits, theta)                      # local normals

# print("normals: ", local_nrm_out_hits)
