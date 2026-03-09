
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
  
# def uniformSampledSuperellipse(epsilon, scale, threshold = 1e-2, num_limit = 10000, arclength = 0.02):

#     # initialize array storing sampled theta
#     theta = np.zeros(num_limit)
#     theta[0] = 0

#     for i in range(num_limit):
#         dt = dtheta(theta[i], arclength, threshold, scale, epsilon)
#         theta_temp = theta[i] + dt

#         if theta_temp > np.pi / 4:
#             theta[i + 1] = np.pi / 4
#             break
#         else:
#             if i + 1 < num_limit:
#                 theta[i + 1] = theta_temp
#             else:
#                 raise Exception(
#                 'Number of the sampled points exceed the preset limit', \
#                 num_limit,
#                 'Please decrease the sampling arclength.'
#                 )
#     critical = i + 1

#     for j in range(critical + 1, num_limit):
#         dt = dtheta(theta[j], arclength, threshold, np.flip(scale), epsilon)
#         theta_temp = theta[j] + dt
        
#         if theta_temp > np.pi / 4:
#             break
#         else:
#             if j + 1 < num_limit:
#                 theta[j + 1] = theta_temp
#             else:
#                 raise Exception(
#                 'Number of the sampled points exceed the preset limit', \
#                 num_limit,
#                 'Please decrease the sampling arclength.'
#                 )
#     num_pt = j
#     theta = theta[0 : num_pt + 1]

#     point_fw = angle2points(theta[0 : critical + 1], scale, epsilon)
#     point_bw = np.flip(angle2points(theta[critical + 1: num_pt + 1], np.flip(scale), epsilon), (0, 1))
#     point = np.concatenate((point_fw, point_bw), 1)
#     point = np.concatenate((point, np.flip(point[:, 0 : num_pt], 1) * np.array([[-1], [1]]), 
#                            point[:, 1 : num_pt + 1] * np.array([[-1], [-1]]),
#                            np.flip(point[:, 0 : num_pt], 1) * np.array([[1], [-1]])), 1)

#     return point
def uniformSampledSuperellipse(epsilon, scale, threshold=1e-2, num_limit=10000, arclength=0.02):
    theta = np.zeros(num_limit)
    theta[0] = 0.0

    # first branch
    for i in range(num_limit - 1):
        dt = dtheta(theta[i], arclength, threshold, scale, epsilon)
        theta_temp = theta[i] + dt

        if theta_temp > np.pi / 4:
            theta[i + 1] = np.pi / 4
            critical = i + 1
            break
        else:
            theta[i + 1] = theta_temp
    else:
        raise Exception(
            'Number of the sampled points exceed the preset limit',
            num_limit,
            'Please decrease the sampling arclength.'
        )

    # second branch
    last_idx = critical
    for j in range(critical + 1, num_limit):
        dt = dtheta(theta[j - 1], arclength, threshold, np.flip(scale), epsilon)
        theta_temp = theta[j - 1] + dt

        if theta_temp > np.pi / 4:
            break
        else:
            theta[j] = theta_temp
            last_idx = j

    theta = theta[:last_idx + 1]

    point_fw = angle2points(theta[:critical + 1], scale, epsilon)
    point_bw = np.flip(
        angle2points(theta[critical + 1:last_idx + 1], np.flip(scale), epsilon),
        axis=(0, 1)
    )

    point = np.concatenate((point_fw, point_bw), axis=1)
    point = np.concatenate((
        point,
        np.flip(point[:, :-1], axis=1) * np.array([[-1], [1]]),
        point[:, 1:] * np.array([[-1], [-1]]),
        np.flip(point[:, :-1], axis=1) * np.array([[1], [-1]])
    ), axis=1)

    return point

import numpy as np

import numpy as np

def sample_superquadric_surface_parametric(theta, n_eta=25, n_omega=40):
    e1, e2 = float(theta[0]), float(theta[1])
    a1, a2, a3 = abs(float(theta[2])), abs(float(theta[3])), abs(float(theta[4]))

    eta = np.linspace(-np.pi/2, np.pi/2, n_eta)
    omega = np.linspace(-np.pi, np.pi, n_omega, endpoint=False)

    ETA, OMEGA = np.meshgrid(eta, omega, indexing='ij')

    def spow(v, p):
        return np.sign(v) * (np.abs(v) ** p)

    ceta = np.cos(ETA)
    seta = np.sin(ETA)
    comega = np.cos(OMEGA)
    somega = np.sin(OMEGA)

    x = a1 * spow(ceta, e1) * spow(comega, e2)
    y = a2 * spow(ceta, e1) * spow(somega, e2)
    z = a3 * spow(seta, e1)

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts

import numpy as np

def sample_superquadric_projected(theta, n_u=25, n_v=25, axes=('x', 'y', 'z')):
    """
    Generic SQ surface sampler using orthographic projection from ±x, ±y, ±z.
    Works for any standard superquadric and gives good coverage of face centers.

    theta = [e1, e2, a1, a2, a3, ...]

    Returns:
        pts: (N,3)
    """
    e1, e2 = float(theta[0]), float(theta[1])
    a1, a2, a3 = abs(float(theta[2])), abs(float(theta[3])), abs(float(theta[4]))

    e1 = max(abs(e1), 1e-9)
    e2 = max(abs(e2), 1e-9)

    pts_all = []

    def add_pts(P):
        if len(P) > 0:
            pts_all.append(P)

    # ---------- project along z: solve z from (x,y)
    if 'z' in axes:
        x_vals = np.linspace(-a1, a1, n_u)
        y_vals = np.linspace(-a2, a2, n_v)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

        q = (np.abs(X / a1) ** (2.0 / e2) + np.abs(Y / a2) ** (2.0 / e2))
        base = 1.0 - (q ** (e2 / e1))
        mask = base >= -1e-12
        base = np.maximum(base, 0.0)

        Z = a3 * (base ** (e1 / 2.0))

        add_pts(np.stack([X[mask], Y[mask],  Z[mask]], axis=1))
        add_pts(np.stack([X[mask], Y[mask], -Z[mask]], axis=1))

    # ---------- project along x: solve x from (y,z)
    if 'x' in axes:
        y_vals = np.linspace(-a2, a2, n_u)
        z_vals = np.linspace(-a3, a3, n_v)
        Y, Z = np.meshgrid(y_vals, z_vals, indexing='xy')

        cz = np.abs(Z / a3) ** (2.0 / e1)
        base = 1.0 - cz
        mask0 = base >= -1e-12
        base = np.maximum(base, 0.0)

        rhs = (base ** (e1 / e2)) - (np.abs(Y / a2) ** (2.0 / e2))
        mask = mask0 & (rhs >= -1e-12)
        rhs = np.maximum(rhs, 0.0)

        X = a1 * (rhs ** (e2 / 2.0))

        add_pts(np.stack([ X[mask], Y[mask], Z[mask]], axis=1))
        add_pts(np.stack([-X[mask], Y[mask], Z[mask]], axis=1))

    # ---------- project along y: solve y from (x,z)
    if 'y' in axes:
        x_vals = np.linspace(-a1, a1, n_u)
        z_vals = np.linspace(-a3, a3, n_v)
        X, Z = np.meshgrid(x_vals, z_vals, indexing='xy')

        cz = np.abs(Z / a3) ** (2.0 / e1)
        base = 1.0 - cz
        mask0 = base >= -1e-12
        base = np.maximum(base, 0.0)

        rhs = (base ** (e1 / e2)) - (np.abs(X / a1) ** (2.0 / e2))
        mask = mask0 & (rhs >= -1e-12)
        rhs = np.maximum(rhs, 0.0)

        Y = a2 * (rhs ** (e2 / 2.0))

        add_pts(np.stack([X[mask],  Y[mask], Z[mask]], axis=1))
        add_pts(np.stack([X[mask], -Y[mask], Z[mask]], axis=1))

    if not pts_all:
        return np.zeros((0, 3))

    pts = np.vstack(pts_all)
    pts = np.unique(np.round(pts, 10), axis=0)
    return pts
def sample_superquadric_rings_with_faces(theta, n_rings=5, n_points_ring=40,
                                         include_poles=True, axis='z',
                                         add_face_samples=True,
                                         n_eta=20, n_omega=30):
    pts_rings = sample_superquadric_rings(theta, n_rings, n_points_ring, include_poles, axis)

    if not add_face_samples:
        return pts_rings

    pts_faces = sample_superquadric_surface_parametric(theta, n_eta=n_eta, n_omega=n_omega)

    pts = np.vstack([pts_rings, pts_faces])

    # optional dedup
    pts = np.unique(np.round(pts, 6), axis=0)
    return pts

def sample_superquadric_rings(theta, n_rings=5, n_points_ring=40, include_poles=True, axis='z'):
    e1, e2 = float(theta[0]), float(theta[1])
    a1, a2, a3 = abs(float(theta[2])), abs(float(theta[3])), abs(float(theta[4]))

    eps = 1e-12
    e1 = max(e1, eps)
    e2 = max(e2, eps)

    all_pts = []

    def make_ring_levels(radius):
        vals = np.linspace(-radius, radius, n_rings + 2)[1:-1]
        vals = np.concatenate([vals, [0.0]])
        vals = np.unique(np.round(vals, 12))
        vals.sort()
        return vals

    def signed_power(v, p):
        return np.sign(v) * (np.abs(v) ** p)

    def sample_xy_ring_at_z(z, n):
        """
        Correct for z = const:
        xy section is a scaled superellipse.
        """
        r = min(abs(z) / a3, 1.0)
        base = 1.0 - (r ** (2.0 / e1))
        if base <= 0.0:
            return np.zeros((0, 3))

        scale = base ** (e1 / 2.0)

        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        c = np.cos(t)
        s = np.sin(t)

        x = a1 * scale * signed_power(c, e2)
        y = a2 * scale * signed_power(s, e2)
        z_arr = np.full_like(x, z)

        return np.vstack([x, y, z_arr]).T

    def sample_xz_ring_at_y(y, n):
        """
        Correct for y = const:
        solve x as a function of z from the implicit equation.
        """
        cy = abs(y / a2) ** (2.0 / e2)

        # maximum |z| occurs at x = 0
        zmax_base = 1.0 - abs(y / a2) ** (2.0 / e1)
        if zmax_base < 0.0:
            return np.zeros((0, 3))

        zmax = a3 * (zmax_base ** (e1 / 2.0))
        if zmax < 1e-12:
            return np.array([[0.0, y, 0.0]])

        z_pos = np.linspace(0.0, zmax, max(2, n // 4), endpoint=True)

        pts = []

        for z in z_pos:
            cz = abs(z / a3) ** (2.0 / e1)
            rhs = (max(0.0, 1.0 - cz) ** (e1 / e2)) - cy

            if rhs < -1e-12:
                continue

            rhs = max(rhs, 0.0)
            xabs = a1 * (rhs ** (e2 / 2.0))

            # 4 symmetric points for this (|x|, |z|)
            candidates = [
                [ xabs, y,  z],
                [-xabs, y,  z],
            ]
            if z > 1e-12:
                candidates += [
                    [ xabs, y, -z],
                    [-xabs, y, -z],
                ]

            pts.extend(candidates)

        pts = np.asarray(pts, dtype=float)

        # remove duplicates and optionally downsample
        if len(pts) == 0:
            return np.zeros((0, 3))

        pts = np.unique(np.round(pts, 12), axis=0)

        if len(pts) > n:
            idx = np.linspace(0, len(pts) - 1, n).astype(int)
            pts = pts[idx]

        return pts

    def sample_yz_ring_at_x(x, n):
        """
        Correct for x = const:
        solve y as a function of z from the implicit equation.
        """
        cx = abs(x / a1) ** (2.0 / e2)

        # maximum |z| occurs at y = 0
        zmax_base = 1.0 - abs(x / a1) ** (2.0 / e1)
        if zmax_base < 0.0:
            return np.zeros((0, 3))

        zmax = a3 * (zmax_base ** (e1 / 2.0))
        if zmax < 1e-12:
            return np.array([[x, 0.0, 0.0]])

        z_pos = np.linspace(0.0, zmax, max(2, n // 4), endpoint=True)

        pts = []

        for z in z_pos:
            cz = abs(z / a3) ** (2.0 / e1)
            rhs = (max(0.0, 1.0 - cz) ** (e1 / e2)) - cx

            if rhs < -1e-12:
                continue

            rhs = max(rhs, 0.0)
            yabs = a2 * (rhs ** (e2 / 2.0))

            candidates = [
                [x,  yabs,  z],
                [x, -yabs,  z],
            ]
            if z > 1e-12:
                candidates += [
                    [x,  yabs, -z],
                    [x, -yabs, -z],
                ]

            pts.extend(candidates)

        pts = np.asarray(pts, dtype=float)

        if len(pts) == 0:
            return np.zeros((0, 3))

        pts = np.unique(np.round(pts, 12), axis=0)

        if len(pts) > n:
            idx = np.linspace(0, len(pts) - 1, n).astype(int)
            pts = pts[idx]

        return pts

    if axis == 'z':
        z_vals = make_ring_levels(a3)
        for z in z_vals:
            ring = sample_xy_ring_at_z(z, n_points_ring)
            if len(ring) > 0:
                all_pts.append(ring)

        pts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))

        if include_poles:
            pts = np.vstack([pts, [[0.0, 0.0, a3], [0.0, 0.0, -a3]]])

    elif axis == 'y':
        y_vals = make_ring_levels(a2)
        for y in y_vals:
            ring = sample_xz_ring_at_y(y, n_points_ring)
            if len(ring) > 0:
                all_pts.append(ring)

        pts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))

        if include_poles:
            pts = np.vstack([pts, [[0.0, a2, 0.0], [0.0, -a2, 0.0]]])

    elif axis == 'x':
        x_vals = make_ring_levels(a1)
        for x in x_vals:
            ring = sample_yz_ring_at_x(x, n_points_ring)
            if len(ring) > 0:
                all_pts.append(ring)

        pts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))

        if include_poles:
            pts = np.vstack([pts, [[a1, 0.0, 0.0], [-a1, 0.0, 0.0]]])

    else:
        raise ValueError("Wrong axis selection. Select 'x', 'y', or 'z'.")

    return pts
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
        pts2 = uniformSampledSuperellipse(e1, [a1, a3], threshold, num_limit, arclength)  # (2,M)
        if method == "index":
            pts2 = downsample_equally_by_index(pts2, n_points)
        else:
            pts2 = downsample_equally_by_arclength(pts2, n_points)
        x, z = pts2[0], pts2[1]
        y = np.zeros_like(x)

    elif plane == "x":
        pts2 = uniformSampledSuperellipse(e1, [a2, a3], threshold, num_limit, arclength)
        pts2 = downsample_equally_by_index(pts2, n_points) if method=="index" else downsample_equally_by_arclength(pts2, n_points)
        y, z = pts2[0], pts2[1]
        x = np.zeros_like(y)

    elif plane == "z":
        pts2 = uniformSampledSuperellipse(e2, [a1, a2], threshold, num_limit, arclength)
        pts2 = downsample_equally_by_index(pts2, n_points) if method=="index" else downsample_equally_by_arclength(pts2, n_points)
        x, y = pts2[0], pts2[1]
        z = np.zeros_like(x)

    else:
        raise ValueError("plane must be 'x', 'y', or 'z'.")

    return np.vstack([x, y, z]).T  # (n_points, 3)

def sq_F_local(points, theta):
    e1, e2 = float(theta[0]), float(theta[1])
    a1 = max(abs(float(theta[2])), 1e-6)
    a2 = max(abs(float(theta[3])), 1e-6)
    a3 = max(abs(float(theta[4])), 1e-6)

    x = points[:, 0] / a1
    y = points[:, 1] / a2
    z = points[:, 2] / a3

    term1 = (np.abs(x)**(2.0/e2) + np.abs(y)**(2.0/e2))**(e2/e1)
    term2 = (np.abs(z)**(2.0/e1))
    return term1 + term2  # surface: F==1
  

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
