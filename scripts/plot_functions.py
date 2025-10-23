import numpy as np
import tools
from mayavi import mlab

import torch

def _to_numpy3(x):
    """Acepta torch/numpy, mueve a CPU si hace falta y devuelve (N,3) float64."""
    if torch.is_tensor(x):
        x = x.detach().to('cpu').contiguous().numpy()
    else:
        x = np.asarray(x)
    x = np.ascontiguousarray(x, dtype=np.float64)
    assert x.ndim == 2 and x.shape[1] == 3, f"Esperaba (N,3), got {x.shape}"
    return x

def show_vectors(P, Q, color=(1, 0, 0), mode='arrow',
                 every=20, scale_factor=1.0, tube_radius=None, figure=None):
    """
    Dibuja vectores desde P -> Q (admite torch CUDA/CPU o numpy).
    - P: (N,3) puntos del objeto
    - Q: (N,3) puntos correspondientes en la forma (p.ej. qmin)
    - mode: 'arrow' (quiver3d) o 'line' (plot3d con tubos)
    - every: submuestreo (dibuja 1 de cada 'every' vectores)
    """
    P = _to_numpy3(P)
    Q = _to_numpy3(Q)

    # Submuestreo ANTES de crear U para ahorrar RAM/tiempo
    if every is None or every < 1: every = 1
    P = P[::every]
    Q = Q[::every]
    U = Q - P

    if figure is None:
        figure = mlab.figure(size=(600, 600), bgcolor=(1, 1, 1))

    if mode == 'arrow':
        mlab.quiver3d(
            P[:, 0], P[:, 1], P[:, 2],
            U[:, 0], U[:, 1], U[:, 2],
            mode='arrow', scale_mode='vector', scale_factor=scale_factor,
            color=color, figure=figure
        )
    else:
        # líneas/tubos (más pesado)
        if tube_radius is None:
            tube_radius = 0.0007  # ~0.7 mm por defecto
        for p, q in zip(P, Q):
            mlab.plot3d([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                        tube_radius=tube_radius, color=color, figure=figure)
    return figure


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
    RotM = tools.build_rotation_matrix_numpy(x)
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
            point_temp = point_temp@RotM.T + tools.get_translation_numpy(x)

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

    RotM = tools.build_rotation_matrix_numpy(x)      # your function
    t    = tools.get_translation_numpy(x)            # your function

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
    x = np.asarray(x).flatten()

    # Avoid numerical issues
    x[0] = max(x[0], 0.007)
    x[1] = max(x[1], 0.007)

    e1, e2 = x[0], x[1]
    a1, a2, a3 = x[2], x[3], x[4]
    rot = tools.build_rotation_matrix_numpy(x)
    trans = tools.get_translation_numpy(x)

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
