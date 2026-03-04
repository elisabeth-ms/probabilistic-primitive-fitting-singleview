from __future__ import annotations

from pathlib import Path

from sklearn.cluster import DBSCAN
import open3d as o3d
from collections import defaultdict
import re, datetime as dt
import re, yaml, numpy as np
import torch
from mayavi import mlab
import plot_functions
import tools

import math

import numpy as np;
from typing import Tuple, Dict

FX = 554.254691191187
FY = 554.254691191187
CX = 320.5
CY = 240.5

scene_ = "scene_48"
method_ = "ems"
number_samples_per_ray_ = 300
sampled_by_shape = []
device = 'cuda'
base_path = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/results/check"
# base_path = "/home/elisabeth/repos/EMS-superquadric_fitting/results_EMS/check"
scene_dir = Path(base_path) / scene_

point_cloud = tools.read_with_open3d("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/pcds/"+scene_+"/cloud.pcd")
point_cloud = tools.remove_close_points(point_cloud, 0.003)
filtered_points, plane_points, plane_model = tools.remove_largest_plane(point_cloud, distance_threshold=0.003)
filtered_points, plane_points1, plane_model1 = tools.remove_largest_plane(filtered_points, distance_threshold=0.003)
point_cloud = tools.filter_by_z(point_cloud, -np.inf, 1.5)

all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor

labels, id2color = tools.load_seg_as_labels("/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes/segmasks/"+scene_+"/gtseg_ord-nearest_first_step-0.png", inflate_px=5)
point_labels = tools.label_points_from_seg(filtered_points, labels, FX, FY, CX, CY)
clusters = tools.split_points_by_label(filtered_points, point_labels)

# clusters_pruned, rep = tools.prune_clusters_like(
#     clusters,
#     dbscan_min_samples=20,
#     keep_quantile=0.95,
#     min_points_after=30,
#     gap_min=0.01,        # 5 cm gap to drop tiny islands
#     rel_size_max=0.30    # drop components <20% of main if also far
# )

# clusters = clusters_pruned
fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
mlab.view(azimuth=108.51, elevation=168.97, distance=0.5805, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.864720847838999), roll=-177.93)

plot_functions.showPoints(plane_points, scale_factor=0.0025, color=(0.702, 0.702, 0.702), figure=fig)
plot_functions.showPoints(filtered_points, scale_factor=0.0025, color=(0.702, 0.702, 0.702), figure=fig)

for lid, cluster in clusters.items():
    # if lid ==4 or lid==2 or lid==3 or lid==5:
    #     continue

    c = tools.read_cluster_yaml(base_path, scene_, cluster_id=lid, which="latest")
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
            plot_functions.showSuperquadrics(theta,b, alpha)
        elif shape_type == "supertoroid":
            plot_functions.showSupertoroid(theta)
        else:
            plot_functions.showTaperedSuperparaboloidWithBase(theta,k)

mlab.show()
