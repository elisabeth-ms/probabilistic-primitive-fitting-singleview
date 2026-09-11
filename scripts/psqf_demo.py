import psqf
from pathlib import Path
import numpy as np
import torch
from collections import deque
from mayavi import mlab
from collections import defaultdict
import re, datetime as dt


################ PARAMETERS ###############
scene_ = "scene_25"
N_ref_ = 1000
base_lr_ = 1e-3
T_ = 2000
K_=3
freeze_every_ = T_
sigma_momentum_ = 0.0    # EMA for sigma2
sigma_every_ = 2        # update cadence
lambda_free_ = 120.0
lambda_transverse_table_ = 10.0
lambda_mass_ = 0.0
w0_=0.05
w_final_ = 0.35
ramp_start_ = 0.7
T_supertoroid_ = 100
lambda_free_supertoroid_ = 30.0
lambda_transverse_table_supertoroid_ = 10.0

T_superparaboloid_ = 2000
number_samples_per_ray_ = 200
weight_decay_ = 0.01

params = {
    "scene_": scene_,
    "N_ref_": N_ref_,
    "base_lr_": base_lr_,
    "T_": T_,
    "K_": K_,
    "freeze_every_": freeze_every_,
    "sigma_momentum_": sigma_momentum_,
    "sigma_every_": sigma_every_,
    "lambda_free_": lambda_free_,
    "lambda_transverse_table_": lambda_transverse_table_,
    "lambda_mass_": lambda_mass_,
    "w0_": w0_,
    "w_final_": w_final_,
    "ramp_start_": ramp_start_,
    "T_supertoroid_": T_supertoroid_,
    "lambda_free_supertoroid_": lambda_free_supertoroid_,
    "T_superparaboloid_": T_superparaboloid_,
    "number_samples_per_ray_": number_samples_per_ray_,
    "weight_decay_": weight_decay_,
}

base_path = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/results"

scene_dir = Path(base_path) / scene_
scene_dir.mkdir(parents=True, exist_ok=True)

try:
    test_n  # noqa: F821
except NameError:
    test_n = psqf._next_test_num(scene_dir)
    
out_dir = scene_dir / f"test{test_n}"
out_dir.mkdir(exist_ok=True)

# --- write params once (won't overwrite if already present)
params_path = out_dir / "params.yaml"
if not params_path.exists():
    run_params = {
        "scene_": scene_,
        "N_ref_": N_ref_,
        "base_lr_": base_lr_,
        "T_": T_,
        "K_": K_,
        "freeze_every_": freeze_every_,
        "sigma_momentum_": sigma_momentum_,
        "sigma_every_": sigma_every_,
        "lambda_free_": lambda_free_,
        "lambda_transverse_table_": lambda_transverse_table_,
        "lambda_mass_": lambda_mass_,
        "w0_": w0_,
        "w_final_": w_final_,
        "ramp_start_": ramp_start_,
        "T_supertoroid_": T_supertoroid_,
        "lambda_free_supertoroid_": lambda_free_supertoroid_,
        "T_superparaboloid_": T_superparaboloid_,
        "number_samples_per_ray_": number_samples_per_ray_,
        "weight_decay_": weight_decay_,
    }
    params_path.write_text("# ################ PARAMETERS ###############\n" + psqf._dump(run_params))


# point_cloud = read_ply("data/objects7.ply")
# point_cloud = remove_close_points(point_cloud, 0.005)

# point_cloud = filter_by_z(point_cloud, -np.inf, 1.94)

# all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor



# fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
# showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))

point_cloud = psqf.read_with_open3d("data/sceneReplica/final_scenes/pcds/"+scene_+"/cloud.pcd")
point_cloud = psqf.remove_close_points(point_cloud, 0.003)



filtered_points, plane_points, plane_model = psqf.remove_largest_plane(point_cloud, distance_threshold=0.003)
filtered_points, plane_points1, plane_model1 = psqf.remove_largest_plane(filtered_points, distance_threshold=0.003)
point_cloud = psqf.filter_by_z(point_cloud, -np.inf, 1.0)
all_points = psqf.torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor



fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
psqf.showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))
psqf.showPoints(np.array([[0,0,0]]))
mlab.show()


print("plane model: ", plane_model)

table_normal = torch.tensor(plane_model[:3], dtype=torch.float32, device='cuda')

# points_np: (N,3) from your /head_camera/depth_registered/points (same camera!)
# seg_png: color segmentation aligned with that camera (same resolution)
labels, id2color = psqf.load_seg_as_labels("data/sceneReplica/final_scenes/segmasks/"+scene_+"/gtseg_ord-nearest_first_step-0.png", inflate_px=10)
point_labels = psqf.label_points_from_seg(filtered_points, labels)
clusters = psqf.split_points_by_label(filtered_points, point_labels)

# clusters_pruned, rep = prune_clusters_like(
#     clusters,
#     dbscan_min_samples=20,
#     keep_quantile=0.95,
#     min_points_after=30,
#     gap_min=0.01,        # 5 cm gap to drop tiny islands
#     rel_size_max=0.30    # drop components <20% of main if also far
# )

# # for lab, r in rep.items():
# #     print(f"label {lab}: {r['orig']} -> {r['kept']} (dropped {r['dropped']})")

# clusters = clusters_pruned
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
number_samples_per_ray = 150
number_of_rays = all_points.shape[0]

t_vals = torch.linspace(0.03, 1.1, number_samples_per_ray, device=all_points.device)  # go slightly past the surface
ray_points = camera_origin[:, None, :] + t_vals[None, :, None] * directions[:, None, :]
ray_samples_flat = ray_points.reshape(-1, 3)

k = torch.tensor(0.6)
k = torch.nn.Parameter(k)


all_params_modeled = {}
idx = 0
# for i in range(0, len(clusters)-1):
timing_report = []
for lid, cluster in clusters.items():
    print(lid, cluster.shape)                    # each pts is Nx3
    loss_per_iteration = []
    if lid ==3 or lid ==2 or lid==5 or lid==4:
      continue
    current_cluster = cluster

    sub_idx = 0
    
    theta_np_sq = None
    queue = deque([current_cluster])
    
    while queue:
        current_cluster = queue.popleft()
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        mlab.view(azimuth=108.51, elevation=168.97, distance=0.5, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.7), roll=-177.93)
        psqf.showPoints(current_cluster, scale_factor=0.0025, color = (0.894, 0.447, 0.0))
        psqf.showPoints(point_cloud, scale_factor=0.001, color=(0.702, 0.702, 0.702))
        mlab.show()
        
        if current_cluster.shape[0]<=50:
          continue
        theta_np, k_np, b_np, alpha_np, selected_indices_good, remaining_indices, selected_indices_bad, free_space_penalty, distances, loss_history_se, fit_time_se = psqf.fit_shape_to_cluster(current_cluster, 'superquadric', 
                                                                      plane_model=plane_model, N_ref=N_ref_, all_points=all_points, 
                                                                      number_samples_per_ray=number_samples_per_ray_, base_lr=base_lr_, T=T_, K=K_, sigma_momentum=sigma_momentum_
                                                                      ,sigma_every=sigma_every_, lambda_free=lambda_free_, lambda_transverse_table=lambda_transverse_table_,
                                                                      lambda_mass=lambda_mass_, w0=w0_, w_final=w_final_, ramp_start=ramp_start_, T_supertoroid=T_supertoroid_,
                                                                      lambda_free_supertoroid=lambda_free_supertoroid_, lambda_transverse_table_supertoroid= lambda_transverse_table_supertoroid_,
                                                                      T_superparaboloid=T_superparaboloid_, table_normal=table_normal, weight_decay=weight_decay_)
        print("selected indices good: ", selected_indices_good.shape)
        print("theta_np: ", theta_np)
        print("b_np: ", b_np)
        print("alpha_np: ", alpha_np)
        
        timing_report.append({
            "cluster_id": int(lid),
            "sub_id": int(sub_idx),
            "shape": "superquadric",         # o "supertoroid" / "superparaboloid"
            "n_points": int(current_cluster.shape[0]),
            "elapsed_seconds": fit_time_se,  # o fit_time_st / fit_time_sp
        })
        
        all_params_modeled[idx] = {"cluster": lid,"type": "superquadric", "theta": theta_np, "k": 0, "b": b_np, "alpha": alpha_np, "free_space_penalty":free_space_penalty, 
                                  "indices_good": selected_indices_good, "indices_bad": selected_indices_bad, "indices_remaining": remaining_indices}
        idx +=1
        
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        mlab.view(azimuth=108.51, elevation=168.97, distance=0.5, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.7), roll=-177.93)
        psqf.showPoints(current_cluster[selected_indices_good], scale_factor=0.002, color=(1.0, 0.992, 0.157))
        if selected_indices_bad.size >0:
            psqf.showPoints(current_cluster[selected_indices_bad], scale_factor=0.002, color=(0.224, 0.004, 0.278))
        psqf.showPoints(current_cluster[remaining_indices], scale_factor=0.002, color=(0.243, 0.675, 0.647))
        psqf.showPoints(point_cloud, scale_factor=0.001, color=(0.702, 0.702, 0.702))
        psqf.showSuperquadrics(theta_np, b_np, alpha_np)
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
        for id, params in list(all_params_modeled.items())[-1:]:
          if params["free_space_penalty"]>=0.015 or params["indices_good"].shape[0] == 0:
              print("good: ",params["indices_good"])
              print("Before selected indices good: ", params["indices_good"].shape)
              print("free_space_penalty: ", params["free_space_penalty"])
              # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
              # showPoints(current_cluster[params["indices_good"]], scale_factor=0.01, color=(0,1,0))
              # showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
              # mlab.show()
              
              good1 = psqf.as_idx1d(params["indices_good"])           # from the 1st (SQ) fit, relative to current_cluster
              bad1 = psqf.as_idx1d(params["indices_bad"])
              remaining_indices1  = psqf.as_idx1d(params["indices_remaining"])

              used_indices = None
              # If the first fit had no good points, skip carryover entirely
              
              print("good1.size: ", good1.size)
              if good1.size == 0:
                  remaining_indices_for_toroid = np.union1d(remaining_indices1, bad1)
                  used_indices = remaining_indices_for_toroid
                  sub_pts = current_cluster[used_indices]
              else:
                  used_indices = np.union1d(good1, remaining_indices1)
                  sub_pts = current_cluster[used_indices]                  # pass exactly these to the paraboloid fit
              
              
              # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
              # showPoints(sub_pts, scale_factor=0.01, color=(0,1,0))
              # showPoints(point_cloud, scale_factor=0.005, color=(0,0.5,0.5))
              # mlab.show()
              
              theta_np, k_np, b_np, alpha_np, selected_indices_good2, remaining_indices2, selected_indices_bad2, free_space_penalty, distances_st, loss_history_st, fit_time_st = psqf.fit_shape_to_cluster(sub_pts, 'supertoroid', 
                                                                      plane_model=plane_model, N_ref=N_ref_, all_points=all_points, 
                                                                      number_samples_per_ray=number_samples_per_ray_, base_lr=base_lr_, T=T_, K=K_, sigma_momentum=sigma_momentum_
                                                                      ,sigma_every=sigma_every_, lambda_free=lambda_free_, lambda_transverse_table=lambda_transverse_table_,
                                                                      lambda_mass=lambda_mass_, w0=w0_, w_final=w_final_, ramp_start=ramp_start_, T_supertoroid=T_supertoroid_,
                                                                      lambda_free_supertoroid=lambda_free_supertoroid_, lambda_transverse_table_supertoroid= lambda_transverse_table_supertoroid_,
                                                                      T_superparaboloid=T_superparaboloid_, table_normal=table_normal, weight_decay=weight_decay_)
              distances_st = distances_st[selected_indices_good2]
              
              timing_report.append({
                  "cluster_id": int(lid),
                  "sub_id": int(sub_idx),
                  "shape": "supertoroid",
                  "n_points": int(sub_pts.shape[0]),
                  "elapsed_seconds": fit_time_st,
              })

              d = np.asarray(distances_st).reshape(-1)              # shape (N_sub,)
              idx_st = np.asarray(selected_indices_good2, dtype=np.int64)

              # Limpieza/seguridad por si viene algún índice fuera de rango
              N_st = d.shape[0]
              idx_st = idx_st[(idx_st >= 0) & (idx_st < N_st)]
              if idx_st.size == 0:
                  sum_st_distances_good = float('inf')
                  mean_st_distances_good = float('inf')
              else:
                  # (opcional) quitar duplicados y ordenar
                  idx_st = np.unique(idx_st)
                  sum_st_distances_good = float(np.nansum(d[idx_st]**2))
                  mean_st_distances_good = float(np.nanmean(d[idx_st]**2))
                  J_st, G_st, mse_st = psqf.option1_score(d**2, selected_indices_good2, sub_pts.shape[0])         
              
              
              fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
              mlab.view(azimuth=108.51, elevation=168.97, distance=0.5, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.7), roll=-177.93)
              psqf.showPoints(sub_pts[selected_indices_good2], scale_factor=0.002, color=(1.0, 0.992, 0.157))
              if selected_indices_bad2.size >0:
                  psqf.showPoints(sub_pts[selected_indices_bad2], scale_factor=0.002, color=(0.224, 0.004, 0.278))
              psqf.showPoints(sub_pts[remaining_indices2], scale_factor=0.002, color=(0.243, 0.675, 0.647))
              psqf.showPoints(point_cloud, scale_factor=0.001, color=(0.702, 0.702, 0.702))
              psqf.showSupertoroid(theta_np)
              mlab.show()
              
              
              theta_np_sp, k_np_sp, b_np_sp, alpha_np_sp, selected_indices_good2_sp, remaining_indices2_sp, selected_indices_bad2_sp, free_space_penalty_sp, distances_sp, loss_history_sp, fit_time_sp = psqf.fit_shape_to_cluster(sub_pts, 'superparaboloid', 
                                                                      plane_model=plane_model, N_ref=N_ref_, all_points=all_points, 
                                                                      number_samples_per_ray=number_samples_per_ray_, base_lr=base_lr_, T=T_, K=K_, sigma_momentum=sigma_momentum_
                                                                      ,sigma_every=sigma_every_, lambda_free=lambda_free_, lambda_transverse_table=lambda_transverse_table_,
                                                                      lambda_mass=lambda_mass_, w0=w0_, w_final=w_final_, ramp_start=ramp_start_, T_supertoroid=T_supertoroid_,
                                                                      lambda_free_supertoroid=lambda_free_supertoroid_, lambda_transverse_table_supertoroid= lambda_transverse_table_supertoroid_,
                                                                      T_superparaboloid=T_superparaboloid_, table_normal=table_normal, weight_decay=weight_decay_)

              timing_report.append({
                  "cluster_id": int(lid),
                  "sub_id": int(sub_idx),
                  "shape": "superparaboloid",
                  "n_points": int(sub_pts.shape[0]),
                  "elapsed_seconds": fit_time_sp,
              })
              
              idx_sp = np.asarray(selected_indices_good2_sp, dtype=np.int64)

              # Limpieza/seguridad por si viene algún índice fuera de rango
              N_sp = d.shape[0]
              idx_sp = idx_sp[(idx_sp >= 0) & (idx_sp < N_sp)]
              if idx_sp.size == 0:
                  sum_sp_distances_good = float('inf')
                  mean_sp_distances_good = float('inf')
              else:
                  # (opcional) quitar duplicados y ordenar
                  idx_sp = np.unique(idx_sp)
                  sum_sp_distances_good = float(np.nansum(d[idx_sp]**2))
                  mean_sp_distances_good = float(np.nanmean(d[idx_sp]**2))
                  J_sp, G_sp, mse_sp = psqf.option1_score(d**2, selected_indices_good2_sp, sub_pts.shape[0])     
                  
              
              print("-------------------------------- Superparaboloid ------------------------------")
              print("good sp: ", selected_indices_good2_sp)
              print("bad sp: ", selected_indices_bad2_sp)
              print("remaining sp: ", remaining_indices2_sp)
              
              fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
              mlab.view(azimuth=108.51, elevation=168.97, distance=0.5, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.7), roll=-177.93)
              psqf.showPoints(sub_pts[selected_indices_good2_sp], scale_factor=0.002, color=(1.0, 0.992, 0.157))
              if selected_indices_bad2_sp.size >0:
                  psqf.showPoints(sub_pts[selected_indices_bad2_sp], scale_factor=0.002, color=(0.224, 0.004, 0.278))
              psqf.showPoints(sub_pts[remaining_indices2_sp], scale_factor=0.002, color=(0.243, 0.675, 0.647))
              psqf.showPoints(point_cloud, scale_factor=0.001, color=(0.702, 0.702, 0.702))
              psqf.showTaperedSuperparaboloidWithBase(theta_np_sp, k_np_sp)
              mlab.show()
              


              if J_sp<J_st and selected_indices_good2_sp.size!=0:
                  theta_np = theta_np_sp
                  k_np = k_np_sp
                  alpha_np = alpha_np_sp
                  selected_indices_good2 = selected_indices_good2_sp
                  remaining_indices2 = remaining_indices2_sp
                  selected_indices_bad2 = selected_indices_bad2_sp
                  free_space_penalty = free_space_penalty_sp
              
                  params["type"] = "superparaboloid"
                  params["theta"] = theta_np
                  params["k"] = k_np
                  params["free_space_penalty"]=free_space_penalty
                  
                  loss_history = loss_history_sp
                  loss_payload = {
                      "scene": scene_,
                      "test": int(test_n),
                      "cluster_id": int(lid),
                      "sub_id": int(sub_idx),
                      "shape": "superparaboloid",
                      "n_points": int(current_cluster.shape[0]),
                      "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                      "fit_time_seconds": float(fit_time_sp),
                      "history": loss_history,  # list[dict]
                  }
                  cluster_dir = out_dir / f"cluster{lid}"
                  loss_file = cluster_dir / f"sub{sub_idx}_loss.yaml"
                  psqf._dump_yaml(loss_payload, loss_file)
                  print("Saved loss history:", loss_file)
                  sub_idx +=1
              else:
                  params["type"] = "supertoroid"
                  params["theta"] = theta_np
                  params["free_space_penalty"]=free_space_penalty
                  loss_history = loss_history_st
                  loss_payload = {
                      "scene": scene_,
                      "test": int(test_n),
                      "cluster_id": int(lid),
                      "sub_id": int(sub_idx),
                      "shape": "supertoroid",
                      "n_points": int(current_cluster.shape[0]),
                      "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                      "fit_time_seconds": float(fit_time_st),
                      "history": loss_history,  # list[dict]
                  }
                  cluster_dir = out_dir / f"cluster{lid}"
                  loss_file = cluster_dir / f"sub{sub_idx}_loss.yaml"
                  psqf._dump_yaml(loss_payload, loss_file)
                  print("Saved loss history:", loss_file)
                  sub_idx +=1
              print("J supertoroid: ", J_st)
              print("J superparaboloid: ", J_sp)
              
              
              
              
              selected_indices_good2 = psqf.as_idx1d(selected_indices_good2)                           # indices relative to sub_pts
              selected_indices_bad2  = psqf.as_idx1d(selected_indices_bad2)
              remaining_indices2  = psqf.as_idx1d(remaining_indices2)

              # Map back to original current_cluster:
              selected_indices_good_p = used_indices[selected_indices_good2]
              selected_indices_bad_p  = used_indices[selected_indices_bad2]
              remaining_indices_p  = used_indices[remaining_indices2]
              
              if good1.size == 0:
                  remaining_indices = remaining_indices_p
                  selected_indices_bad = selected_indices_bad2
              else:
                  remaining_indices =  remaining_indices_p
                  selected_indices_bad = np.union1d(selected_indices_bad, selected_indices_bad_p)
              print("remaining_indices: ", remaining_indices)
              print("selected_indices_bad: ", selected_indices_bad)
          else: # superellipsoid
            loss_history = loss_history_se
            loss_payload = {
                "scene": scene_,
                "test": int(test_n),
                "cluster_id": int(lid),
                "sub_id": int(sub_idx),
                "shape": "superquadric",
                "n_points": int(current_cluster.shape[0]),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "fit_time_seconds": float(fit_time_se),
                "history": loss_history,  # list[dict]
            }
            cluster_dir = out_dir / f"cluster{lid}"
            loss_file = cluster_dir / f"sub{sub_idx}_loss.yaml"
            psqf._dump_yaml(loss_payload, loss_file)
            print("Saved loss history:", loss_file)
            sub_idx +=1

      
        # if selected_indices_good.size == 0 and selected_indices_good2.size == 0 and selected_indices_good2_sp.size==0:
        #     continue
        next_indices = np.union1d(remaining_indices, selected_indices_bad)
        residual = current_cluster[next_indices]
        
        fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        mlab.view(azimuth=108.51, elevation=168.97, distance=0.5, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.7), roll=-177.93)
        psqf.showPoints(residual, scale_factor=0.0025, color=(0.894, 0.447, 0.0))
        psqf.showPoints(point_cloud, scale_factor=0.001, color=(0.702, 0.702, 0.702))
        mlab.show()
        print(next_indices.shape)
        # mlab.show()
        # ---- NEW: subcluster residual and enqueue ----
        subclusters = psqf.split_by_distance(residual, eps=1e-2, min_samples=5)
        
        print("subclusters: ", len(subclusters))
        for sub in subclusters:
            if sub.shape[0] > 10:
                queue.append(sub)

#       showTaperedSuperparaboloidWithBase(params['theta'], params['k'])
# showPoints(point_cloud, scale_factor=0.0025, color=(0,0.5,0.5))
# mlab.show()

# --- guardar el resumen de tiempos ---
total_time = sum(t["elapsed_seconds"] for t in timing_report)
by_shape = defaultdict(float)
for t in timing_report:
    by_shape[t["shape"]] += t["elapsed_seconds"]

timing_payload = {
    "scene": scene_,
    "test": int(test_n),
    "total_time_seconds": total_time,
    "time_by_shape_seconds": dict(by_shape),
    "fits": timing_report,
}
timing_path = out_dir / "timing.yaml"
psqf._dump_yaml(timing_payload, timing_path)
print("Saved timing report:", timing_path)

# --- group modeled shapes by cluster from all_params_modeled and dump YAML
by_cluster = defaultdict(list)
for _i, p in all_params_modeled.items():
    lid = int(p["cluster"])
    by_cluster[lid].append({
        "type": p["type"],
        "theta": psqf._to_py(p.get("theta")),
        "k": psqf._to_py(p.get("k")),
        "b": psqf._to_py(p.get("b")),
        "alpha": psqf._to_py(p.get("alpha")),
        "free_space_penalty": psqf._to_py(p.get("free_space_penalty")),
        "indices_good": psqf._flow_list(p.get("indices_good")),
        "indices_bad": psqf._flow_list(p.get("indices_bad")),
        "indices_remaining": psqf._flow_list(p.get("indices_remaining")),
    })                


for lid, shapes in by_cluster.items():
    payload = {
        "scene": scene_,
        "test": int(test_n),
        "cluster_id": int(lid),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "n_shapes": len(shapes),
        "shapes": shapes,
    }
    out_file = out_dir / f"cluster{lid}.yaml"
    out_file.write_text("# ############## CLUSTER RESULTS ##############\n" + psqf._dump(payload))
    print("Saved", out_file)

import sys

def print_camera_view(fig):
    az, el, dist, fp = mlab.view(figure=fig)
    roll = mlab.roll(figure=fig)
    print(f"Azimuth: {az:.2f}  Elevation: {el:.2f}  Distance: {dist:.4f}")
    print(f"Focal Point: {tuple(fp)}")
    print(f"Roll: {roll:.2f}")
    print("-"*50)
    sys.stdout.flush()

# --- fire on interaction (drag/zoom/rotate) ---
def _on_interaction(obj, evt):
    print_camera_view(fig)

def _on_end_interaction(obj, evt):
    print_camera_view(fig)


print(all_params_modeled)

fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
# mlab.figure(fig)  # make it current
# fig.scene.interactor.add_observer("InteractionEvent", _on_interaction)
# Print on every camera move (use 'EndInteractionEvent' to print only when the user releases)
# fig.scene.interactor.add_observer('InteractionEvent', on_interaction)

mlab.view(azimuth=108.51, elevation=168.97, distance=0.5805, focalpoint=(0.14501899292528592, -0.018065290097470238, 0.864720847838999), roll=-177.93)
for idx,params in all_params_modeled.items():
    if params["type"] == "superquadric":
      psqf.showSuperquadrics(params['theta'], params["b"], params["alpha"])
    elif params["type"] == "supertoroid":
      psqf.showSupertoroid(params['theta'])
    else:
      psqf.showTaperedSuperparaboloidWithBase(params["theta"], params["k"])
psqf.showPoints(point_cloud, scale_factor=0.0025, color=(0.702, 0.702, 0.702), figure=fig)
print_camera_view(fig)

mlab.show()


