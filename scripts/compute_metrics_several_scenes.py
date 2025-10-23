from pathlib import Path
import numpy as np
import yaml, csv, time
import torch
import tools
import plot_functions
# -----------------------------
# Config you’ll pass in
# -----------------------------
# scenes = ["scene_101","scene_104","scene_122", ...]
# base_path_results = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/results/check"
# base_path_data    = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes"
# τ’s in meters:

TAU_5 = 0.006
TAU_10 = 0.010
# --- Intrinsics---
FX = 554.254691191187
FY = 554.254691191187
CX = 320.5
CY = 240.5
number_samples_per_ray_ = 300
plane_normal = None
dp = None
method_ ="ems"

def _to_float(x):
    try: return float(x)
    except: return None

def _as_np(x):
    import numpy as np
    return np.asarray(x)

def _device_like(points_np):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_cluster_metrics_from_yaml(cluster_pts_np, shapes, device, tau5=TAU_5, tau10=TAU_10, all_points=None, plane_normal=None, dp=None):
    """
    shapes: list of dicts with keys: type, theta, k, ...
    returns dict with rmse, mae, cov5, cov10, N, n_shapes
    """
    N = cluster_pts_np.shape[0]
    if N == 0:
        return dict(rmse=None, mae=None, cov5=None, cov10=None, N=0, n_shapes=len(shapes))
    
    dmins_list   = []   # list of (N,) tensors       — per-shape nearest distances
    qmins_list   = []   # list of (N,3) tensors     — per-shape nearest sampled points (world)
    idx_list     = []   # list of (N,) long tensors — per-shape indices into that shape's samples
    shape_ids    = []   # list of ints              — index in shapes_list (or your own id)

    dmins_list_object   = []   # list of (N,) tensors       — per-shape nearest distances
    qmins_list_object   = []   # list of (N,3) tensors     — per-shape nearest sampled points (world)
    idx_list_object     = []   # list of (N,) long tensors — per-shape indices into that shape's samples
    shape_ids_object    = []   # list of ints              — index in shapes_list (or your own id)
    
    thetas_sq = []           # list[torch.Tensor] where each is shape (11,)
    sq_indices = []          # keep mapping to shapes_list indices if you need it
    
    pts = torch.tensor(cluster_pts_np, dtype=torch.float32, device=device)
    all_dists = []
    free_total = 0.0
    table_loss_total = 0.0
    penn_table_mean = 0.0
    for s in shapes:
        typ = s["type"]
        theta = torch.tensor(_as_np(s["theta"]), dtype=torch.float32, device=device)
        if typ == "superquadric":
            d = tools.sq_distances(pts, theta).unsqueeze(1)
            free_val= tools.compute_free_space(number_samples_per_ray_, all_points, pts, theta, None, None, typ)
    
            table_pts = tools.make_table_grid_points(
                        theta=theta,
                        plane_normal=plane_normal,
                        plane_d=torch.tensor(float(dp), device=theta.device),
                        half_size=1.0,      # ±1 m in both in-plane directions
                        step=0.002,          # 2 cm spacing; adjust as you like
                        offset_above=0.01,    # or e.g. 0.005 to sit 5 mm above the plane
            )
                
            table_loss= tools.table_transverse_loss(table_pts, theta)
            free_total += float(free_val.detach().cpu())    # <- convert to float
            table_loss_total += float(table_loss.detach().cpu())    # <- convert to float
            penn_table_val = tools.table_violation_metrics(table_pts, theta)
            penn_table_mean+=penn_table_val
            
            if method_ =='ems':
                points = torch.tensor(cluster_pts_np, dtype=torch.float32, device='cuda')
                pts_world = tools.sample_superquadric_points_torch(_as_np(s["theta"]), b=s.get("b", 0.0) or 0.0, alpha=s.get("alpha", 0.0) or 0.0,
                                                    threshold=1e-2, num_limit=15000, arclength=0.003, device='cuda')
                
                dmin, idx, qmin = tools.nearest_on_superquadric(points, pts_world)
                
                # 3) store
                dmins_list.append(dmin)
                qmins_list.append(qmin)
                idx_list.append(idx)


                print("dmin: ", dmin)
                                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                thetas_sq.append(theta)

        elif typ == "supertoroid":
            d = tools.st_distances(pts, theta).unsqueeze(1)
            free_val= tools.compute_free_space(number_samples_per_ray_, all_points, pts, theta, None, None, typ)
            free_total += float(free_val.detach().cpu())    # <- convert to float
            table_pts = tools.make_table_grid_points(
                        theta=theta,
                        plane_normal=plane_normal,
                        plane_d=torch.tensor(float(dp), device=theta.device),
                        half_size=1.0,      # ±1 m in both in-plane directions
                        step=0.01,          # 2 cm spacing; adjust as you like
                        offset_above=0.01,    # or e.g. 0.005 to sit 5 mm above the plane
            )
            
            table_loss = tools.table_transverse_loss_toroid(table_pts, theta)
            table_loss_total += float(table_loss.detach().cpu())    # <- convert to float
            penn_table_val = tools.table_violation_metrics(table_pts, theta)
            penn_table_mean+=penn_table_val

            if method_ =='ems':
                points = torch.tensor(cluster_pts_np, dtype=torch.float32, device='cuda')
                pts_world = tools.sample_supertoroid_points_torch(_as_np(s["theta"]), b=s.get("b", 0.0) or 0.0, alpha=s.get("alpha", 0.0) or 0.0,
                                                    threshold=1e-2, num_limit=10000, arclength=0.003, device='cuda')
                
                dmin, idx, qmin = tools.nearest_on_superquadric(points, pts_world)
                
                # 3) store
                dmins_list.append(dmin)
                qmins_list.append(qmin)
                idx_list.append(idx)

                
                print("dmin: ", dmin)
                                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                thetas_sq.append(theta)


        else:  # superparaboloid (tapered)
            k = torch.tensor(_to_float(s["k"]), dtype=torch.float32, device=device)
            d = tools.spb_distances_autograd(pts, theta, k).unsqueeze(1)
            if method_ =='ems':
                points = torch.tensor(cluster_pts_np, dtype=torch.float32, device='cuda')
                pts_world = tools.sample_tapered_superparaboloid_points_torch(_as_np(s["theta"]), _to_float(s["k"]), b=s.get("b", 0.0) or 0.0, alpha=s.get("alpha", 0.0) or 0.0,
                                                    threshold=1e-2, num_limit=10000, arclength=0.003, device='cuda')
                
                dmin, idx, qmin = tools.nearest_on_superquadric(points, pts_world)
                
                # 3) store
                dmins_list.append(dmin)
                qmins_list.append(qmin)
                idx_list.append(idx)

                
                print("dmin: ", dmin)
                                
                pts_world_np = np.ascontiguousarray(pts_world.detach().cpu().numpy())
                
                thetas_sq.append(theta)
        all_dists.append(d)

    d_all = torch.cat(all_dists, dim=1)               # (N, Mshapes)
    d_min, _ = torch.min(d_all, dim=1)                # (N,)
    penn_table_mean = penn_table_mean/len(shapes)
    if method_== 'ems':
        D = torch.stack(dmins_list, dim=1)   # (N, M)
        Q = torch.stack(qmins_list, dim=1)   # (N, M, 3)
        chosen_d, chosen_q, chosen_m = tools.select_outside_nearest_from_samples(D, Q, thetas_sq)
        tau_5, tau_10 = 0.005, 0.010
        cov5  = (chosen_d < tau_5 ).float().mean().item()
        cov10 = (chosen_d < tau_10).float().mean().item()
        rmse = torch.sqrt((chosen_d**2).mean()).item()
        mae  = torch.mean(torch.abs(chosen_d)).item()
        free_inv = free_total

    else:
        rmse = torch.sqrt(torch.mean(d_min**2)).item()
        mae  = torch.mean(torch.abs(d_min)).item()
        cov5 = (d_min < tau5).float().mean().item()
        cov10 = (d_min < tau10).float().mean().item()
        free_inv = free_total

    return dict(rmse=rmse, mae=mae, cov5=cov5, cov10=cov10, N=N, n_shapes=len(shapes), free=free_inv, table_loss=table_loss_total, penn_table_mean=penn_table_mean)

def iterate_scene_clusters(scene_id, base_path_results, base_path_data, inflate_px=10):
    """
    Yields tuples: (cluster_id, object_name, cluster_points_np, shapes_list)
    shapes_list is exactly the list stored in the YAML.
    """
    scene_dir = Path(base_path_results) / scene_id

    # --- rebuild clusters from data (like your current code) ---
    pcd_path = f"{base_path_data}/pcds/{scene_id}/cloud.pcd"
    seg_path = f"{base_path_data}/segmasks/{scene_id}/gtseg_ord-nearest_first_step-0.png"

    point_cloud = tools.read_with_open3d(pcd_path)
    point_cloud = tools.remove_close_points(point_cloud, 0.003)
    filtered_points, plane_points, plane_model = tools.remove_largest_plane(point_cloud, distance_threshold=0.003)
    filtered_points, plane_points1, plane_model1 = tools.remove_largest_plane(filtered_points, distance_threshold=0.003)
    filtered_points = tools.filter_by_z(filtered_points, -np.inf, 1.5)
    all_points = torch.from_numpy(point_cloud).float().cuda()         # convert to CUDA tensor

    labels, id2color = tools.load_seg_as_labels(seg_path, inflate_px=inflate_px)
    point_labels = tools.label_points_from_seg(filtered_points, labels, FX, FY, CX, CY)
    clusters = tools.split_points_by_label(filtered_points, point_labels)
    
    ap,bp,cp,dp = plane_model
    plane_normal = torch.tensor([ap,bp,cp], dtype=torch.float32, device='cuda')
    

    # --- read latest YAML for each cluster and yield ---
    for lid, cluster in clusters.items():
        try:
            c = tools.read_cluster_yaml(base_path_results, scene_id, cluster_id=lid, which="latest")
        except Exception as e:
            print(f"[WARN] No YAML for {scene_id} cluster {lid}: {e}")
            continue
        shapes_list = c.get("shapes", [])
        object_name = c.get("object", f"cluster{lid}")
        yield lid, object_name, cluster, shapes_list, all_points, plane_normal, dp

def aggregate_history_by_object(scenes, base_path_results, base_path_data,
                                tau5=TAU_5, tau10=TAU_10,
                                out_dir=None):
    """
    Returns: history dict (object -> list of records) and summary dict.
    Also saves YAML/CSV if out_dir is provided.
    """
    device = _device_like(np.zeros((1,3),dtype=np.float32))
    history = {}   # object -> list of {scene, cluster, rmse, mae, cov5, cov10, N, n_shapes}
    t0 = time.time()

    for s in scenes:
        print(f"\n=== {s} ===")
        for lid, obj, cluster_pts, shapes, all_points, plane_normal, dp in iterate_scene_clusters(s, base_path_results, base_path_data):
            if len(shapes) == 0 or cluster_pts.shape[0] == 0:
                continue
            metrics = compute_cluster_metrics_from_yaml(cluster_pts, shapes, device, tau5, tau10, all_points, plane_normal, dp)
            rec = dict(scene=s, cluster_id=int(lid), **metrics)
            history.setdefault(obj, []).append(rec)
            print(f"  {obj:20s}  c{lid:>2}  N={metrics['N']:4d}  rmse={metrics['rmse']:.4f}  "
                  f"mae={metrics['mae']:.4f}  cov5={metrics['cov5']:.3f}  cov10={metrics['cov10']:.3f}  free={metrics['free']:.8f} table_loss={metrics['table_loss']:.8f} penn_table_mean={metrics['penn_table_mean']:.8f}")

    # --- build summary per object ---
    def _stats(vals):
        import numpy as np
        a = np.asarray(vals, dtype=float)
        if a.size == 0: return dict(mean=None, std=None, min=None, max=None, n=0)
        return dict(mean=float(np.nanmean(a)),
                    std=float(np.nanstd(a)),
                    min=float(np.nanmin(a)),
                    max=float(np.nanmax(a)),
                    n=int(a.size))

    summary = {}
    for obj, recs in history.items():
        rmse_vals = [r["rmse"] for r in recs if r["rmse"] is not None]
        mae_vals  = [r["mae"] for r in recs if r["mae"] is not None]
        cov5_vals = [r["cov5"] for r in recs if r["cov5"] is not None]
        cov10_vals= [r["cov10"] for r in recs if r["cov10"] is not None]
        N_vals    = [r["N"] for r in recs if r["N"] is not None]
        free_vals = [r["free"] for r in recs if r["free"] is not None]
        table_loss = [r["table_loss"] for r in recs if r["table_loss"] is not None]
        penn_table_mean = [r["penn_table_mean"] for r in recs if r["penn_table_mean"] is not None]

        summary[obj] = dict(
            appearances=len(recs),
            N=_stats(N_vals),
            rmse=_stats(rmse_vals),
            mae=_stats(mae_vals),
            cov5=_stats(cov5_vals),
            cov10=_stats(cov10_vals),
            free=_stats(free_vals),
            table_loss=_stats(table_loss),
            penn_table_mean=_stats(penn_table_mean)
        )

    # --- save ---
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "objects_history.yaml").write_text(yaml.safe_dump(history, sort_keys=False))
        (out_dir / "objects_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

        # also a flat CSV
        csv_path = out_dir / "objects_history.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["object","scene","cluster_id","N","n_shapes","rmse","mae","cov5","cov10", "free", "table_loss", "penn_table_mean"])
            for obj, recs in history.items():
                for r in recs:
                    w.writerow([obj, r["scene"], r["cluster_id"], r["N"], r["n_shapes"],
                                r["rmse"], r["mae"], r["cov5"], r["cov10"], r["free"], r["table_loss"], r["penn_table_mean"]])
        print(f"\nSaved history/summary in: {out_dir}")

    print(f"\nDone in {time.time()-t0:.1f}s — objects: {len(history)}")
    return history, summary
  
scenes = ["scene_10","scene_25", "scene_27", "scene_33", "scene_36", "scene_38", "scene_39", "scene_48", "scene_56", "scene_77", "scene_83", "scene_84", "scene_104", "scene_122",
          "scene_130", "scene_141", "scene_148"]

base_path_results = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/results/check"
base_path_data    = "/home/elisabeth/repos/ProbabilisticSuperquadricFitting/data/sceneReplica/final_scenes"

# base_path_results = "/home/elisabeth/repos/EMS-superquadric_fitting/results_EMS/check"
out_dir = f"{base_path_results}/_metrics"

history, summary = aggregate_history_by_object(
    scenes,
    base_path_results,
    base_path_data,
    tau5=0.005, tau10=0.010,
    out_dir=out_dir
)
