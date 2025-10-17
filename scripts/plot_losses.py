#!/usr/bin/env python3
"""
Plot per-iteration loss histories saved as YAML.

Folder layout expected (yours):
  <base>/<scene>/test<NUM>/cluster<LID>/sub<SUBID>_loss.yaml

Each YAML has:
  scene, test, cluster_id, sub_id, shape, n_points, timestamp, history=[{iter, fit, free, table, loss, p_mean, p_med, sigma2}, ...]

Usage:
  python plot_losses.py --base /home/.../results_EMS --scene scene_10 --test 3
  # optional:
  #   --out out_dir   : where to write PNGs (default: alongside YAML files)
  #   --clusters 0 2 5: only plot given cluster ids
  #   --no-overview   : skip per-cluster overview grid
"""

import argparse
from pathlib import Path
import yaml
import math

import matplotlib.pyplot as plt

def find_loss_files(base: Path, scene: str, test: int, only_clusters=None):
    root = base / scene / f"test{test}"
    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")
    files = []
    for cdir in sorted(root.glob("cluster*")):
        try:
            lid = int(cdir.name.replace("cluster", ""))
        except Exception:
            continue
        if only_clusters is not None and lid not in only_clusters:
            continue
        for y in sorted(cdir.glob("sub*_loss.yaml")):
            files.append((lid, y))
    return files

def load_history(yaml_path: Path):
    data = yaml.safe_load(yaml_path.read_text())
    hist = data.get("history", [])
    # ensure required keys exist, coerce to numeric
    xs = [int(row.get("iter", i)) for i, row in enumerate(hist)]
    def _f(row, k, default=0.0):
        v = row.get(k, default)
        try: return float(v)
        except Exception: return default
    

    meta = {
        "scene": data.get("scene", ""),
        "test": int(data.get("test", -1)),
        "cluster_id": int(data.get("cluster_id", -1)),
        "sub_id": int(data.get("sub_id", -1)),
        "shape": data.get("shape", ""),
        "n_points": int(data.get("n_points", 0)),
        "timestamp": data.get("timestamp", ""),
    }
    if meta['shape'] == 'superparaboloid':
        ys = {
            "loss":  [_f(r, "loss")  for r in hist],
            "fit":   [_f(r, "fit")   for r in hist],
            "drop":  [_f(r, "drop")  for r in hist],
            "extend": [_f(r, "extend") for r in hist],
            "p_mean":[_f(r, "p_mean")for r in hist],
            "p_med": [_f(r, "p_med") for r in hist],
            "sigma2":[_f(r, "sigma2")for r in hist],
        }
    else:
        ys = {
            "loss":  [_f(r, "loss")  for r in hist],
            "fit":   [_f(r, "fit")   for r in hist],
            "free":  [_f(r, "free")  for r in hist],
            "table": [_f(r, "table") for r in hist],
            "p_mean":[_f(r, "p_mean")for r in hist],
            "p_med": [_f(r, "p_med") for r in hist],
            "sigma2":[_f(r, "sigma2")for r in hist],
        }
    return xs, ys, meta

def plot_single(xs, ys, meta, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    title = (f"{meta['scene']} test{meta['test']} | cluster{meta['cluster_id']} "
             f"sub{meta['sub_id']} | {meta['shape']} (N={meta['n_points']})")

    # main loss components
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(xs, ys["loss"],  label="loss")
    ax.plot(xs, ys["fit"],   label="fit")
    # 'free' and 'table' can be zeros depending on shape; still plot for consistency
    if meta['shape'] == 'superparaboloid':
        ax.plot(xs, ys["drop"],  label="drop")
        ax.plot(xs, ys["extend"], label="extend")
    else:
        ax.plot(xs, ys["free"],  label="free")
        ax.plot(xs, ys["table"], label="table")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # inset / secondary plot for sigma2 & p_mean
    fig2 = plt.figure(figsize=(8, 4))
    ax2 = plt.gca()
    ax2.plot(xs, ys["sigma2"], label="sigma2")
    ax2.plot(xs, ys["p_mean"], label="p_mean")
    ax2.plot(xs, ys["p_med"],  label="p_med")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("value")
    ax2.set_title(title + " — sigma2 / p stats")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # save two files: *_loss.png and *_stats.png
    stem = out_png.with_suffix("")  # strip .png
    f1 = stem.with_name(stem.name + "_loss.png")
    f2 = stem.with_name(stem.name + "_stats.png")
    fig.tight_layout()
    fig.savefig(f1, dpi=150)
    fig2.tight_layout()
    fig2.savefig(f2, dpi=150)
    plt.close(fig)
    plt.close(fig2)
    return [f1, f2]

def plot_overview(cluster_id: int, records, out_png: Path):
    """records: list of (xs, ys, meta) for one cluster"""
    if not records:
        return None
    cols = 2
    rows = math.ceil(len(records)/cols)
    fig = plt.figure(figsize=(10, 4*rows))
    for i, (xs, ys, meta) in enumerate(records, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.plot(xs, ys["loss"], label=f"sub{meta['sub_id']} ({meta['shape']})")
        ax.set_xlabel("iter"); ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(f"cluster{cluster_id} — per-sub loss")
    fig.tight_layout(rect=[0,0,1,0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path, help="Base results dir (results)")
    ap.add_argument("--scene", required=True, help="Scene folder name (e.g., scene_10)")
    ap.add_argument("--test", required=True, type=int, help="Test number (e.g., 3)")
    ap.add_argument("--out", type=Path, default=None, help="Output dir for PNGs (default: alongside YAMLs)")
    ap.add_argument("--clusters", type=int, nargs="*", default=None, help="Only these cluster ids")
    ap.add_argument("--no-overview", action="store_true", help="Disable per-cluster overview plot")
    args = ap.parse_args()

    files = find_loss_files(args.base, args.scene, args.test, set(args.clusters) if args.clusters else None)
    if not files:
        print("No loss files found.")
        return

    # group by cluster for overviews
    by_cluster = {}
    for lid, ypath in files:
        xs, ys, meta = load_history(ypath)
        # per-sub plots
        if args.out is None:
            out_png = ypath.with_suffix(".png")
        else:
            # mirror structure in --out
            rel = ypath.relative_to(args.base / args.scene / f"test{args.test}")
            out_png = (args.out / rel).with_suffix(".png")
        saved = plot_single(xs, ys, meta, out_png)
        print("Saved:", *(str(p) for p in saved))
        by_cluster.setdefault(lid, []).append((xs, ys, meta))

    if not args.no_overview:
        for lid, recs in by_cluster.items():
            # drop into cluster folder near first yaml or into --out
            if args.out is None:
                cluster_dir = files[0][1].parent.parent / f"cluster{lid}"
            else:
                cluster_dir = args.out / f"cluster{lid}"
            out_png = cluster_dir / f"cluster{lid}_overview.png"
            res = plot_overview(lid, recs, out_png)
            if res:
                print("Saved overview:", res)

if __name__ == "__main__":
    main()
