#!/usr/bin/env python3
"""
Plot per-iteration loss histories saved as YAML.

Folder layout expected (same as before):
  <base>/<scene>/test<NUM>/cluster<LID>/sub<SUBID>_loss.yaml

Each YAML has:
  scene, test, cluster_id, sub_id, shape, n_points, timestamp,
  history=[{iter, ...loss terms..., p_mean, p_med, sigma2}, ...]

Unlike the original script, this one does NOT hardcode which keys exist
per shape ("fit"/"free"/"table" vs "fit"/"drop"/"extend"). Instead it
auto-detects every numeric field present in the first row of `history`
(besides "iter") and plots whatever it finds. That means it works
unmodified for:
  - superquadric logs with the simple schema (fit, free, table, ...)
  - superquadric logs with the raw+weighted schema
    (fit_raw, free_raw, table_raw, fit_weighted, free_weighted,
     table_weighted, free_worst_F, ...)
  - supertoroid logs (fit, free, table, free_worst_F, ...)
  - superparaboloid logs (fit, drop, extend, ...)
  - any future field you add later, with no code changes needed.

"p_mean", "p_med", "sigma2" are always split out into a second
("stats") plot since they're on a different scale/meaning than the
loss terms; everything else numeric (besides "iter" and "loss") is
treated as a loss-term curve and plotted together with "loss" itself
on the first plot.

Usage:
  python plot_losses_evolution.py --base /home/.../results_EMS --scene scene_10 --test 3
  # optional:
  #   --out out_dir     : where to write PNGs (default: alongside YAML files)
  #   --clusters 0 2 5  : only plot given cluster ids
  #   --no-overview     : skip per-cluster overview grid
  #   --file path.yaml  : plot a single YAML file directly, ignoring --base/--scene/--test
  #   --log-scale       : use a log y-axis on the loss-terms plot (handy when
  #                        terms differ by orders of magnitude, e.g. free_worst_F vs loss)
"""

import argparse
from pathlib import Path
import math

import yaml
import matplotlib.pyplot as plt

STATS_KEYS = {"p_mean", "p_med", "sigma2"}
SKIP_KEYS = {"iter", "shape"}  # "shape" is a per-row string label (e.g. "superquadric"), not a numeric term


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


def _to_float(v, default=float("nan")):
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _is_numeric_field(row, key):
    """True unless the value is a non-numeric string (e.g. shape='superquadric').
    None is allowed (becomes NaN) since fields like free_worst_F may be logged
    conditionally on some rows."""
    v = row.get(key)
    if v is None:
        return True
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    return False


def load_history(yaml_path: Path):
    data = yaml.safe_load(yaml_path.read_text())
    hist = data.get("history", [])

    meta = {
        "scene": data.get("scene", ""),
        "test": int(data.get("test", -1)),
        "cluster_id": int(data.get("cluster_id", -1)),
        "sub_id": int(data.get("sub_id", -1)),
        "shape": data.get("shape", ""),
        "n_points": int(data.get("n_points", 0)),
        "timestamp": data.get("timestamp", ""),
    }

    if not hist:
        return [], {}, meta

    xs = [int(row.get("iter", i)) for i, row in enumerate(hist)]

    # Union of keys across all rows, in first-seen order, so a field that's
    # only present on some rows (e.g. free_worst_F logged conditionally)
    # still gets its own curve instead of being silently dropped.
    field_order = []
    seen = set()
    for row in hist:
        for k in row.keys():
            if k in SKIP_KEYS or k in seen:
                continue
            if not _is_numeric_field(row, k):
                continue
            seen.add(k)
            field_order.append(k)

    ys = {k: [_to_float(r.get(k)) for r in hist] for k in field_order}
    loss_term_keys = [k for k in field_order if k not in STATS_KEYS]
    stats_keys = [k for k in field_order if k in STATS_KEYS]

    # Split loss terms into "raw" vs "weighted" groups.
    #
    # If the schema explicitly uses the _raw/_weighted suffix convention
    # (e.g. fit_raw/fit_weighted/free_raw/free_weighted/...), honor that
    # directly. "loss" (the total) goes with the weighted group since
    # loss = sum(weighted terms). Diagnostic fields like free_worst_F
    # (not a loss term at all, just an unweighted probe value) go with
    # the raw group.
    #
    # If the schema has no _raw/_weighted suffixes at all (e.g. the
    # simple superquadric/supertoroid schema "fit, free, table, loss",
    # or superparaboloid's "fit, drop, extend, loss") there's nothing
    # to split by suffix. In that case everything goes in the weighted
    # figure (since those values, whatever their name, are exactly what
    # sums into "loss"), and the raw figure only gets non-loss
    # diagnostics such as free_worst_F, if present.
    raw_keys = [k for k in loss_term_keys if k.endswith("_raw")]
    weighted_keys = [k for k in loss_term_keys if k.endswith("_weighted")]
    handled = set(raw_keys) | set(weighted_keys)

    for k in loss_term_keys:
        if k in handled:
            continue
        if k == "loss":
            weighted_keys.append(k)
        elif "worst" in k.lower():
            continue  # diagnostic-only field (e.g. free_worst_F) — not plotted
        else:
            # unsuffixed term (simple schema) -> treat as part of the total
            weighted_keys.append(k)
        handled.add(k)

    return xs, ys, {**meta, "loss_term_keys": loss_term_keys, "stats_keys": stats_keys,
                    "raw_keys": raw_keys, "weighted_keys": weighted_keys}


import itertools

_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["", "o", "s", "^", "D", "x"]
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_terms(ax, xs, ys, keys, log_scale):
    style_cycle = itertools.cycle(_LINESTYLES)
    marker_cycle = itertools.cycle(_MARKERS)
    color_cycle = itertools.cycle(_COLORS)
    n = max(1, len(xs) // 25)  # ~25 markers across the line, not one per point
    for k in keys:
        ax.plot(
            xs, ys[k], label=k,
            linestyle=next(style_cycle),
            marker=next(marker_cycle),
            markevery=n,
            markersize=5,
            linewidth=1.8,
            color=next(color_cycle),
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_single(xs, ys, meta, out_png: Path, log_scale: bool = False, show: bool = False):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    title = (f"{meta['scene']} test{meta['test']} | cluster{meta['cluster_id']} "
             f"sub{meta['sub_id']} | {meta['shape']} (N={meta['n_points']})")

    stem = out_png.with_suffix("")
    saved = []

    # --- raw terms (unweighted; e.g. fit_raw/free_raw/table_raw, free_worst_F) ---
    if meta["raw_keys"]:
        fig = plt.figure(figsize=(9, 5))
        ax = plt.gca()
        _plot_terms(ax, xs, ys, meta["raw_keys"], log_scale)
        ax.set_title(title + " — raw terms")
        f_raw = stem.with_name(stem.name + "_raw.png")
        fig.tight_layout()
        fig.savefig(f_raw, dpi=150)
        if not show:
            plt.close(fig)
        saved.append(f_raw)

    # --- weighted terms (as they actually contribute to "loss") ---
    if meta["weighted_keys"]:
        fig = plt.figure(figsize=(9, 5))
        ax = plt.gca()
        _plot_terms(ax, xs, ys, meta["weighted_keys"], log_scale)
        ax.set_title(title + " — weighted terms (sum = loss)")
        f_weighted = stem.with_name(stem.name + "_weighted.png")
        fig.tight_layout()
        fig.savefig(f_weighted, dpi=150)
        if not show:
            plt.close(fig)
        saved.append(f_weighted)

    # --- stats (p_mean, p_med, sigma2) ---
    fig2 = plt.figure(figsize=(9, 4))
    ax2 = plt.gca()
    for k in meta["stats_keys"]:
        ax2.plot(xs, ys[k], label=k)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("value")
    ax2.set_title(title + " — stats")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    f2 = stem.with_name(stem.name + "_stats.png")
    fig2.tight_layout()
    fig2.savefig(f2, dpi=150)
    if not show:
        plt.close(fig2)
    saved.append(f2)

    return saved



def filter_by_iter(xs, ys, min_iter: int):
    """Drop rows with iter < min_iter, e.g. to hide an initial-transient spike
    (very common right after init, before the optimizer has converged) that
    otherwise dwarfs the y-axis and hides everything afterward."""
    if min_iter <= 0:
        return xs, ys
    keep = [i for i, x in enumerate(xs) if x >= min_iter]
    if not keep:
        return xs, ys  # nothing left -> don't filter, just show everything
    xs_f = [xs[i] for i in keep]
    ys_f = {k: [v[i] for i in keep] for k, v in ys.items()}
    return xs_f, ys_f


def plot_overview(cluster_id: int, records, out_png: Path, show: bool = False):
    """records: list of (xs, ys, meta) for one cluster — one 'loss' curve per sub."""
    if not records:
        return None
    cols = 2
    rows = math.ceil(len(records) / cols)
    fig = plt.figure(figsize=(10, 4 * rows))
    for i, (xs, ys, meta) in enumerate(records, start=1):
        ax = fig.add_subplot(rows, cols, i)
        if "loss" in ys:
            ax.plot(xs, ys["loss"], label=f"sub{meta['sub_id']} ({meta['shape']})")
        ax.set_xlabel("iter")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(f"cluster{cluster_id} — per-sub loss")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    if not show:
        plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, help="Base results dir (results)")
    ap.add_argument("--scene", help="Scene folder name (e.g., scene_10)")
    ap.add_argument("--test", type=int, help="Test number (e.g., 3)")
    ap.add_argument("--file", type=Path, default=None,
                     help="Plot a single YAML file directly (ignores --base/--scene/--test)")
    ap.add_argument("--out", type=Path, default=None, help="Output dir for PNGs (default: alongside YAMLs)")
    ap.add_argument("--clusters", type=int, nargs="*", default=None, help="Only these cluster ids")
    ap.add_argument("--no-overview", action="store_true", help="Disable per-cluster overview plot")
    ap.add_argument("--log-scale", action="store_true", help="Log y-axis on the loss-terms plot")
    ap.add_argument("--skip-first", type=int, default=0,
                     help="Hide iterations before this value (e.g. --skip-first 100). "
                          "Use this when an initial-transient spike right after init "
                          "dwarfs the y-axis and hides the rest of the curve.")
    ap.add_argument("--show", action="store_true",
                     help="Open the figures in a window (in addition to saving PNGs). "
                          "With multiple files, all figures open at once when the run finishes.")
    args = ap.parse_args()

    if args.file is not None:
        xs, ys, meta = load_history(args.file)
        xs, ys = filter_by_iter(xs, ys, args.skip_first)
        out_png = args.file.with_suffix(".png") if args.out is None else args.out / (args.file.stem + ".png")
        saved = plot_single(xs, ys, meta, out_png, log_scale=args.log_scale, show=args.show)
        print("Saved:", *(str(p) for p in saved))
        if args.show:
            plt.show()
        return

    if not (args.base and args.scene and args.test is not None):
        ap.error("either --file, or all of --base/--scene/--test are required")

    files = find_loss_files(args.base, args.scene, args.test,
                             set(args.clusters) if args.clusters else None)
    if not files:
        print("No loss files found.")
        return

    by_cluster = {}
    for lid, ypath in files:
        xs, ys, meta = load_history(ypath)
        if not xs:
            print(f"Skipping empty history: {ypath}")
            continue
        xs, ys = filter_by_iter(xs, ys, args.skip_first)
        if args.out is None:
            out_png = ypath.with_suffix(".png")
        else:
            rel = ypath.relative_to(args.base / args.scene / f"test{args.test}")
            out_png = (args.out / rel).with_suffix(".png")
        saved = plot_single(xs, ys, meta, out_png, log_scale=args.log_scale, show=args.show)
        print("Saved:", *(str(p) for p in saved))
        by_cluster.setdefault(lid, []).append((xs, ys, meta))

    if not args.no_overview:
        for lid, recs in by_cluster.items():
            if args.out is None:
                cluster_dir = files[0][1].parent.parent / f"cluster{lid}"
            else:
                cluster_dir = args.out / f"cluster{lid}"
            out_png = cluster_dir / f"cluster{lid}_overview.png"
            res = plot_overview(lid, recs, out_png, show=args.show)
            if res:
                print("Saved overview:", res)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()