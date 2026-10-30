"""Assignment 5: learned / adaptive replacements for ALINA's manual color threshold.

    venv\\Scripts\\python task5_color_threshold.py            # everything (~30-40 min on CPU)
    venv\\Scripts\\python task5_color_threshold.py --no-demo  # skip the demo video

Outputs land in outputs/task5/. Teacher labels and trained models are cached, so reruns are fast.
Only step 3.5 of ALINA changes between methods. ROI, warp, histogram, CIRCLEDAT and unwarp are
identical, so differences in the metrics come from the threshold alone.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
import warnings
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import cv2
import numpy as np
import pandas as pd

from alina import threshold as T
from alina.config import PipelineConfig
from alina.pipeline import dst_rect, process_image
from eval.metrics import calculate_f1, calculate_precision, calculate_recall

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "Raw_Data"
GT_ROOT = ROOT / "data" / "gt_alina_labels"
MANUAL_ROI_PATH = ROOT / "outputs" / "task3a" / "manual_rois.json"
OUT = ROOT / "outputs" / "task5"
VIDEOS = ["vidd_1", "vidd_2", "vidd_3"]
WORK_W, WORK_H = 1920, 1080        # GT coordinates are 1080p
PIXEL_TOL = 10                     # pixel-level P/R/F1 tolerance (same as Task 3a)
BOUNDARY_TOL = 5                   # threshold-stage boundary P/R/F1 tolerance
SEEDS = [0, 1, 2]
BOOTSTRAP = 2000
F1_TIE_MARGIN = 1.0                # paper-F1 points treated as a tie when picking the best method

METHODS = {
    # name: (family, stochastic?)
    "manual": ("original ALINA (fixed HSV box)", False),
    "otsu_lab": ("classical adaptive (per-frame Otsu on b*)", False),
    "gmm_lab": ("unsupervised ML (per-frame GMM)", True),
    "learned_box": ("supervised, learned HSV bounds", True),
    "gbdt": ("supervised ML (gradient-boosted trees)", True),
    "unet": ("deep learning (tiny U-Net)", True),
    "m2f_teacher": ("foundation model (Mask2Former, zero-shot)", False),
}
COLORS = {"manual": "#7a7a7a", "otsu_lab": "#2a78d6", "gmm_lab": "#eb6834", "learned_box": "#1baf7a",
          "gbdt": "#eda100", "unet": "#e87ba4", "m2f_teacher": "#6250d6"}

PERTURBATIONS = ["none", "dark x0.55", "bright x1.5", "gamma 1.8", "cool WB", "warm WB", "noise s=12"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


# data
def split_clips(paths):
    clips = [[paths[0]]]
    for a, b in zip(paths, paths[1:]):
        (clips[-1].append(b) if int(b.stem) - int(a.stem) == 1 else clips.append([b]))
    return clips


FRAMES = {v: sorted((RAW / v).glob("*.jpg")) for v in VIDEOS}
CLIPS = {v: split_clips(FRAMES[v]) for v in VIDEOS}
GT_DIRS = {v: GT_ROOT / "canny_textfiles" / f"canny_textfiles_{i}" for i, v in enumerate(VIDEOS, 1)}
GT_FRAMES = {v: [p for p in FRAMES[v] if (GT_DIRS[v] / f"{p.stem}.txt").exists()] for v in VIDEOS}
MANUAL = {k: np.array(q, float) for k, q in json.loads(MANUAL_ROI_PATH.read_text()).items()}


MIN_ROI_H = 0.05 * WORK_H   # vidd_1/00001 and /00011 are 32 px tall and sit above the line: mis-clicks


def frame_roi(video, path):
    """Oracle ROI: the manual ROI of the closest valid labeled frame in the same clip (preferring
    earlier frames), so that the threshold is the only thing that differs between methods."""
    clip = next(c for c in CLIPS[video] if path in c)
    labeled = [p for p in clip if f"{video}/{p.name}" in MANUAL
               and np.ptp(MANUAL[f"{video}/{p.name}"][:, 1]) >= MIN_ROI_H]
    best = min(labeled, key=lambda p: (abs(int(p.stem) - int(path.stem)), int(p.stem) > int(path.stem)))
    return MANUAL[f"{video}/{best.name}"]


def quad_to_roi_points(quad):
    bl, tl, tr, br = [tuple(int(round(v)) for v in pt) for pt in quad]
    return np.array([[bl, tl, tl, tr, tr, br]], dtype=np.int32)


@lru_cache(maxsize=None)
def load(path):
    return cv2.resize(cv2.imread(str(path)), (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)


def load_coords(path):
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.loadtxt(path, dtype=np.int32, ndmin=2).reshape(-1, 2)


CFG = PipelineConfig(input_dir=RAW, output_images_dir=OUT / "_tmp", output_coords_dir=OUT / "_tmp", fast_traversal=True)
RECT = dst_rect(CFG)


def warp(img, quad):
    """The same perspective warp process_image performs (step 3.3)."""
    r = CFG.roi
    dst = np.float32([r.dst_bottom_left, r.dst_top_left, r.dst_top_right, r.dst_bottom_right])
    M = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0])), M


def perturb(img, kind, seed=0):
    f = img.astype(np.float32)
    if kind == "dark x0.55":
        f *= 0.55
    elif kind == "bright x1.5":
        f *= 1.5
    elif kind == "gamma 1.8":
        f = 255 * (f / 255) ** 1.8
    elif kind == "cool WB":
        f *= np.array([1.2, 1.0, 0.85])     # BGR
    elif kind == "warm WB":
        f *= np.array([0.8, 1.0, 1.15])
    elif kind == "noise s=12":
        f += np.random.RandomState(seed).normal(0, 12, f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


# metrics
def paper_prf(gt, pred):
    """Exactly eval/evaluate.py: x-set and y-set overlap, averaged."""
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    r = (calculate_recall(gt[:, 0], pred[:, 0]) + calculate_recall(gt[:, 1], pred[:, 1])) / 2
    p = (calculate_precision(gt[:, 0], pred[:, 0]) + calculate_precision(gt[:, 1], pred[:, 1])) / 2
    return p, r, calculate_f1(p, r)


def pixel_prf(gt, pred, tol, shape=(WORK_H, WORK_W)):
    """A predicted pixel is correct if a GT pixel lies within tol px, and vice versa (Task 3a metric)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0

    def dist_to(points):
        canvas = np.ones(shape[:2], np.uint8)
        canvas[np.clip(points[:, 1], 0, shape[0] - 1), np.clip(points[:, 0], 0, shape[1] - 1)] = 0
        return cv2.distanceTransform(canvas, cv2.DIST_L2, 3)

    d_gt, d_pred = dist_to(gt), dist_to(pred)
    p = 100 * float((d_gt[np.clip(pred[:, 1], 0, shape[0] - 1), np.clip(pred[:, 0], 0, shape[1] - 1)] <= tol).mean())
    r = 100 * float((d_pred[np.clip(gt[:, 1], 0, shape[0] - 1), np.clip(gt[:, 0], 0, shape[1] - 1)] <= tol).mean())
    return p, r, calculate_f1(p, r)


def mask_boundary_points(mask, M):
    """Boundary of the threshold mask (as ALINA sees it, inside the ROI rect) in frame coordinates."""
    m = np.zeros_like(mask)
    x0, y0, x1, y1 = RECT
    m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    un = cv2.warpPerspective(m, np.linalg.inv(M), (WORK_W, WORK_H), flags=cv2.INTER_NEAREST)
    edge = cv2.morphologyEx((un > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(edge)
    return np.column_stack((xs, ys))


def teacher_iou(mask, labels, cls=T.YELLOW):
    valid = labels != T.IGNORE
    pred, ref = (mask > 0) & valid, (labels == cls) & valid
    union = (pred | ref).sum()
    return float((pred & ref).sum() / union) if union else np.nan


# teacher labels
TEACHER_DIR = OUT / "teacher"


def teacher_cache(video, path):
    return TEACHER_DIR / video / f"{path.stem}.png"


def build_teacher_labels():
    from alina import teacher

    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    times_path = TEACHER_DIR / "latency.csv"
    times = pd.read_csv(times_path).to_dict("records") if times_path.exists() else []
    done = {(t["video"], t["frame"]) for t in times}
    for v in VIDEOS:
        (TEACHER_DIR / v).mkdir(exist_ok=True)
        for i, p in enumerate(FRAMES[v]):
            if teacher_cache(v, p).exists() and (v, int(p.stem)) in done:
                continue
            quad = frame_roi(v, p)
            img = load(p)
            lab, ms = teacher.frame_labels(img, quad)
            if i == 0 and not times:      # first call includes model loading; time it again
                lab, ms = teacher.frame_labels(img, quad)
            cv2.imwrite(str(teacher_cache(v, p)), lab)
            times.append(dict(video=v, frame=int(p.stem), teacher_ms=ms))
            pd.DataFrame(times).to_csv(times_path, index=False)
        print(f"teacher labels: {v} done")
    return pd.read_csv(times_path)


@lru_cache(maxsize=None)
def warped_labels(video, path):
    from alina.teacher import warp_labels

    frame_lab = cv2.imread(str(teacher_cache(video, path)), cv2.IMREAD_GRAYSCALE)
    _, M = warp(load(path), frame_roi(video, path))
    return warp_labels(frame_lab, M, RECT, (WORK_H, WORK_W))


class TeacherThresholder(T.Thresholder):
    """Uses Mask2Former's yellow labels directly as the mask. `live=True` reruns the model
    (needed for perturbed frames); otherwise the cached labels are read."""
    name = "m2f_teacher"

    def __init__(self):
        self.frame = self.quad = self.key = None
        self.live = False

    def __call__(self, warped_img, rect):
        from alina import teacher

        if self.live:
            lab, self.last_ms = teacher.frame_labels(self.frame, self.quad)
        else:
            lab = cv2.imread(str(teacher_cache(*self.key)), cv2.IMREAD_GRAYSCALE)
        _, M = warp(self.frame, self.quad)
        w = cv2.warpPerspective(lab, M, (warped_img.shape[1], warped_img.shape[0]), flags=cv2.INTER_NEAREST)
        return ((w == T.YELLOW) * 255).astype(np.uint8)


# training (leave-one-video-out)
MODEL_DIR = OUT / "models"

def train_fold(method, held_out, seed):
    """Train on teacher labels of the two other videos. No CBEM ground truth is ever used for training."""
    ext = {"learned_box": ".json", "unet": ".pt"}.get(method, ".joblib")
    path = MODEL_DIR / f"{method}_no-{held_out}_seed{seed}{ext}"
    if path.exists():
        return T.load_thresholder(path), np.nan
    set_seed(seed)
    train = [(v, p) for v in VIDEOS if v != held_out for p in FRAMES[v]]
    imgs = [warp(load(p), frame_roi(v, p))[0] for v, p in train]
    labs = [warped_labels(v, p) for v, p in train]
    t0 = time.perf_counter()
    model = T.build(method)
    if method == "unet":
        model.fit(imgs, labs, seed=seed, rects=[RECT] * len(imgs))
    else:
        model.fit(imgs, labs, seed=seed)
    train_s = time.perf_counter() - t0
    T.save_thresholder(model, path)
    print(f"  trained {method} (held out {held_out}, seed {seed}) in {train_s:.0f}s")
    return model, train_s


def get_thresholders(seeds):
    """{(method, video, seed): thresholder}; deterministic methods use seed None."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out, train_rows = {}, []
    teacher_t = TeacherThresholder()
    for m, (_, stochastic) in METHODS.items():
        for v in VIDEOS:
            for s in (seeds if stochastic else [None]):
                if m == "manual":
                    t = T.ManualHSV(CFG.yellow_lower, CFG.yellow_upper)
                elif m == "otsu_lab":
                    t = T.OtsuLab()
                elif m == "gmm_lab":
                    t = T.GMMLab(seed=s)
                elif m == "m2f_teacher":
                    t = teacher_t
                else:
                    t, secs = train_fold(m, v, s)
                    train_rows.append(dict(method=m, held_out=v, seed=s, train_s=secs))
                out[(m, v, s)] = t
    return out, pd.DataFrame(train_rows)


# experiment 1: end-to-end
def run_one(t, img, quad, video, path, live_teacher=False):
    if isinstance(t, TeacherThresholder):
        t.frame, t.quad, t.key, t.live = img, quad, (video, path), live_teacher
    timing = {}

    def timed_t(warped_img, rect):
        t0 = time.perf_counter()
        m = t(warped_img, rect)
        timing["thr_ms"] = 1000 * (time.perf_counter() - t0)
        return m

    dbg = {}
    t0 = time.perf_counter()
    annotated, coords, had = process_image(img, quad_to_roi_points(quad), CFG, thresholder=timed_t, debug=dbg)
    alina_ms = 1000 * (time.perf_counter() - t0)
    return annotated, coords, had, dbg, timing.get("thr_ms", np.nan), alina_ms


def score_gt(video, path, coords, dbg):
    gt = load_coords(GT_DIRS[video] / f"{path.stem}.txt")
    pp, pr, pf = paper_prf(gt, coords)
    xp, xr, xf = pixel_prf(gt, coords, PIXEL_TOL)
    if "mask" in dbg:
        bpts = mask_boundary_points(dbg["mask"], dbg["M"])
        bp, br, bf = pixel_prf(gt, bpts, BOUNDARY_TOL)
    else:
        bp = br = bf = 0.0
    return dict(paper_P=pp, paper_R=pr, paper_F1=pf, pix_P=xp, pix_R=xr, pix_F1=xf, bnd_P=bp, bnd_R=br, bnd_F1=bf)


def experiment_1(thr, teacher_ms):
    rows = []
    for (m, v, s), t in thr.items():
        for p in FRAMES[v]:
            img, quad = load(p), frame_roi(v, p)
            _, coords, had, dbg, thr_ms, alina_ms = run_one(t, img, quad, v, p)
            if m == "m2f_teacher":
                thr_ms = float(teacher_ms.loc[(teacher_ms.video == v) & (teacher_ms.frame == int(p.stem)), "teacher_ms"].iloc[0])
            row = dict(method=m, seed=s, video=v, frame=p.stem, had_lines=had, n_px=len(coords),
                       thr_ms=thr_ms, alina_ms=alina_ms)
            if m != "m2f_teacher" and "mask" in dbg:
                row["teacher_iou"] = teacher_iou(dbg["mask"], warped_labels(v, p))
            if p in GT_FRAMES[v]:
                row.update(score_gt(v, p, coords, dbg))
            rows.append(row)
        print(f"  exp1: {m} {v} seed={s}")
    return pd.DataFrame(rows)


METRICS = ["paper_P", "paper_R", "paper_F1", "pix_F1", "bnd_P", "bnd_R", "bnd_F1"]


def summarize_1(df):
    """Mean over frames per seed, then mean +- std across seeds."""
    per_seed = df.groupby(["method", "seed"], dropna=False, sort=False).agg(
        detection_rate=("had_lines", lambda s: 100 * s.mean()),
        teacher_iou=("teacher_iou", "mean"),
        thr_ms=("thr_ms", "median"), alina_ms=("alina_ms", "median"),
        **{c: (c, "mean") for c in METRICS}).reset_index()
    cols = ["detection_rate"] + METRICS + ["teacher_iou", "thr_ms", "alina_ms"]
    mean = per_seed.groupby("method", sort=False)[cols].mean()
    std = per_seed.groupby("method", sort=False)[cols].std().fillna(0.0)
    mean.insert(0, "family", [METHODS[m][0] for m in mean.index])
    mean.insert(1, "n_seeds", per_seed.groupby("method", sort=False).size())
    return mean, std


def bootstrap(df, metric="paper_F1", baseline="manual", n=BOOTSTRAP, seed=0):
    """Frame-level bootstrap over the GT frames (seed-averaged): 95% CI of each method's mean and
    of its paired difference to the baseline."""
    gt = df.dropna(subset=[metric]).groupby(["method", "video", "frame"], sort=False)[metric].mean().unstack("method")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(gt), size=(n, len(gt)))
    rows = []
    for m in gt.columns:
        a = gt[m].to_numpy(); d = a - gt[baseline].to_numpy()
        ma, md = a[idx].mean(1), d[idx].mean(1)
        rows.append(dict(method=m, metric=metric, mean=a.mean(), ci_lo=np.percentile(ma, 2.5), ci_hi=np.percentile(ma, 97.5),
                         delta_vs_manual=d.mean(), delta_ci_lo=np.percentile(md, 2.5), delta_ci_hi=np.percentile(md, 97.5),
                         p_better=float((md > 0).mean())))
    return pd.DataFrame(rows).set_index("method")


# experiment 2: robustness
def experiment_2(thr):
    rows = []
    gt_items = [(v, p) for v in VIDEOS for p in GT_FRAMES[v]]
    for kind in PERTURBATIONS:
        for (m, v, s), t in thr.items():
            for vv, p in gt_items:
                if vv != v:
                    continue
                img = perturb(load(p), kind, seed=int(p.stem))
                live = m == "m2f_teacher" and kind != "none"
                _, coords, had, dbg, _, _ = run_one(t, img, frame_roi(v, p), v, p, live_teacher=live)
                rows.append(dict(perturbation=kind, method=m, seed=s, video=v, frame=p.stem, had_lines=had,
                                 **score_gt(v, p, coords, dbg)))
        print(f"  exp2: {kind}")
    return pd.DataFrame(rows)


# figures
def exp2_delta_figure(rob, metric="paper_F1"):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3dd", "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.edgecolor": "#9c9b93", "font.size": 10})

    boot = pd.concat([bootstrap(rob[rob.perturbation == k], metric).assign(perturbation=k) for k in PERTURBATIONS])
    boot.round(2).to_csv(OUT / "exp2_bootstrap.csv")
    kinds = PERTURBATIONS[1:]
    methods = [m for m in boot.index.unique() if m != "manual"]
    lo, hi = boot["delta_ci_lo"].min(), boot["delta_ci_hi"].max()
    pad = 0.05 * (hi - lo)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, kinds):
        b = boot[boot.perturbation == k].loc[methods]
        y = np.arange(len(b))
        ax.axvline(0, color="#3b3b38", lw=1)
        ax.errorbar(b["delta_vs_manual"], y, xerr=[b["delta_vs_manual"] - b["delta_ci_lo"], b["delta_ci_hi"] - b["delta_vs_manual"]],
                    fmt="none", ecolor="#3b3b38", lw=1.2, capsize=4)
        ax.scatter(b["delta_vs_manual"], y, s=70, c=[COLORS[m] for m in b.index], zorder=3, edgecolors="white", linewidths=2)
        manual_f1 = boot.loc[(boot.index == "manual") & (boot.perturbation == k), "mean"].iloc[0]
        ax.set_title(f"{k}   (manual F1 = {manual_f1:.1f})", loc="left", fontsize=10)
        ax.set_yticks(y, b.index); ax.set_xlim(lo - pad, hi + pad)
        ax.grid(axis="y", visible=False)
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("Δ paper F1 vs manual (points)")
    fig.suptitle("Experiment 2: is each method better than the hand-set threshold under each lighting change?  "
                 "(95% paired bootstrap CI over GT frames)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "exp2_delta_ci.png", dpi=150); plt.close(fig)
    return boot


def figures(summary, std, boot, rob, frames_df, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3dd", "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.edgecolor": "#9c9b93", "font.size": 10})
    methods = list(summary.index)

    # 1. headline metrics, small multiples (one panel per metric -> one axis each)
    panels = [("paper_F1", "Paper F1 (ALINA metric)"), ("pix_F1", f"Pixel F1 @ {PIXEL_TOL}px"),
              ("bnd_F1", f"Threshold-mask boundary F1 @ {BOUNDARY_TOL}px"), ("detection_rate", "Detection rate (150 frames)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        y = np.arange(len(methods))
        ax.barh(y, summary[col], xerr=std[col], color=[COLORS[m] for m in methods], height=0.6,
                error_kw=dict(ecolor="#3b3b38", lw=1, capsize=3), edgecolor="white", linewidth=2)
        for yi, val in zip(y, summary[col]):
            ax.text(val + 2, yi, f"{val:.1f}", va="center", fontsize=9, color="#3b3b38")
        ax.set_yticks(y, methods); ax.set_xlim(0, 110); ax.set_title(title, fontsize=10, loc="left")
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    fig.suptitle("Experiment 1: ALINA end-to-end with each thresholder (error bars = std over 3 seeds)", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "exp1_metrics.png", dpi=150); plt.close(fig)

    # 2. bootstrap CI of paired difference vs manual
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b = boot.drop(index="manual")
    y = np.arange(len(b))
    ax.errorbar(b["delta_vs_manual"], y, xerr=[b["delta_vs_manual"] - b["delta_ci_lo"], b["delta_ci_hi"] - b["delta_vs_manual"]],
                fmt="none", ecolor="#3b3b38", lw=1.2, capsize=4)
    ax.scatter(b["delta_vs_manual"], y, s=70, c=[COLORS[m] for m in b.index], zorder=3, edgecolors="white", linewidths=2)
    ax.axvline(0, color="#3b3b38", lw=1)
    ax.set_yticks(y, b.index); ax.invert_yaxis()
    ax.set_xlabel("Δ paper F1 vs manual threshold (points), 95% bootstrap CI over GT frames")
    ax.set_title("Is each method better than the hand-set threshold?", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "exp1_delta_ci.png", dpi=150); plt.close(fig)

    exp2_delta_figure(rob)

    # 3. robustness: small multiples, one per method, showing F1 per perturbation
    r = rob.groupby(["perturbation", "method"], sort=False)["paper_F1"].mean().unstack("method").reindex(PERTURBATIONS)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(PERTURBATIONS)); w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ax.bar(x + (i - len(methods) / 2 + 0.5) * w, r[m], width=w * 0.9, color=COLORS[m], label=m, edgecolor="white", linewidth=1)
    ax.set_xticks(x, PERTURBATIONS); ax.set_ylabel("Paper F1 (GT frames)"); ax.set_ylim(0, 105)
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "exp2_robustness.png", dpi=150); plt.close(fig)

    # 4. speed vs accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        ax.scatter(summary.loc[m, "thr_ms"], summary.loc[m, "paper_F1"], s=80, color=COLORS[m], edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(m, (summary.loc[m, "thr_ms"], summary.loc[m, "paper_F1"]), xytext=(6, 4), textcoords="offset points", fontsize=9, color="#3b3b38")
    ax.set_xscale("log"); ax.set_xlabel("threshold step latency, median ms/frame (CPU, log)"); ax.set_ylabel("Paper F1")
    ax.set_title("Speed vs accuracy of the threshold step", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "speed_vs_f1.png", dpi=150); plt.close(fig)

    # 5. mask gallery: bird's-eye ROI + each method's mask on a few GT frames (seed 0)
    picks = [("vidd_1", "00002"), ("vidd_1", "01063"), ("vidd_2", "00005")]
    fig, axes = plt.subplots(len(picks), len(methods) + 2, figsize=(2.1 * (len(methods) + 2), 1.6 * len(picks)))
    x0, y0, x1, y1 = RECT
    for r_i, (v, stem) in enumerate(picks):
        p = RAW / v / f"{stem}.jpg"
        img, quad = load(p), frame_roi(v, p)
        w, _ = warp(img, quad)
        axes[r_i, 0].imshow(cv2.cvtColor(w[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        lab = warped_labels(v, p)[y0:y1, x0:x1]
        vis = np.zeros(lab.shape + (3,), np.uint8); vis[lab == T.YELLOW] = (237, 161, 0); vis[lab == T.WHITE] = (255, 255, 255); vis[lab == T.IGNORE] = (70, 70, 70)
        vis[:, : max(0, CFG.mask_ignore_left_columns - x0)] //= 3
        axes[r_i, 1].imshow(vis)
        for c_i, m in enumerate(methods, 2):
            t = thr[(m, v, 0 if METHODS[m][1] else None)]
            _, _, _, dbg, _, _ = run_one(t, img, quad, v, p)
            mk = dbg.get("mask")
            axes[r_i, c_i].imshow(mk[y0:y1, x0:x1] if mk is not None else np.zeros((y1 - y0, x1 - x0)), cmap="gray", vmin=0, vmax=255)
        for c_i, title in enumerate(["bird's-eye ROI", "teacher (Y/W)"] + methods):
            axes[r_i, c_i].set_xticks([]); axes[r_i, c_i].set_yticks([])
            if r_i == 0:
                axes[r_i, c_i].set_title(title, fontsize=9)
        axes[r_i, 0].set_ylabel(f"{v}/{stem}", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "mask_gallery.png", dpi=150); plt.close(fig)

    # 6. learned HSV bounds vs manual
    boxes = [(k, t) for k, t in thr.items() if k[0] == "learned_box"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
    for d, ch in enumerate(["H (normalized)", "S (normalized)", "V (normalized)"]):
        ax = axes[d]
        ax.plot([CFG.yellow_lower[d], CFG.yellow_upper[d]], [0, 0], color=COLORS["manual"], lw=6, solid_capstyle="butt")
        for i, ((_, v, s), t) in enumerate(boxes, 1):
            ax.plot([t.lower[d], t.upper[d]], [i, i], color=COLORS["learned_box"], lw=6, solid_capstyle="butt")
        ax.set_xlim(0, 255); ax.set_title(ch, fontsize=10, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(boxes) + 1), ["manual"] + [f"no-{v[-1]} s{s}" for (_, v, s), _ in boxes], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("inRange bounds: manual (gray) vs learned per fold/seed (green)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "learned_box_bounds.png", dpi=150); plt.close(fig)


# demo video
def demo_video(best, thr, fps=4):
    path = OUT / f"demo_{best}_vs_manual.mp4"
    W, H = 1280, 720
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    x0, y0, x1, y1 = RECT
    for v in VIDEOS:
        for p in FRAMES[v]:
            img, quad = load(p), frame_roi(v, p)
            tiles = []
            for m in ("manual", best):
                t = thr[(m, v, 0 if METHODS[m][1] else None)]
                ann, coords, had, dbg, thr_ms, _ = run_one(t, img, quad, v, p)
                ann = ann.copy()
                cv2.polylines(ann, [np.round(quad).astype(np.int32)], True, (0, 255, 0), 2)
                top = cv2.resize(ann, (640, 360), interpolation=cv2.INTER_AREA)
                label = f"{m}: {'line found' if had else 'no line'}  ({len(coords)} px)"
                cv2.rectangle(top, (0, 0), (640, 30), (0, 0, 0), -1)
                cv2.putText(top, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                mk = dbg.get("mask", np.zeros(img.shape[:2], np.uint8))[y0:y1, x0:x1]
                bev = cv2.resize(dbg.get("warped", img)[y0:y1, x0:x1], (640, 360))
                ov = bev.copy(); ov[cv2.resize(mk, (640, 360), interpolation=cv2.INTER_NEAREST) > 0] = (255, 0, 255)
                bot = cv2.addWeighted(bev, 0.4, ov, 0.6, 0)
                cv2.putText(bot, f"{m} threshold mask (bird's-eye ROI)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                tiles.append(np.vstack([top, bot]))
            frame = np.hstack(tiles)
            cv2.putText(frame, f"{v}/{p.name}", (W - 210, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(frame)
    vw.release()
    return path


# tables
def md_table(df, std=None, floatfmt=1):
    cols = list(df.columns)
    lines = ["| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                s = "" if np.isnan(v) else f"{v:.{floatfmt}f}"
                if std is not None and c in std.columns and std.loc[m, c] > 0:
                    s += f" ± {std.loc[m, c]:.{floatfmt}f}"
                cells.append(s)
            else:
                cells.append(str(v))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--no-demo", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse saved exp1/2/3 CSVs instead of recomputing")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    set_seed(0)

    print(f"GT frames with raw images: {sum(len(g) for g in GT_FRAMES.values())} "
          f"({', '.join(f'{v}: {len(GT_FRAMES[v])}' for v in VIDEOS)})")

    print("[1/6] teacher labels (Mask2Former)")
    teacher_ms = build_teacher_labels()

    print("[2/6] training (leave-one-video-out, teacher labels only)")
    thr, train_df = get_thresholders(args.seeds)
    train_df.to_csv(OUT / "training_time.csv", index=False)
    (OUT / "learned_boxes.json").write_text(json.dumps(
        {f"{v}/seed{s}": t.state() for (m, v, s), t in thr.items() if m == "learned_box"}, indent=1))

    print("[3/6] experiment 1: end-to-end")
    reuse = lambda name: args.reuse and (OUT / name).exists()
    read = lambda name: pd.read_csv(OUT / name, dtype={"frame": str})
    if reuse("exp1_frames.csv"):
        frames_df = read("exp1_frames.csv")
    else:
        frames_df = experiment_1(thr, teacher_ms)
        frames_df.to_csv(OUT / "exp1_frames.csv", index=False)
    summary, std = summarize_1(frames_df)
    summary.round(2).to_csv(OUT / "exp1_summary.csv"); std.round(2).to_csv(OUT / "exp1_summary_std.csv")
    boot = pd.concat([bootstrap(frames_df, "paper_F1"), bootstrap(frames_df, "bnd_F1")])
    boot.round(2).to_csv(OUT / "exp1_bootstrap.csv")
    per_video = frames_df.pivot_table(index="method", columns="video", values=["paper_F1", "bnd_F1", "had_lines"], aggfunc="mean", sort=False)
    per_video.round(3).to_csv(OUT / "exp1_per_video.csv")

    print("[4/6] experiment 2: photometric robustness")
    if reuse("exp2_frames.csv"):
        rob = read("exp2_frames.csv")
    else:
        rob = experiment_2(thr)
        rob.to_csv(OUT / "exp2_frames.csv", index=False)
    rob_tab = rob.groupby(["method", "perturbation"], sort=False)["paper_F1"].mean().unstack("perturbation")[PERTURBATIONS]
    rob_tab["mean_drop"] = rob_tab["none"] - rob_tab[PERTURBATIONS[1:]].mean(1)
    rob_tab["worst"] = rob_tab[PERTURBATIONS[1:]].min(1)
    rob_tab.round(2).to_csv(OUT / "exp2_summary.csv")

    print("[5/6] best method, figures, tables")
    # Selection: paper F1 within a practical-equivalence margin of the top counts as a tie (with 17
    # near-static GT frames, sub-point differences are noise); ties go to boundary F1, then detection.
    cand = summary.drop(index=["manual"])
    tied = cand[cand["paper_F1"] >= cand["paper_F1"].max() - F1_TIE_MARGIN]
    best = tied.sort_values(["bnd_F1", "detection_rate", "thr_ms"], ascending=[False, False, True]).index[0]
    figures(summary, std, boot.query("metric == 'paper_F1'"), rob, frames_df, thr)

    final = None
    if METHODS[best][1] and best not in ("gmm_lab",):
        # deployable model: same recipe, trained on all three videos' teacher labels, seed 0
        set_seed(0)
        items = [(v, p) for v in VIDEOS for p in FRAMES[v]]
        model = T.build(best)
        imgs = [warp(load(p), frame_roi(v, p))[0] for v, p in items]
        labs = [warped_labels(v, p) for v, p in items]
        model.fit(imgs, labs, seed=0, **({"rects": [RECT] * len(imgs)} if best == "unet" else {}))
        final = MODEL_DIR / f"final_{best}"
        T.save_thresholder(model, final)
    (OUT / "best_method.json").write_text(json.dumps({
        "method": best, "selection": f"paper F1 within {F1_TIE_MARGIN} pt of the top = tie; ties broken by boundary F1, then detection rate; manual excluded",
        "tied_on_paper_F1": list(tied.index),
        "summary": summary.loc[best].drop("family").astype(float).round(3).to_dict(),
        "final_model": str(next(MODEL_DIR.glob(f"final_{best}.*")).relative_to(ROOT)) if final else None,
    }, indent=1))

    show = ["detection_rate", "paper_P", "paper_R", "paper_F1", "pix_F1", "bnd_F1", "teacher_iou", "thr_ms"]
    b = boot.query("metric == 'paper_F1'")
    ci = pd.DataFrame({"paper_F1 95% CI": [f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(b.ci_lo, b.ci_hi)],
                       "Δ vs manual [95% CI]": [f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]" for d, lo, hi in zip(b.delta_vs_manual, b.delta_ci_lo, b.delta_ci_hi)],
                       "P(better)": b.p_better.round(2).astype(str)}, index=b.index)
    tables = "\n\n".join([
        f"## Experiment 1: end-to-end (mean ± std over seeds {args.seeds}; GT metrics on {len(frames_df.dropna(subset=['paper_F1']).groupby(['video','frame']))} CBEM frames)",
        md_table(summary[show], std[show]),
        "### Bootstrap (paper F1, frame-level, seed-averaged)", md_table(ci),
        "## Experiment 2: paper F1 under photometric perturbation (GT frames)", md_table(rob_tab),
        "## Training time (s, per fold)", md_table(train_df.groupby("method")[["train_s"]].mean()),
        f"**Best method:** `{best}`",
    ])
    (OUT / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)

    if not args.no_demo:
        print("[6/6] demo video")
        print("wrote", demo_video(best, thr).relative_to(ROOT))


if __name__ == "__main__":
    main()
