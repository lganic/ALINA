"""Foundation-model "teacher" that labels line-marking pixels without human annotation.

Mask2Former (Swin-L, Mapillary Vistas) has explicit Lane Marking classes. It is run on a crop
around the ROI (full frames at Mapillary scale lose thin, distant taxiway lines), and its
marking pixels are split into yellow / white by CIELAB b*. Labels are then warped into the
same bird's-eye frame ALINA thresholds in.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .threshold import BG, IGNORE, WHITE, YELLOW

M2F_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
YELLOW_MIN_B = 138        # OpenCV 8-bit Lab: neutral grey is b* = 128; painted yellow sits well above
WHITE_MIN_DL = 25         # white paint must be clearly brighter than the surrounding pavement
CROP_LONG_SIDE = 1024
IGNORE_BAND_PX = 5        # teacher boundaries are soft; don't train on the outermost pixels

_MODEL = {}


def _get():
    if not _MODEL:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_ID).eval()
        ids = [int(i) for i, n in model.config.id2label.items() if "lane marking" in n.lower()]
        _MODEL.update(proc=AutoImageProcessor.from_pretrained(M2F_ID), model=model, ids=ids, torch=torch)
    return _MODEL


def roi_crop_box(quad: np.ndarray, shape, pad_frac=0.5, min_side=400):
    H, W = shape[:2]
    x0, y0 = quad.min(0); x1, y1 = quad.max(0)
    pad = pad_frac * max(x1 - x0, y1 - y0, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) / 2 + pad, (y1 - y0) / 2 + pad
    return (int(max(0, cx - half_w)), int(max(0, cy - half_h)), int(min(W, cx + half_w)), int(min(H, cy + half_h)))


def marking_mask(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """uint8 {0,1} mask of lane-marking pixels in full-frame coordinates."""
    from PIL import Image

    m = _get()
    x0, y0, x1, y1 = roi_crop_box(np.asarray(quad, float), img.shape)
    crop = img[y0:y1, x0:x1]
    s = CROP_LONG_SIDE / max(crop.shape[:2])
    crop_r = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    inputs = m["proc"](images=Image.fromarray(cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB)), return_tensors="pt")
    with m["torch"].no_grad():
        out = m["model"](**inputs)
    seg = m["proc"].post_process_semantic_segmentation(out, target_sizes=[crop.shape[:2]])[0].numpy()
    full = np.zeros(img.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = np.isin(seg, m["ids"])
    return full


def frame_labels(img: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, float]:
    """Full-frame label map (BG / YELLOW / WHITE) and teacher latency in ms."""
    t0 = time.perf_counter()
    mark = marking_mask(img, quad) > 0
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, b = lab_img[:, :, 0].astype(np.int32), lab_img[:, :, 2].astype(np.int32)
    # pavement brightness = median L of non-marking pixels inside the ROI
    roi = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi, [np.round(quad).astype(np.int32)], 1)
    pave = L[(roi > 0) & ~mark]
    pave_L = np.median(pave) if len(pave) else np.median(L)
    lab = np.full(img.shape[:2], BG, np.uint8)
    lab[mark] = IGNORE   # marking pixels that are neither (e.g. the black outline around yellow paint)
    lab[mark & (b >= YELLOW_MIN_B)] = YELLOW
    lab[mark & (b < YELLOW_MIN_B) & (L >= pave_L + WHITE_MIN_DL)] = WHITE
    return lab, 1000 * (time.perf_counter() - t0)


def warp_labels(frame_lab: np.ndarray, M: np.ndarray, rect, canvas_shape) -> np.ndarray:
    """Warp a full-frame label map into ALINA's bird's-eye canvas; outside the ROI rect and a
    thin band around every marking boundary become IGNORE."""
    H, W = canvas_shape[:2]
    warped = cv2.warpPerspective(frame_lab, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
    fg = ((warped == YELLOW) | (warped == WHITE)).astype(np.uint8)
    k = np.ones((2 * IGNORE_BAND_PX + 1,) * 2, np.uint8)
    band = cv2.dilate(fg, k) - cv2.erode(fg, k)
    out = warped.copy()
    out[band > 0] = IGNORE
    x0, y0, x1, y1 = rect
    keep = np.zeros_like(out, bool); keep[y0:y1, x0:x1] = True
    out[~keep] = IGNORE
    return out
