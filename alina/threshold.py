"""Learned / adaptive replacements for ALINA's fixed HSV color threshold (step 3.5).

Every thresholder maps the bird's-eye warped BGR image to a uint8 mask (255 = yellow line
marking), the same contract as the original ``cv2.inRange`` call, so the rest of ALINA
(histogram, CIRCLEDAT, unwarp) is unchanged.

Supervised thresholders are trained on per-pixel label maps in the warped frame:
0 = background, 1 = yellow marking, 2 = white marking, 255 = ignore.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .color import normalize_color_features

BG, YELLOW, WHITE, IGNORE = 0, 1, 2, 255


def rect_slice(rect):
    x0, y0, x1, y1 = rect
    return slice(y0, y1), slice(x0, x1)


def to_mask(binary: np.ndarray, shape) -> np.ndarray:
    m = binary.astype(np.uint8) * 255
    if m.shape[:2] != tuple(shape[:2]):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


class Thresholder:
    name = "base"
    needs_training = False

    def fit(self, images: list[np.ndarray], labels: list[np.ndarray], seed: int = 0) -> "Thresholder":
        return self

    def __call__(self, warped: np.ndarray, rect) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- baseline
class ManualHSV(Thresholder):
    """The original ALINA step: fixed bounds on per-frame min-max normalized HSV."""
    name = "manual"

    def __init__(self, lower=(0, 70, 170), upper=(255, 255, 255)):
        self.lower, self.upper = np.array(lower), np.array(upper)

    def __call__(self, warped, rect):
        return cv2.inRange(normalize_color_features(warped), self.lower, self.upper)


# --------------------------------------------------------------------------- unsupervised, per frame
class OtsuLab(Thresholder):
    """Per-frame Otsu threshold on the CIELAB b* (blue-yellow) channel, fitted inside the ROI."""
    name = "otsu_lab"

    def __call__(self, warped, rect):
        b = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[:, :, 2]
        t, _ = cv2.threshold(np.ascontiguousarray(b[rect_slice(rect)]), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.last_threshold = t
        return to_mask(b > t, warped.shape)


class GMMLab(Thresholder):
    """Per-frame Gaussian mixture on (L*, a*, b*) inside the ROI; the most yellow component is the line."""
    name = "gmm_lab"

    def __init__(self, k=3, n_samples=20000, seed=0, min_contrast=4.0):
        self.k, self.n_samples, self.seed, self.min_contrast = k, n_samples, seed, min_contrast

    def fit(self, images, labels, seed=0):
        self.seed = seed
        return self

    def __call__(self, warped, rect):
        from sklearn.mixture import GaussianMixture

        small = cv2.resize(warped, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        r = [v // 2 for v in rect]
        roi = lab[rect_slice(r)].reshape(-1, 3)
        rng = np.random.RandomState(self.seed)
        sample = roi[rng.choice(len(roi), size=min(self.n_samples, len(roi)), replace=False)]
        gmm = GaussianMixture(self.k, covariance_type="full", random_state=self.seed).fit(sample)
        yellow = int(np.argmax(gmm.means_[:, 2]))
        # a "yellow" component that is no yellower than the pavement means there is no line
        if gmm.means_[yellow, 2] - np.median(sample[:, 2]) < self.min_contrast:
            return np.zeros(warped.shape[:2], np.uint8)
        post = gmm.predict_proba(lab.reshape(-1, 3))[:, yellow].reshape(lab.shape[:2])
        return to_mask(post > 0.5, warped.shape)


# --------------------------------------------------------------------------- supervised
def _sample_pixels(images, labels, per_frame, rng, feature_fn):
    """Class-balanced pixel sample; w re-weights each pixel back to its true frequency in the frame."""
    X, y, w = [], [], []
    for img, lab in zip(images, labels):
        F = feature_fn(img)
        if lab.shape != F.shape[:2]:
            lab = cv2.resize(lab, (F.shape[1], F.shape[0]), interpolation=cv2.INTER_NEAREST)
        F, lab = F.reshape(-1, F.shape[-1]), lab.reshape(-1)
        for cls in (BG, YELLOW, WHITE):
            idx = np.flatnonzero(lab == cls)
            if len(idx) == 0:
                continue
            take = rng.choice(idx, size=min(per_frame // 3, len(idx)), replace=False)
            X.append(F[take]); y.append(lab[take]); w.append(np.full(len(take), len(idx) / len(take)))
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


class LearnedHSVBox(Thresholder):
    """Learns the six inRange bounds themselves (drop-in for the manual values).

    Coordinate ascent on yellow-vs-rest F1 over a 3-D histogram of ALINA-normalized HSV,
    with seeded random restarts. The result can be pasted into PipelineConfig.yellow_lower/upper.
    """
    name = "learned_box"
    needs_training = True

    def __init__(self, bins=64, restarts=8, per_frame=30000):
        self.bins, self.restarts, self.per_frame = bins, restarts, per_frame
        self.lower, self.upper = np.array([0, 70, 170]), np.array([255, 255, 255])

    @staticmethod
    def _features(img):
        return normalize_color_features(img)

    def fit(self, images, labels, seed=0):
        rng = np.random.RandomState(seed)
        X, y, w = _sample_pixels(images, labels, self.per_frame, rng, self._features)
        nb = self.bins
        q = (X.astype(np.int32) * nb) // 256
        pos = np.zeros((nb,) * 3); neg = np.zeros((nb,) * 3)
        # weighted so the F1 being optimized reflects the true yellow/background ratio
        np.add.at(pos, tuple(q[y == YELLOW].T), w[y == YELLOW])
        np.add.at(neg, tuple(q[y != YELLOW].T), w[y != YELLOW])
        P = np.pad(pos, ((1, 0),) * 3).cumsum(0).cumsum(1).cumsum(2)
        N = np.pad(neg, ((1, 0),) * 3).cumsum(0).cumsum(1).cumsum(2)
        total_pos = pos.sum()

        def box_sum(C, lo, hi):
            a0, a1, a2 = lo; b0, b1, b2 = hi + 1
            return (C[b0, b1, b2] - C[a0, b1, b2] - C[b0, a1, b2] - C[b0, b1, a2]
                    + C[a0, a1, b2] + C[a0, b1, a2] + C[b0, a1, a2] - C[a0, a1, a2])

        def f1(lo, hi):
            if np.any(lo > hi):
                return 0.0
            tp = box_sum(P, lo, hi); fp = box_sum(N, lo, hi)
            return 2 * tp / (tp + fp + total_pos) if tp else 0.0

        best, best_score = None, -1.0
        for r in range(self.restarts):
            if r == 0:
                lo, hi = (np.array([0, 70, 170]) * nb) // 256, np.full(3, nb - 1)   # start from the manual box
            else:
                lo = rng.randint(0, nb // 2, 3); hi = rng.randint(nb // 2, nb, 3)
            score = f1(lo, hi)
            improved = True
            while improved:
                improved = False
                for d in range(3):
                    for which in (0, 1):
                        for v in range(nb):
                            l2, h2 = lo.copy(), hi.copy()
                            (l2 if which == 0 else h2)[d] = v
                            s = f1(l2, h2)
                            if s > score + 1e-9:
                                lo, hi, score, improved = l2, h2, s, True
            if score > best_score:
                best, best_score = (lo.copy(), hi.copy()), score
        lo, hi = best
        self.lower = (lo * 256) // nb
        self.upper = np.minimum(((hi + 1) * 256) // nb - 1, 255)
        self.train_f1 = best_score
        return self

    def __call__(self, warped, rect):
        return cv2.inRange(normalize_color_features(warped), self.lower, self.upper)

    def state(self):
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}


def pixel_features(img: np.ndarray) -> np.ndarray:
    """Per-pixel color + local-contrast features (computed at whatever resolution img is)."""
    f = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    feats = [f[..., 0], f[..., 1], f[..., 2], hsv[..., 0], hsv[..., 1], hsv[..., 2], L, a, b]
    # center-surround contrast: paint vs. surrounding pavement, invariant to global illumination
    for ch in (L, b):
        for k in (15, 61):
            feats.append(ch - cv2.blur(ch, (k, k)))
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    feats.append(cv2.blur(np.hypot(gx, gy), (5, 5)))
    feats.append(cv2.blur(L, (5, 5)) - cv2.blur(L, (5, 5)).mean())   # brightness relative to frame mean
    return np.stack(feats, -1)


class GBDTPixel(Thresholder):
    """Histogram gradient-boosted trees on color + local-contrast features (3 classes)."""
    name = "gbdt"
    needs_training = True

    def __init__(self, scale=0.5, per_frame=12000):
        self.scale, self.per_frame = scale, per_frame

    def _features(self, img):
        return pixel_features(cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA))

    def fit(self, images, labels, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier

        rng = np.random.RandomState(seed)
        X, y, w = _sample_pixels(images, labels, self.per_frame, rng, self._features)
        # square-root weights: halfway between the balanced sample and the true (very rare) yellow prior
        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                    l2_regularization=1.0, random_state=seed).fit(X, y, sample_weight=np.sqrt(w))
        return self

    def predict_proba(self, warped, rect=None):
        F = self._features(warped)
        p = self.model.predict_proba(F.reshape(-1, F.shape[-1]))
        out = np.zeros((p.shape[0], 3), np.float32)
        out[:, self.model.classes_] = p
        return out.reshape(F.shape[0], F.shape[1], 3)

    def __call__(self, warped, rect):
        p = self.predict_proba(warped)
        return to_mask(p.argmax(-1) == YELLOW, warped.shape)


# --------------------------------------------------------------------------- deep learning
def _unet(width=16):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = block(6, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.pool = nn.MaxPool2d(2)
            self.up2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), block(4 * w, 2 * w)
            self.up1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), block(2 * w, w)
            self.head = nn.Conv2d(w, 3, 1)

        def forward(self, x):
            import torch
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.head(d1)

    return TinyUNet()


def _unet_input(img, stats_from=None):
    """BGR + Lab, each standardized per image (removes global exposure / white-balance shifts).
    stats_from: (y-slice, x-slice) the mean/std are computed on (the ROI rect at inference)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    x = np.concatenate([img, lab], -1).astype(np.float32)
    ref = x[stats_from] if stats_from is not None else x
    x = (x - ref.reshape(-1, 6).mean(0)) / (ref.reshape(-1, 6).std(0) + 1e-3)
    return x.transpose(2, 0, 1)


def photometric_jitter(img, rng):
    """Brightness / contrast / gamma / white-balance jitter used as training augmentation."""
    f = img.astype(np.float32) / 255.0
    f = f ** rng.uniform(0.7, 1.4)
    f = f * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08)
    f = f * rng.uniform(0.9, 1.1, size=3)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


class TinyUNetSeg(Thresholder):
    """A ~120k-parameter U-Net (3 classes) trained on bird's-eye ROI crops at 1/4 resolution."""
    name = "unet"
    needs_training = True

    def __init__(self, scale=0.25, epochs=40, lr=3e-3, batch=8, crop=(256, 160)):
        self.scale, self.epochs, self.lr, self.batch, self.crop = scale, epochs, lr, batch, crop

    def _prep(self, img, rect=None):
        if rect is not None:
            img = img[rect_slice(rect)]
        return cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def fit(self, images, labels, seed=0, rects=None):
        import torch

        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        data = []
        for i, (img, lab) in enumerate(zip(images, labels)):
            r = rects[i] if rects else None
            small = self._prep(img, r)
            l = lab[rect_slice(r)] if r else lab
            data.append((small, cv2.resize(l, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)))
        counts = np.bincount(np.concatenate([d[1][d[1] != IGNORE].ravel() for d in data]), minlength=3)[:3]
        weight = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32)
        self.net = _unet()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(data) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE)
        cw, ch = self.crop
        self.net.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(data))
            for s in range(0, len(order), self.batch):
                xs, ys = [], []
                for i in order[s:s + self.batch]:
                    img, lab = data[i]
                    H, W = lab.shape
                    y0 = rng.randint(0, max(1, H - ch + 1)); x0 = rng.randint(0, max(1, W - cw + 1))
                    img = img[y0:y0 + ch, x0:x0 + cw]; lab = lab[y0:y0 + ch, x0:x0 + cw]
                    img = cv2.copyMakeBorder(img, 0, ch - img.shape[0], 0, cw - img.shape[1], cv2.BORDER_REFLECT)
                    lab = cv2.copyMakeBorder(lab, 0, ch - lab.shape[0], 0, cw - lab.shape[1], cv2.BORDER_CONSTANT, value=IGNORE)
                    if rng.rand() < 0.5:
                        img, lab = img[:, ::-1], lab[:, ::-1]
                    xs.append(_unet_input(photometric_jitter(np.ascontiguousarray(img), rng)))
                    ys.append(np.ascontiguousarray(lab))
                x = torch.from_numpy(np.stack(xs)); y = torch.from_numpy(np.stack(ys).astype(np.int64))
                opt.zero_grad()
                loss = loss_fn(self.net(x), y)
                loss.backward()
                opt.step(); sched.step()
        self.net.eval()
        return self

    def predict_proba(self, warped, rect):
        import torch

        small = self._prep(warped)
        H, W = small.shape[:2]
        ph, pw = (-H) % 4, (-W) % 4
        small = cv2.copyMakeBorder(small, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        stats = rect_slice([int(round(v * self.scale)) for v in rect])
        with torch.no_grad():
            p = torch.softmax(self.net(torch.from_numpy(_unet_input(small, stats))[None]), 1)[0]
        return p.numpy().transpose(1, 2, 0)[:H, :W]

    def __call__(self, warped, rect):
        return to_mask(self.predict_proba(warped, rect).argmax(-1) == YELLOW, warped.shape)


# --------------------------------------------------------------------------- persistence
def save_thresholder(t: Thresholder, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(t, LearnedHSVBox):
        path.with_suffix(".json").write_text(json.dumps({"method": t.name, **t.state()}, indent=1))
    elif isinstance(t, TinyUNetSeg):
        import torch
        torch.save({"method": t.name, "state_dict": t.net.state_dict(), "scale": t.scale}, path.with_suffix(".pt"))
    else:
        import joblib
        joblib.dump(t, path.with_suffix(".joblib"))


def load_thresholder(path: str | Path) -> Thresholder:
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        t = LearnedHSVBox(); t.lower, t.upper = np.array(d["lower"]), np.array(d["upper"])
        return t
    if path.suffix == ".pt":
        import torch
        d = torch.load(path, map_location="cpu")
        t = TinyUNetSeg(scale=d["scale"]); t.net = _unet(); t.net.load_state_dict(d["state_dict"]); t.net.eval()
        return t
    import joblib
    return joblib.load(path)


def build(name: str, **kw) -> Thresholder:
    return {"manual": ManualHSV, "otsu_lab": OtsuLab, "gmm_lab": GMMLab, "learned_box": LearnedHSVBox,
            "gbdt": GBDTPixel, "unet": TinyUNetSeg}[name](**kw)


def timed(t: Thresholder, warped, rect):
    t0 = time.perf_counter()
    m = t(warped, rect)
    return m, 1000 * (time.perf_counter() - t0)
