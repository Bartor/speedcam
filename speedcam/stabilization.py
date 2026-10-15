"""Frame → reference registration.

Frames are never warped for analysis; we only estimate the transform that maps
frame pixels to reference-frame pixels and apply it to detection points.

Default method: reference corners tracked into each frame with pyramidal
Lucas-Kanade (initialised from the previous transform), then a RANSAC
similarity/affine/homography fit. If that loses lock, ORB matching against the
reference and then against the previous frame (chained) is tried, and finally
the previous transform is reused.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import StabilizationSettings

log = logging.getLogger(__name__)


@dataclass
class Registration:
    T: np.ndarray  # 3x3 frame px -> reference px
    ok: bool
    inliers: int
    source: str  # "klt" | "reference" | "previous" | "fallback"


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    shape = pts.shape
    if pts.size == 0:
        return pts
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), T).reshape(-1, 2)
    return out.reshape(shape)


class Stabilizer:
    def __init__(self, reference_bgr: np.ndarray, settings: StabilizationSettings,
                 reference_boxes: Optional[np.ndarray] = None):
        self.s = settings
        h, w = reference_bgr.shape[:2]
        self.k = min(1.0, settings.work_width / w)
        self.S = np.diag([self.k, self.k, 1.0])
        self.S_inv = np.linalg.inv(self.S)
        if settings.features == "sift":
            self.det = cv2.SIFT_create(nfeatures=settings.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.det = cv2.ORB_create(nfeatures=settings.max_features, fastThreshold=10)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_gray = self._gray(reference_bgr)
        ref_mask = self._mask(self.ref_gray.shape, reference_boxes)
        self.ref_mask = ref_mask
        self._ref_feat = None  # descriptors computed lazily (only needed on fallback)
        self.ref_corners = cv2.goodFeaturesToTrack(self.ref_gray, settings.max_features, 0.005, 7,
                                                   mask=ref_mask, blockSize=7)
        if self.ref_corners is None or len(self.ref_corners) < settings.min_inliers:
            log.warning("Reference frame has little texture (%d corners); stabilization may fail",
                        0 if self.ref_corners is None else len(self.ref_corners))
        self.prev_T = np.eye(3)
        self.prev_feat = None
        self.n_fallback = 0

    # -- helpers ---------------------------------------------------------------
    def _gray(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.k < 1.0:
            gray = cv2.resize(gray, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_AREA)
        return gray

    def _mask(self, shape, boxes: Optional[np.ndarray]) -> np.ndarray:
        mask = np.full(shape, 255, np.uint8)
        if boxes is not None and len(boxes):
            pad = self.s.vehicle_mask_pad
            for x0, y0, x1, y1 in np.asarray(boxes, float):
                dw, dh = (x1 - x0) * pad, (y1 - y0) * pad
                cv2.rectangle(mask, (int((x0 - dw) * self.k), int((y0 - dh) * self.k)),
                              (int((x1 + dw) * self.k), int((y1 + dh) * self.k)), 0, -1)
        return mask

    def _features(self, gray: np.ndarray, mask: np.ndarray):
        kp, desc = self.det.detectAndCompute(gray, mask)
        pts = np.array([p.pt for p in kp], np.float32) if kp else np.zeros((0, 2), np.float32)
        return pts, desc

    def _fit(self, src: np.ndarray, dst: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        """Fit src->dst in working coordinates; return full-resolution 3x3 and inlier count."""
        if len(src) < 8:
            return None, 0
        thr = self.s.ransac_thresh
        if self.s.model == "similarity":
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                                 maxIters=2000, confidence=0.999, refineIters=20)
        elif self.s.model == "affine":
            M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                          maxIters=2000, confidence=0.999, refineIters=20)
        else:
            M, inl = cv2.findHomography(src, dst, cv2.RANSAC, thr, maxIters=2000, confidence=0.999)
        if M is None:
            return None, 0
        T = np.eye(3)
        if M.shape == (2, 3):
            T[:2] = M
        else:
            T = M
        return self.S_inv @ T @ self.S, int(inl.sum()) if inl is not None else 0

    def _match(self, feat, other) -> tuple[Optional[np.ndarray], int]:
        (kp, desc), (okp, odesc) = feat, other
        if desc is None or odesc is None or len(kp) < 8 or len(okp) < 8:
            return None, 0
        knn = self.matcher.knnMatch(desc, odesc, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 8:
            return None, len(good)
        return self._fit(kp[[m.queryIdx for m in good]], okp[[m.trainIdx for m in good]])

    def _klt(self, gray: np.ndarray, mask: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        if self.ref_corners is None or len(self.ref_corners) < 8:
            return None, 0
        # predict where reference corners are in this frame from the previous transform
        pred_full = transform_points(np.linalg.inv(self.prev_T), self.ref_corners.reshape(-1, 2) / self.k)
        pred = (pred_full * self.k).astype(np.float32).reshape(-1, 1, 2)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self.ref_gray, gray, self.ref_corners, pred, winSize=(21, 21), maxLevel=3,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        ok = st.reshape(-1) == 1
        p = nxt.reshape(-1, 2)
        h, w = gray.shape
        xi = np.clip(p[:, 0].astype(int), 0, w - 1)
        yi = np.clip(p[:, 1].astype(int), 0, h - 1)
        ok &= (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        ok &= mask[yi, xi] > 0  # drop points now covered by vehicles
        return self._fit(p[ok], self.ref_corners.reshape(-1, 2)[ok])

    # -- main entry ------------------------------------------------------------
    def register(self, frame_bgr: np.ndarray, vehicle_boxes: Optional[np.ndarray] = None) -> Registration:
        gray = self._gray(frame_bgr)
        mask = self._mask(gray.shape, vehicle_boxes)
        min_in = self.s.min_inliers
        reg: Optional[Registration] = None
        if self.s.features == "klt":
            T, n = self._klt(gray, mask)
            if T is not None and n >= min_in:
                reg = Registration(T, True, n, "klt")
        n_best = 0
        if reg is None:
            if self._ref_feat is None:
                self._ref_feat = self._features(self.ref_gray, self.ref_mask)
            feat = self._features(gray, mask)
            T, n = self._match(feat, self._ref_feat)
            n_best = n
            if T is not None and n >= min_in:
                reg = Registration(T, True, n, "reference")
            elif self.prev_feat is not None:
                T2, n2 = self._match(feat, self.prev_feat)
                n_best = max(n, n2)
                if T2 is not None and n2 >= min_in:
                    reg = Registration(self.prev_T @ T2, True, n2, "previous")
            self.prev_feat = feat
        else:
            self.prev_feat = None
        if reg is None:
            self.n_fallback += 1
            reg = Registration(self.prev_T.copy(), False, n_best, "fallback")
        self.prev_T = reg.T
        return reg
