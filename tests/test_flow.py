"""Flow tracker: sub-pixel positions from a moving textured patch, re-seeding after occlusion."""
import cv2
import numpy as np
import pandas as pd
import pytest

from speedcam.config import FlowSettings
from speedcam.flow import FlowTracker, reference_positions

W, H = 480, 240
CAR = (60, 30)  # patch size, px


def _texture(w, h, seed, sigma=1.5):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(128, 40, (h, w)).astype(np.float32), (0, 0), sigma)
    return np.clip(img, 0, 255).astype(np.uint8)


BG = _texture(W, H, 1)
PATCH = _texture(*CAR, 2)


WIRE = _texture(12, H, 5, sigma=0.8)  # high-contrast, strongly textured, static


def _frame(cx, cy, occlude=None, wire=None):
    """Background with the patch centred at sub-pixel (cx, cy); `occlude` = (x0, x1) grey bar,
    `wire` = x of a static textured bar drawn over everything (a pole or a wire)."""
    M = np.float32([[1, 0, cx - CAR[0] / 2], [0, 1, cy - CAR[1] / 2]])
    car = cv2.warpAffine(PATCH, M, (W, H), flags=cv2.INTER_CUBIC)
    alpha = cv2.warpAffine(np.ones(CAR[::-1], np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    img = (BG * (1 - alpha) + car * alpha).astype(np.uint8)
    if occlude:
        img[:, occlude[0]:occlude[1]] = 128
    if wire is not None:
        img[:, wire:wire + WIRE.shape[1]] = WIRE
    return img


def _box(cx, cy):
    w, h = CAR
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                     [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def _run(n, v, jitter=0.0, occlude=None, seed=0, fps=60.0, wire=None):
    rng = np.random.default_rng(seed)
    ft = FlowTracker(FlowSettings(), fps)
    truth, box, out, npts = [], [], [], []
    for i in range(n):
        cx, cy = 80 + v[0] * i, 120 + v[1] * i
        occ = occlude(i) if occlude else None
        gray = _frame(cx, cy, occ, wire)
        bc = np.array([cx, cy]) + rng.normal(0, jitter, 2)
        xy, k = ft.update(gray, np.eye(3), np.array([1]), _box(*bc)[None], bc[None])
        truth.append((cx, cy))
        box.append(bc)
        out.append(xy[0])
        npts.append(k[0])
    return np.array(truth), np.array(box), np.array(out), np.array(npts)


def test_subpixel_motion_tracked():
    truth, _, out, npts = _run(120, (2.37, 0.41))
    assert np.isfinite(out[1:]).all() and (npts[1:] >= 6).all()
    err = out[1:] - truth[1:]
    assert np.abs(err).max() < 0.1
    # frame-to-frame displacement (what speed is made of) is far below a pixel
    dv = np.diff(out[1:], axis=0) - (2.37, 0.41)
    assert np.std(dv, axis=0).max() < 0.08
    assert np.abs(dv.mean(0)).max() < 0.005


def test_box_wobble_suppressed():
    truth, box, out, _ = _run(240, (1.4, 0.2), jitter=2.0, seed=3)
    s = slice(60, None)  # past the filter warm-up
    box_err = np.std(np.diff(box[s], axis=0) - (1.4, 0.2))
    flow_err = np.std(np.diff(out[s], axis=0) - (1.4, 0.2))
    assert flow_err < box_err / 10
    # anchored to the box: no drift, and the position is about as good as a ~1 s average of boxes
    # (2 px jitter / sqrt(60) ~ 0.26 px per axis)
    assert np.abs((out[s] - truth[s]).mean(0)).max() < 0.8
    assert np.sqrt(((out[s] - truth[s]) ** 2).mean()) < 1.0


def test_reseeds_after_full_occlusion():
    # grey bar covers the whole patch for frames 40-49 (box still reported, like a filled detector gap)
    def occ(i):
        return (0, W) if 40 <= i < 50 else None

    truth, _, out, npts = _run(100, (2.0, 0.0), occlude=occ)
    assert np.isnan(out[41:50]).all()  # no texture: falls back to the box centre
    assert np.isfinite(out[52:]).all() and (npts[52:] >= 6).all()  # features re-seeded
    assert np.abs(out[60:] - truth[60:]).max() < 0.15


def test_partial_occlusion_tops_up_features():
    # a pole (fixed bar) the car drives under: features under it are lost and replaced
    truth, _, out, npts = _run(140, (2.0, 0.0), occlude=lambda i: (200, 215))
    assert np.isfinite(out[1:]).mean() > 0.95
    assert npts[-10:].min() >= 0.7 * np.median(npts[5:30])  # back to about as many as before the pole
    d = np.diff(out, axis=0)[1:]
    ok = np.isfinite(d).all(1)
    assert np.std(d[ok, 0] - 2.0) < 0.1


def test_static_wire_does_not_hold_the_vehicle_back():
    # a textured wire crosses the car for ~40 frames: its corners are the strongest in the box,
    # but they don't move with the car and must not be followed
    truth, _, out, _ = _run(120, (2.0, 0.0), wire=180)
    ok = np.isfinite(out[:, 0])
    assert ok[1:].mean() > 0.8  # some frames fall back to the box while the wire hides the texture
    assert np.abs(out[ok] - truth[ok]).max() < 0.5
    d = np.diff(out[:, 0])[1:]
    assert np.nanmax(np.abs(d - 2.0)) < 0.3


def test_camera_motion_is_removed():
    """Features are mapped through the frame->reference transform: a camera pan isn't vehicle motion."""
    ft = FlowTracker(FlowSettings(), 60.0)
    out = []
    for i in range(60):
        pan = 1.3 * i  # camera pans right: everything moves left in the frame
        cx, cy = 100 + 2.0 * i, 120.0
        gray = _frame(cx - pan, cy)
        T = np.array([[1, 0, pan], [0, 1, 0], [0, 0, 1]], float)  # frame -> reference
        xy, _ = ft.update(gray, T, np.array([7]), _box(cx - pan, cy)[None], np.array([[cx, cy]]))
        out.append(xy[0])
    d = np.diff(np.array(out[1:]), axis=0)
    assert d[:, 0] == pytest.approx(2.0, abs=0.1)
    assert d[:, 0].mean() == pytest.approx(2.0, abs=0.01)


def test_reference_positions_fallback():
    df = pd.DataFrame({"rx": [1.0, 2.0], "ry": [3.0, 4.0], "frx": [1.5, np.nan], "fry": [3.5, np.nan]})
    assert reference_positions(df).tolist() == [[1.5, 3.5], [2.0, 4.0]]
    assert reference_positions(df[["rx", "ry"]]).tolist() == [[1.0, 3.0], [2.0, 4.0]]
