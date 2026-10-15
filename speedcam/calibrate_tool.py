"""Minimal OpenCV click tool for calibration (CLI alternative to the web UI).

Keys:  s = distance segment   a = line A   b = line B   r = ROI polygon
       Enter = close ROI      u = undo last  q / Esc = save and quit
Wheel zooms around the cursor; right-drag pans.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import calibration
from .config import Calibration, ProjectConfig, Segment
from .video import FrameGrabber

WIN = "speedcam calibrate"


def run(cfg: ProjectConfig, frame_index: int) -> ProjectConfig:
    grabber = FrameGrabber(cfg.source)
    img = grabber.get(frame_index)
    grabber.close()
    cal = cfg.calibration.model_copy(deep=True)
    cal.reference_frame = frame_index
    state = {"mode": "s", "pending": [], "zoom": min(1.0, 1600 / img.shape[1]), "off": np.zeros(2),
             "pan": None, "history": []}

    def to_img(x, y):
        return (np.array([x, y]) - state["off"]) / state["zoom"]

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_MOUSEWHEEL:
            f = 1.25 if flags > 0 else 0.8
            p = to_img(x, y)
            state["zoom"] *= f
            state["off"] = np.array([x, y]) - p * state["zoom"]
        elif ev == cv2.EVENT_RBUTTONDOWN:
            state["pan"] = (np.array([x, y]), state["off"].copy())
        elif ev == cv2.EVENT_MOUSEMOVE and state["pan"] is not None:
            state["off"] = state["pan"][1] + np.array([x, y]) - state["pan"][0]
        elif ev == cv2.EVENT_RBUTTONUP:
            state["pan"] = None
        elif ev == cv2.EVENT_LBUTTONDOWN:
            p = tuple(float(v) for v in to_img(x, y))
            state["pending"].append(p)
            mode = state["mode"]
            if mode == "s" and len(state["pending"]) == 2:
                a, b = state["pending"]
                try:
                    m = float(input(f"Distance in metres for segment {len(cal.segments) + 1}: "))
                    cal.segments.append(Segment(a=a, b=b, meters=m))
                    state["history"].append("segments")
                except ValueError:
                    print("Not a number, segment discarded")
                state["pending"] = []
            elif mode in "ab" and len(state["pending"]) == 2:
                setattr(cal, f"line_{mode}", tuple(state["pending"]))
                state["history"].append(f"line_{mode}")
                state["pending"] = []

    def draw():
        z, off = state["zoom"], state["off"]
        M = np.array([[z, 0, off[0]], [0, z, off[1]]])
        view = cv2.warpAffine(img, M, (1600, 900))

        def P(p):
            return (int(p[0] * z + off[0]), int(p[1] * z + off[1]))

        for s in cal.segments:
            cv2.line(view, P(s.a), P(s.b), (0, 255, 255), 2)
            cv2.putText(view, f"{s.meters:g} m", P(s.b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for name, line, c in (("A", cal.line_a, (255, 200, 0)), ("B", cal.line_b, (255, 0, 200))):
            if line:
                cv2.line(view, P(line[0]), P(line[1]), c, 2)
                cv2.putText(view, name, P(line[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        if cal.roi:
            cv2.polylines(view, [np.array([P(p) for p in cal.roi])], True, (0, 255, 0), 1)
        for p in state["pending"]:
            cv2.circle(view, P(p), 4, (0, 0, 255), -1)
        info = f"mode: {state['mode']}  (s segment, a/b line, r ROI, Enter close ROI, u undo, q save)"
        try:
            res = calibration.compute(cal, (img.shape[1], img.shape[0]))
            info += f"   {res.m_per_px * 100:.3f} cm/px  rms {res.rms_m * 100:.1f} cm"
            if res.line_gap_m:
                info += f"  gap {res.line_gap_m:.2f} m"
        except ValueError:
            pass
        cv2.putText(view, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return view

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        cv2.imshow(WIN, draw())
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("s"), ord("a"), ord("b"), ord("r")):
            state["mode"], state["pending"] = chr(k), []
        elif k == 13 and state["mode"] == "r" and len(state["pending"]) >= 3:
            cal.roi = list(state["pending"])
            state["history"].append("roi")
            state["pending"] = []
        elif k == ord("u"):
            if state["pending"]:
                state["pending"].pop()
            elif state["history"]:
                what = state["history"].pop()
                if what == "segments":
                    cal.segments.pop()
                else:
                    setattr(cal, what, None)
    cv2.destroyAllWindows()
    cfg = cfg.model_copy(deep=True)
    cfg.calibration = cal
    return cfg
